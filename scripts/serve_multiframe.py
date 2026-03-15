"""
NitroGen Multi-Frame Inference Server

Serves a model trained with frame stacking for temporal awareness.
Uses a 4-frame context by default to capture motion information.

Usage:
    python scripts/serve_multiframe.py --ckpt checkpoints/nitrogen_hades_multiframe_best.pt

This server maintains a rolling buffer of frames and provides all 4
frames to the model for each prediction, enabling temporal reasoning.
"""

import zmq
import argparse
import pickle
import time
from collections import deque

import torch
from PIL import Image
import numpy as np

# Performance optimizations
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from nitrogen.inference_session import InferenceSession


def optimize_model(session, use_fp16=True, use_compile=False):
    """Apply optimizations to the model."""
    model = session.model
    optimizations = []

    if use_fp16:
        try:
            model = model.half()
            session.model = model
            optimizations.append("FP16")
        except Exception as e:
            print(f"Warning: FP16 failed: {e}")

    if use_compile:
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                session.model = model
                optimizations.append("torch.compile")
            else:
                print("Warning: torch.compile not available")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    return optimizations


class MultiFrameServer:
    """Server that maintains frame history for temporal context."""

    def __init__(self, session, num_frames=4):
        self.session = session
        self.num_frames = num_frames
        self.frame_buffer = deque(maxlen=num_frames)
        self.inference_count = 0
        self.total_inference_time = 0.0

    def reset(self):
        """Reset frame buffer and session."""
        self.frame_buffer.clear()
        self.session.reset()

    def predict(self, image):
        """Run prediction with frame history context.

        Maintains a rolling buffer of frames and passes the
        full history to the model for temporal awareness.
        """
        inf_start = time.time()

        # Add current frame to buffer
        self.frame_buffer.append(image)

        # Use the session's built-in multi-frame support
        # The session already maintains obs_buffer internally
        result = self.session.predict(image)

        inf_time = time.time() - inf_start
        self.inference_count += 1
        self.total_inference_time += inf_time

        return result, inf_time

    def get_stats(self):
        """Get inference statistics."""
        if self.inference_count == 0:
            return {"count": 0, "avg_ms": 0, "fps": 0}

        avg_time = self.total_inference_time / self.inference_count
        return {
            "count": self.inference_count,
            "avg_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "buffer_size": len(self.frame_buffer),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Frame Inference Server")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to multi-frame checkpoint")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--ctx", type=int, default=4, help="Frame context length (default: 4)")
    parser.add_argument("--timesteps", type=int, default=4, help="Inference timesteps")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    args = parser.parse_args()

    print("=" * 60)
    print("NitroGen Multi-Frame Server")
    print("=" * 60)
    print(f"Frame context: {args.ctx} frames")
    print()

    # Load checkpoint and check for multiframe config
    print(f"Loading checkpoint: {args.ckpt}")
    load_start = time.time()

    # Check if checkpoint has multiframe config
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "multiframe_config" in checkpoint:
        mf_config = checkpoint["multiframe_config"]
        print(f"Checkpoint trained with {mf_config['num_frames']} frames (skip={mf_config['frame_skip']})")
        if args.ctx != mf_config["num_frames"]:
            print(f"Warning: Using --ctx={args.ctx} but model was trained with {mf_config['num_frames']} frames")

    session = InferenceSession.from_ckpt(
        args.ckpt,
        old_layout=args.old_layout,
        cfg_scale=args.cfg,
        context_length=args.ctx,
        num_inference_timesteps=args.timesteps
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Apply optimizations
    use_fp16 = args.fp16 and not args.no_fp16
    optimizations = optimize_model(session, use_fp16=use_fp16, use_compile=args.compile)

    print(f"\nOptimizations: {', '.join(optimizations) if optimizations else 'None'}")
    print(f"Inference timesteps: {args.timesteps}")

    # Create multi-frame server
    server = MultiFrameServer(session, num_frames=args.ctx)

    # Warmup
    print("\nWarming up model...")
    try:
        dummy_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        for _ in range(args.ctx + 2):  # Fill buffer and run a few inferences
            _ = server.predict(dummy_img)
        server.reset()  # Clear warmup state
        print("Warmup complete!")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Multi-Frame Server running on port {args.port}")
    print(f"Frame context: {args.ctx} | FP16: {use_fp16} | Timesteps: {args.timesteps}")
    print(f"Waiting for requests...")
    print(f"{'='*60}\n")

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                request = socket.recv()
                request = pickle.loads(request)

                if request["type"] == "reset":
                    server.reset()
                    response = {"status": "ok"}

                elif request["type"] == "info":
                    info = session.info()
                    info["multiframe"] = {
                        "num_frames": args.ctx,
                        "buffer_size": len(server.frame_buffer),
                    }
                    info["stats"] = server.get_stats()
                    response = {"status": "ok", "info": info}

                elif request["type"] == "predict":
                    raw_image = request["image"]
                    result, inf_time = server.predict(raw_image)

                    # Print stats every 10 inferences
                    if server.inference_count % 10 == 0:
                        stats = server.get_stats()
                        print(f"Inferences: {stats['count']} | "
                              f"Avg: {stats['avg_ms']:.1f}ms | "
                              f"Buffer: {stats['buffer_size']}/{args.ctx}")

                    response = {
                        "status": "ok",
                        "pred": result,
                        "inference_time_ms": inf_time * 1000,
                    }
                else:
                    response = {"status": "error", "message": f"Unknown request type: {request['type']}"}

                socket.send(pickle.dumps(response))

    except KeyboardInterrupt:
        print("\nShutting down server...")
        stats = server.get_stats()
        if stats["count"] > 0:
            print(f"Final stats: {stats['count']} inferences, {stats['avg_ms']:.1f}ms avg, {stats['fps']:.1f} FPS")
        exit(0)
    finally:
        socket.close()
        context.term()
