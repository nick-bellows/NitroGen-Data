"""
Optimized NitroGen Inference Server

All performance optimizations enabled:
- FP16 (half precision) inference
- torch.compile() for kernel fusion
- cuDNN benchmark mode
- Disabled gradient computation

Usage:
    python scripts/serve_optimized.py <checkpoint> --timesteps 4 --fp16 --compile
"""

import zmq
import argparse
import pickle
import time

import torch

# Performance optimizations - set before loading model
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from nitrogen.inference_session import InferenceSession

def optimize_model(session, use_fp16=True, use_compile=False):
    """Apply optimizations to the model."""
    model = session.model

    optimizations = []

    # FP16 (Half Precision)
    if use_fp16:
        try:
            model = model.half()
            session.model = model
            optimizations.append("FP16")
        except Exception as e:
            print(f"Warning: FP16 failed: {e}")

    # torch.compile (PyTorch 2.0+)
    if use_compile:
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                session.model = model
                optimizations.append("torch.compile")
            else:
                print("Warning: torch.compile not available (requires PyTorch 2.0+)")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    return optimizations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Model Inference Server")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--ctx", type=int, default=1, help="Context length")
    parser.add_argument("--timesteps", type=int, default=4, help="Inference timesteps (default 4, try 2 for max speed)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 (half precision)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    print("=" * 60)
    print("NitroGen Optimized Server")
    print("=" * 60)

    # Load model
    print(f"\nLoading checkpoint: {args.ckpt}")
    load_start = time.time()

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

    print(f"\nOptimizations enabled: {', '.join(optimizations) if optimizations else 'None'}")
    print(f"Inference timesteps: {args.timesteps}")
    print(f"cuDNN benchmark: True")
    print(f"Gradients disabled: True")

    # Warmup inference
    print("\nWarming up model...")
    try:
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        for _ in range(3):
            _ = session.predict(dummy_img)
        print("Warmup complete!")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Create poller
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Server running on port {args.port}")
    print(f"Optimizations: FP16={use_fp16}, Compiled={args.compile}, Timesteps={args.timesteps}")
    print(f"Waiting for requests...")
    print(f"{'='*60}\n")

    inference_count = 0
    total_inference_time = 0.0

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                request = socket.recv()
                request = pickle.loads(request)

                if request["type"] == "reset":
                    session.reset()
                    response = {"status": "ok"}

                elif request["type"] == "info":
                    info = session.info()
                    info["optimizations"] = {
                        "fp16": use_fp16,
                        "compiled": args.compile,
                        "timesteps": args.timesteps,
                    }
                    response = {"status": "ok", "info": info}

                elif request["type"] == "predict":
                    raw_image = request["image"]

                    inf_start = time.time()
                    result = session.predict(raw_image)
                    inf_time = time.time() - inf_start

                    inference_count += 1
                    total_inference_time += inf_time

                    # Print stats every 10 inferences
                    if inference_count % 10 == 0:
                        avg_time = total_inference_time / inference_count
                        print(f"Inferences: {inference_count} | Avg: {avg_time*1000:.1f}ms | Last: {inf_time*1000:.1f}ms")

                    response = {
                        "status": "ok",
                        "pred": result
                    }
                else:
                    response = {"status": "error", "message": f"Unknown request type: {request['type']}"}

                socket.send(pickle.dumps(response))

    except KeyboardInterrupt:
        print("\nShutting down server...")
        if inference_count > 0:
            avg_time = total_inference_time / inference_count
            print(f"Final stats: {inference_count} inferences, {avg_time*1000:.1f}ms avg")
        exit(0)
    finally:
        socket.close()
        context.term()
