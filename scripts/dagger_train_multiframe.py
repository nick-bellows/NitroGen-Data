"""
DAgger Training Script with Frame Stacking for NitroGen

Fine-tunes the NitroGen model using 4-frame temporal context.
This gives the model awareness of motion and direction.

Usage:
    python scripts/dagger_train_multiframe.py --data-dir dagger_data/Hades_20260110

The data directory should contain frames and samples.json.
Frames will be loaded in sequences of 4 for temporal context.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig, NitrogenTokenizer
from nitrogen.cfg import CkptConfig
from nitrogen.shared import PATH_REPO, BUTTON_ACTION_TOKENS

parser = argparse.ArgumentParser(description="DAgger Training with Frame Stacking")
parser.add_argument("--data-dir", type=str, required=True, help="Path to collected DAgger data")
parser.add_argument("--ckpt", type=str, default=None, help="Base checkpoint path (default: use HuggingFace cache)")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory for fine-tuned model")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder")
parser.add_argument("--num-frames", type=int, default=4, help="Number of frames for temporal context")
parser.add_argument("--frame-skip", type=int, default=1, help="Skip every N frames (1=consecutive)")
args = parser.parse_args()

# Setup paths
DATA_DIR = Path(args.data_dir)
if not DATA_DIR.exists():
    print(f"Error: Data directory not found: {DATA_DIR}")
    sys.exit(1)

if args.ckpt:
    CKPT_PATH = Path(args.ckpt)
else:
    # Default HuggingFace cache location
    CKPT_PATH = Path.home() / ".cache/huggingface/hub/models--nvidia--NitroGen/snapshots"
    if CKPT_PATH.exists():
        snapshots = list(CKPT_PATH.iterdir())
        if snapshots:
            CKPT_PATH = snapshots[0] / "ng.pt"

if not CKPT_PATH.exists():
    print(f"Error: Checkpoint not found: {CKPT_PATH}")
    print("Please provide --ckpt path to your NitroGen checkpoint")
    sys.exit(1)

OUTPUT_DIR = Path(args.output_dir) if args.output_dir else PATH_REPO / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Base checkpoint: {CKPT_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Frame stacking: {args.num_frames} frames (skip={args.frame_skip})")


class MultiFrameDataset(Dataset):
    """Dataset for training with frame stacking (temporal context).

    Each sample returns a sequence of N frames along with the action
    for the last frame. This gives the model temporal awareness.
    """

    OFFICIAL_BUTTONS = [
        'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP',
        'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER',
        'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_TRIGGER',
        'SOUTH', 'WEST', 'EAST', 'NORTH',
        'BACK', 'START', 'GUIDE'
    ]

    def __init__(self, data_dir: Path, img_processor, num_frames: int = 4, frame_skip: int = 1):
        self.data_dir = data_dir
        self.img_processor = img_processor
        self.num_frames = num_frames
        self.frame_skip = frame_skip

        # Determine project root (for resolving relative paths in samples.json)
        # samples.json may contain paths relative to project root
        self.project_root = PATH_REPO

        # Load samples
        samples_path = data_dir / "samples.json"
        if not samples_path.exists():
            raise FileNotFoundError(f"No samples.json found in {data_dir}")

        with open(samples_path) as f:
            all_samples = json.load(f)

        # Group samples by session (assuming frame paths have session info)
        # We need to track frame sequences within each recording session
        self.sessions = self._group_by_session(all_samples)

        # Create valid indices (samples that have enough history)
        self.valid_indices = []
        required_history = (num_frames - 1) * frame_skip

        for session_id, session_samples in self.sessions.items():
            for i in range(len(session_samples)):
                if i >= required_history:
                    self.valid_indices.append((session_id, i))

        print(f"Loaded {len(all_samples)} total samples")
        print(f"Valid samples with {num_frames}-frame history: {len(self.valid_indices)}")
        print(f"Sessions detected: {len(self.sessions)}")

    def _group_by_session(self, samples):
        """Group samples by recording session based on frame paths."""
        sessions = {}

        for sample in samples:
            frame_path = Path(sample["frame_path"])
            # Extract session from parent directory or filename pattern
            # Format: games/hades/recordings/Hades_20260110_185339/frames/000000.png
            # We want the session to be the recording directory name
            parts = frame_path.parts
            if "frames" in parts:
                frames_idx = parts.index("frames")
                if frames_idx > 0:
                    session_id = parts[frames_idx - 1]  # Directory before "frames"
                else:
                    session_id = "default"
            else:
                session_id = "default"

            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(sample)

        # Sort each session by frame_id (most reliable) or by filename
        for session_id in sessions:
            sessions[session_id].sort(key=lambda s: s.get("frame_id", int(Path(s["frame_path"]).stem)))

        return sessions

    def __len__(self):
        return len(self.valid_indices)

    def _get_axis_value(self, action, key, default=0):
        val = action.get(key, default)
        if isinstance(val, list):
            val = val[0] if val else default
        return float(val) / 32767.0

    def _get_button_value(self, action, key, is_trigger=False):
        val = action.get(key, 0)
        if isinstance(val, list):
            val = val[0] if val else 0
        if is_trigger:
            return float(val) / 255.0
        return float(val)

    def __getitem__(self, idx):
        session_id, sample_idx = self.valid_indices[idx]
        session_samples = self.sessions[session_id]

        # Get the sequence of frames
        frame_indices = []
        for i in range(self.num_frames):
            frame_idx = sample_idx - (self.num_frames - 1 - i) * self.frame_skip
            frame_indices.append(frame_idx)

        # Load all frames in the sequence
        frames = []
        for frame_idx in frame_indices:
            sample = session_samples[frame_idx]
            frame_path = sample["frame_path"]
            # Resolve path - could be absolute or relative to project root
            frame_path = Path(frame_path)
            if not frame_path.is_absolute():
                frame_path = self.project_root / frame_path
            img = Image.open(frame_path).convert("RGB")
            pixel_values = self.img_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            frames.append(pixel_values)

        # Stack frames: [num_frames, C, H, W]
        stacked_frames = torch.stack(frames, dim=0)

        # Get the action for the LAST frame (current action to predict)
        current_sample = session_samples[sample_idx]
        action = current_sample["action"]

        # Build button tensor
        buttons = []
        for token in BUTTON_ACTION_TOKENS:
            is_trigger = "TRIGGER" in token
            if token in ['RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP']:
                buttons.append(0.0)
            else:
                buttons.append(self._get_button_value(action, token, is_trigger))

        buttons = torch.tensor(buttons, dtype=torch.float32)

        j_left = torch.tensor([
            self._get_axis_value(action, "AXIS_LEFTX"),
            self._get_axis_value(action, "AXIS_LEFTY"),
        ], dtype=torch.float32)

        j_right = torch.tensor([
            self._get_axis_value(action, "AXIS_RIGHTX"),
            self._get_axis_value(action, "AXIS_RIGHTY"),
        ], dtype=torch.float32)

        return {
            "pixel_values": stacked_frames,  # [num_frames, C, H, W]
            "buttons": buttons,
            "j_left": j_left,
            "j_right": j_right,
        }


def load_model(checkpoint_path: str):
    """Load NitroGen model for training."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    model_cfg = ckpt_config.model_cfg
    tokenizer_cfg = ckpt_config.tokenizer_cfg

    img_proc = AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)

    if isinstance(model_cfg, NitroGen_Config):
        tokenizer_cfg.training = True
        tokenizer = NitrogenTokenizer(tokenizer_cfg)
        game_mapping = tokenizer.game_mapping
        model = NitroGen(config=model_cfg, game_mapping=game_mapping)
    else:
        raise ValueError(f"Unsupported model config type: {type(model_cfg)}")

    model.load_state_dict(checkpoint["model"])

    return model, tokenizer, img_proc, ckpt_config


def train():
    print("Loading model...")
    model, tokenizer, img_proc, ckpt_config = load_model(str(CKPT_PATH))

    if args.freeze_vision:
        print("Freezing vision encoder...")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    model.to("cuda")
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    print("Loading dataset with frame stacking...")
    dataset = MultiFrameDataset(
        DATA_DIR,
        img_proc,
        num_frames=args.num_frames,
        frame_skip=args.frame_skip
    )

    if len(dataset) == 0:
        print("Error: No valid training samples with frame history!")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Frame stacking: {args.num_frames} frames")
    print()

    best_loss = float('inf')
    num_frames = args.num_frames
    num_visual_tokens = num_frames * 256  # 256 tokens per frame

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            # pixel_values shape: [B, num_frames, C, H, W]
            pixel_values = batch["pixel_values"].to("cuda")
            target_buttons = batch["buttons"].to("cuda")
            target_j_left = batch["j_left"].to("cuda")
            target_j_right = batch["j_right"].to("cuda")

            optimizer.zero_grad()

            batch_size = pixel_values.shape[0]
            action_horizon = 16

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                j_left_norm = (target_j_left + 1) / 2.0
                j_right_norm = (target_j_right + 1) / 2.0

                target_actions = torch.cat([target_buttons, j_left_norm, j_right_norm], dim=1)
                target_actions = target_actions.unsqueeze(1).expand(-1, action_horizon, -1).clone()

                actions_mask = torch.ones(batch_size, action_horizon, target_actions.shape[-1],
                                         dtype=torch.bool, device="cuda")

                # Create dropped_images tensor (none dropped)
                dropped_images = torch.zeros(batch_size, num_frames, dtype=torch.bool, device="cuda")

                data = {
                    "images": pixel_values,  # [B, num_frames, C, H, W] - already correct shape
                    "dropped_images": dropped_images,
                    "vl_token_ids": torch.full((batch_size, num_visual_tokens), 1, dtype=torch.long, device="cuda"),
                    "sa_token_ids": torch.full((batch_size, action_horizon), 4, dtype=torch.long, device="cuda"),
                    "vl_attn_mask": torch.ones(batch_size, num_visual_tokens, dtype=torch.bool, device="cuda"),
                    "embodiment_id": torch.zeros(batch_size, dtype=torch.long, device="cuda"),
                    "game_ids": [None] * batch_size,
                    "actions": target_actions,
                    "actions_mask": actions_mask,
                    "has_real_action": torch.ones(batch_size, dtype=torch.bool, device="cuda"),
                }

                output = model(data)
                loss = output["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })

        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            game_name = DATA_DIR.name.split('_')[0] if '_' in DATA_DIR.name else DATA_DIR.name
            save_path = OUTPUT_DIR / f"nitrogen_{game_name.lower()}_multiframe_best.pt"

            save_dict = {
                "model": model.state_dict(),
                "ckpt_config": ckpt_config.model_dump(),
                "multiframe_config": {
                    "num_frames": args.num_frames,
                    "frame_skip": args.frame_skip,
                },
                "dagger_info": {
                    "base_ckpt": str(CKPT_PATH),
                    "data_dir": str(DATA_DIR),
                    "epochs": epoch + 1,
                    "best_loss": best_loss,
                    "timestamp": datetime.now().isoformat(),
                }
            }
            torch.save(save_dict, save_path)
            print(f"Saved best model to {save_path}")

    game_name = DATA_DIR.name.split('_')[0] if '_' in DATA_DIR.name else DATA_DIR.name
    final_path = OUTPUT_DIR / f"nitrogen_{game_name.lower()}_multiframe_epoch{args.epochs}.pt"
    save_dict = {
        "model": model.state_dict(),
        "ckpt_config": ckpt_config.model_dump(),
        "multiframe_config": {
            "num_frames": args.num_frames,
            "frame_skip": args.frame_skip,
        },
        "dagger_info": {
            "base_ckpt": str(CKPT_PATH),
            "data_dir": str(DATA_DIR),
            "epochs": args.epochs,
            "final_loss": avg_loss,
            "timestamp": datetime.now().isoformat(),
        }
    }
    torch.save(save_dict, final_path)
    print(f"Saved final model to {final_path}")

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {OUTPUT_DIR}")
    print(f"\nTo use this model for inference:")
    print(f"  python scripts/serve_multiframe.py --ckpt {save_path} --ctx {args.num_frames}")


if __name__ == "__main__":
    train()
