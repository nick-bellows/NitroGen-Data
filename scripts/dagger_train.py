"""
DAgger Training Script for NitroGen

Fine-tunes the NitroGen model on collected human demonstrations.

Usage:
    python scripts/dagger_train.py --data-dir dagger_data/Celeste_20260110_120000

This will:
1. Load the base NitroGen model
2. Load your collected demonstrations
3. Fine-tune the model on your corrections
4. Save a new checkpoint
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

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

parser = argparse.ArgumentParser(description="DAgger Training")
parser.add_argument("--data-dir", type=str, required=True, help="Path to collected DAgger data")
parser.add_argument("--ckpt", type=str, default=None, help="Base checkpoint path (default: use HuggingFace cache)")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory for fine-tuned model")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder")
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
    # Find the actual checkpoint
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


class DAggerDataset(Dataset):
    """Dataset for DAgger training samples."""

    # Official NitroGen button order from HuggingFace dataset
    # Maps from official format to BUTTON_ACTION_TOKENS order
    OFFICIAL_BUTTONS = [
        'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP',
        'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER',
        'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_TRIGGER',
        'SOUTH', 'WEST', 'EAST', 'NORTH',
        'BACK', 'START', 'GUIDE'
    ]

    def __init__(self, data_dir: Path, img_processor):
        self.data_dir = data_dir
        self.img_processor = img_processor

        # Load samples
        samples_path = data_dir / "samples.json"
        if not samples_path.exists():
            raise FileNotFoundError(f"No samples.json found in {data_dir}")

        with open(samples_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} training samples")
        print(f"Using {len(BUTTON_ACTION_TOKENS)} button tokens: {BUTTON_ACTION_TOKENS}")

    def __len__(self):
        return len(self.samples)

    def _get_axis_value(self, action, key, default=0):
        """Safely extract axis value from action dict."""
        val = action.get(key, default)
        if isinstance(val, list):
            val = val[0] if val else default
        return float(val) / 32767.0  # Normalize to -1, 1

    def _get_button_value(self, action, key, is_trigger=False):
        """Safely extract button value from action dict."""
        val = action.get(key, 0)
        if isinstance(val, list):
            val = val[0] if val else 0
        if is_trigger:
            return float(val) / 255.0  # Triggers are 0-255
        return float(val)  # Binary buttons are 0 or 1

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load frame
        frame_path = sample["frame_path"]
        img = Image.open(frame_path).convert("RGB")
        pixel_values = self.img_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        # Convert action to tensor format
        action = sample["action"]

        # Build button tensor matching BUTTON_ACTION_TOKENS order (21 values)
        buttons = []
        for token in BUTTON_ACTION_TOKENS:
            is_trigger = "TRIGGER" in token
            # Handle the 4 extra tokens that may not exist in recorded data
            if token in ['RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP']:
                buttons.append(0.0)  # These are padding/unused
            else:
                buttons.append(self._get_button_value(action, token, is_trigger))

        buttons = torch.tensor(buttons, dtype=torch.float32)

        # Joysticks (normalized to -1, 1)
        j_left = torch.tensor([
            self._get_axis_value(action, "AXIS_LEFTX"),
            self._get_axis_value(action, "AXIS_LEFTY"),
        ], dtype=torch.float32)

        j_right = torch.tensor([
            self._get_axis_value(action, "AXIS_RIGHTX"),
            self._get_axis_value(action, "AXIS_RIGHTY"),
        ], dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
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

    # Freeze vision encoder if requested
    if args.freeze_vision:
        print("Freezing vision encoder...")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    model.to("cuda")
    model.train()

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Load dataset
    print("Loading dataset...")
    dataset = DAggerDataset(DATA_DIR, img_proc)

    if len(dataset) == 0:
        print("Error: No training samples found!")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    # Loss functions
    button_criterion = nn.BCEWithLogitsLoss()
    joystick_criterion = nn.MSELoss()

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()

    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to("cuda")
            target_buttons = batch["buttons"].to("cuda")
            target_j_left = batch["j_left"].to("cuda")
            target_j_right = batch["j_right"].to("cuda")

            optimizer.zero_grad()

            batch_size = pixel_values.shape[0]
            action_horizon = 16  # NitroGen uses 16-action chunks

            # Forward pass through model using flow-matching training
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Pack actions into the format expected by the model
                # NitroGen format: [buttons, j_left, j_right] with joysticks normalized to 0-1
                j_left_norm = (target_j_left + 1) / 2.0  # Convert -1,1 to 0,1
                j_right_norm = (target_j_right + 1) / 2.0

                # Combine into single action tensor [B, action_dim]
                # action_dim = 21 buttons + 2 j_left + 2 j_right = 25
                target_actions = torch.cat([target_buttons, j_left_norm, j_right_norm], dim=1)

                # Expand to action horizon (repeat the same action for all timesteps)
                # Shape: [B, horizon, action_dim]
                target_actions = target_actions.unsqueeze(1).expand(-1, action_horizon, -1).clone()

                # Create action mask (all valid)
                actions_mask = torch.ones(batch_size, action_horizon, target_actions.shape[-1],
                                         dtype=torch.bool, device="cuda")

                # Prepare data dict for model's forward pass
                # NitroGen uses 256 visual tokens per frame
                num_frames = 1
                num_visual_tokens = num_frames * 256  # 256 tokens per frame

                data = {
                    "images": pixel_values.unsqueeze(1),  # Add frame dimension [B, 1, C, H, W]
                    "dropped_images": torch.zeros(batch_size, num_frames, dtype=torch.bool, device="cuda"),
                    "vl_token_ids": torch.full((batch_size, num_visual_tokens), 1, dtype=torch.long, device="cuda"),  # IMG tokens (256 per frame)
                    "sa_token_ids": torch.full((batch_size, action_horizon), 4, dtype=torch.long, device="cuda"),  # ACT tokens
                    "vl_attn_mask": torch.ones(batch_size, num_visual_tokens, dtype=torch.bool, device="cuda"),
                    "embodiment_id": torch.zeros(batch_size, dtype=torch.long, device="cuda"),
                    "game_ids": [None] * batch_size,
                    "actions": target_actions,
                    "actions_mask": actions_mask,
                    "has_real_action": torch.ones(batch_size, dtype=torch.bool, device="cuda"),  # All samples have real actions
                }

                # Use the model's forward pass which computes flow-matching loss
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

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Extract game name from data directory
            game_name = DATA_DIR.name.split('_')[0] if '_' in DATA_DIR.name else DATA_DIR.name
            save_path = OUTPUT_DIR / f"nitrogen_{game_name.lower()}_best.pt"

            # Save in NitroGen format
            save_dict = {
                "model": model.state_dict(),
                "ckpt_config": ckpt_config.model_dump(),
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

    # Save final model
    game_name = DATA_DIR.name.split('_')[0] if '_' in DATA_DIR.name else DATA_DIR.name
    final_path = OUTPUT_DIR / f"nitrogen_{game_name.lower()}_epoch{args.epochs}.pt"
    save_dict = {
        "model": model.state_dict(),
        "ckpt_config": ckpt_config.model_dump(),
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


if __name__ == "__main__":
    train()
