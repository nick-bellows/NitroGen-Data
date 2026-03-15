"""
DAgger Training with Weighted Corrections

Trains on both original data and DAgger corrections, with human
corrections weighted higher to prioritize learning from corrections.

Usage:
    python scripts/dagger_train_weighted.py \
        --base-data games/hades/recordings/Hades_20260110_185339 \
        --corrections games/hades/dagger_sessions/dagger_20260111_120000 \
        --correction-weight 2.0 \
        --epochs 5

The correction weight determines how much more important human
corrections are vs. the original data. A weight of 2.0 means each
correction sample counts as 2 original samples.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig, NitrogenTokenizer
from nitrogen.cfg import CkptConfig
from nitrogen.shared import PATH_REPO, BUTTON_ACTION_TOKENS

parser = argparse.ArgumentParser(description="DAgger Training with Weighted Corrections")
parser.add_argument("--base-data", type=str, default=None, help="Path to base training data")
parser.add_argument("--corrections", type=str, required=True, help="Path to DAgger corrections")
parser.add_argument("--ckpt", type=str, default=None, help="Base checkpoint path")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--correction-weight", type=float, default=2.0, help="Weight for correction samples")
parser.add_argument("--human-only", action="store_true", help="Only train on human corrections, ignore AI frames")
parser.add_argument("--num-frames", type=int, default=4, help="Number of frames for temporal context")
parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder")
args = parser.parse_args()


class DAggerDataset(Dataset):
    """Dataset that loads samples with optional weighting."""

    def __init__(self, data_dir: Path, img_processor, weight: float = 1.0,
                 human_only: bool = False, num_frames: int = 4):
        self.data_dir = Path(data_dir)
        self.img_processor = img_processor
        self.weight = weight
        self.human_only = human_only
        self.num_frames = num_frames
        self.project_root = PATH_REPO

        # Load samples
        samples_path = self.data_dir / "samples.json"
        if not samples_path.exists():
            raise FileNotFoundError(f"No samples.json found in {data_dir}")

        with open(samples_path) as f:
            all_samples = json.load(f)

        # Filter for human-only if requested
        if human_only:
            all_samples = [s for s in all_samples if s.get("is_human", True)]

        # Group by session for frame stacking
        self.samples = all_samples
        self.valid_indices = self._compute_valid_indices()

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"  Valid sequences: {len(self.valid_indices)}")
        print(f"  Weight: {weight}")
        if human_only:
            print(f"  Human-only mode: Yes")

    def _compute_valid_indices(self):
        """Find indices that have enough frame history."""
        # For now, simple sequential check
        required_history = self.num_frames - 1
        valid = []
        for i in range(len(self.samples)):
            if i >= required_history:
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def get_weight(self):
        return self.weight

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
        sample_idx = self.valid_indices[idx]

        # Load frame sequence
        frames = []
        for i in range(self.num_frames):
            frame_idx = sample_idx - (self.num_frames - 1 - i)
            sample = self.samples[frame_idx]

            frame_path = Path(sample["frame_path"])
            if not frame_path.is_absolute():
                frame_path = self.project_root / frame_path

            img = Image.open(frame_path).convert("RGB")
            pixel_values = self.img_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            frames.append(pixel_values)

        stacked_frames = torch.stack(frames, dim=0)

        # Get action for current frame
        current_sample = self.samples[sample_idx]
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
            "pixel_values": stacked_frames,
            "buttons": buttons,
            "j_left": j_left,
            "j_right": j_right,
            "weight": torch.tensor(self.weight, dtype=torch.float32),
            "is_human": torch.tensor(current_sample.get("is_human", True), dtype=torch.bool),
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


def find_checkpoint():
    """Find the best available checkpoint."""
    # Priority order
    candidates = [
        PATH_REPO / "checkpoints" / "nitrogen_hades_multiframe_best.pt",
        PATH_REPO / "checkpoints" / "nitrogen_hades_best.pt",
    ]

    for ckpt in candidates:
        if ckpt.exists():
            return ckpt

    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache/huggingface/hub/models--nvidia--NitroGen/snapshots"
    if hf_cache.exists():
        snapshots = list(hf_cache.iterdir())
        if snapshots:
            return snapshots[0] / "ng.pt"

    return None


def train():
    # Find checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_checkpoint()
        if ckpt_path is None:
            print("Error: No checkpoint found. Provide --ckpt path.")
            sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    model, tokenizer, img_proc, ckpt_config = load_model(str(ckpt_path))

    if args.freeze_vision:
        print("Freezing vision encoder...")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    model.to("cuda")
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Load datasets
    datasets = []

    # Load base data if provided
    if args.base_data:
        base_dataset = DAggerDataset(
            Path(args.base_data),
            img_proc,
            weight=1.0,
            human_only=False,
            num_frames=args.num_frames,
        )
        datasets.append(base_dataset)
        print(f"\nBase data: {len(base_dataset)} samples (weight=1.0)")

    # Load corrections
    corrections_path = Path(args.corrections)
    if corrections_path.is_dir():
        # Single session or multiple?
        if (corrections_path / "samples.json").exists():
            # Single session
            correction_dataset = DAggerDataset(
                corrections_path,
                img_proc,
                weight=args.correction_weight,
                human_only=args.human_only,
                num_frames=args.num_frames,
            )
            datasets.append(correction_dataset)
            print(f"Corrections: {len(correction_dataset)} samples (weight={args.correction_weight})")
        else:
            # Multiple sessions
            for session_dir in sorted(corrections_path.iterdir()):
                if session_dir.is_dir() and (session_dir / "samples.json").exists():
                    correction_dataset = DAggerDataset(
                        session_dir,
                        img_proc,
                        weight=args.correction_weight,
                        human_only=args.human_only,
                        num_frames=args.num_frames,
                    )
                    datasets.append(correction_dataset)
                    print(f"  Session {session_dir.name}: {len(correction_dataset)} samples")

    if not datasets:
        print("Error: No training data found!")
        sys.exit(1)

    # Combine datasets
    combined_dataset = ConcatDataset(datasets)
    print(f"\nTotal samples: {len(combined_dataset)}")

    # Create weighted sampler
    weights = []
    for dataset in datasets:
        weight = dataset.get_weight()
        weights.extend([weight] * len(dataset))

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined_dataset),
        replacement=True,
    )

    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else PATH_REPO / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting DAgger training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Correction weight: {args.correction_weight}")
    print(f"Frame stacking: {args.num_frames} frames")
    print()

    best_loss = float('inf')
    num_frames = args.num_frames
    num_visual_tokens = num_frames * 256

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        human_samples = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to("cuda")
            target_buttons = batch["buttons"].to("cuda")
            target_j_left = batch["j_left"].to("cuda")
            target_j_right = batch["j_right"].to("cuda")
            is_human = batch["is_human"]

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

                dropped_images = torch.zeros(batch_size, num_frames, dtype=torch.bool, device="cuda")

                data = {
                    "images": pixel_values,
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
            human_samples += is_human.sum().item()
            total_samples += batch_size

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "human%": f"{human_samples/total_samples*100:.0f}%",
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Human samples: {human_samples/total_samples*100:.1f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = output_dir / "nitrogen_hades_dagger_best.pt"

            save_dict = {
                "model": model.state_dict(),
                "ckpt_config": ckpt_config.model_dump(),
                "multiframe_config": {
                    "num_frames": args.num_frames,
                },
                "dagger_info": {
                    "base_ckpt": str(ckpt_path),
                    "base_data": args.base_data,
                    "corrections": args.corrections,
                    "correction_weight": args.correction_weight,
                    "epochs": epoch + 1,
                    "best_loss": best_loss,
                    "timestamp": datetime.now().isoformat(),
                }
            }
            torch.save(save_dict, save_path)
            print(f"Saved best model to {save_path}")

    # Save final model
    final_path = output_dir / f"nitrogen_hades_dagger_epoch{args.epochs}.pt"
    save_dict = {
        "model": model.state_dict(),
        "ckpt_config": ckpt_config.model_dump(),
        "multiframe_config": {
            "num_frames": args.num_frames,
        },
        "dagger_info": {
            "base_ckpt": str(ckpt_path),
            "base_data": args.base_data,
            "corrections": args.corrections,
            "correction_weight": args.correction_weight,
            "epochs": args.epochs,
            "final_loss": avg_loss,
            "timestamp": datetime.now().isoformat(),
        }
    }
    torch.save(save_dict, final_path)
    print(f"Saved final model to {final_path}")

    print("\nDAgger training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\nTo use this model:")
    print(f"  python scripts/serve_multiframe.py --ckpt {save_path} --ctx {args.num_frames}")


if __name__ == "__main__":
    train()
