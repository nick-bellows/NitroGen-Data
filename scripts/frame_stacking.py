"""
Frame Stacking Utilities for NitroGen

Provides utilities for multi-frame temporal context:
- FrameBuffer: Rolling buffer for maintaining frame history
- FrameStackDataset: Dataset wrapper for loading frame sequences
- channel_stack: Stack frames along channel dimension (12ch from 4x3ch)

Usage:
    from frame_stacking import FrameBuffer, channel_stack

    buffer = FrameBuffer(num_frames=4)
    buffer.add(current_frame)
    stacked = buffer.get_stacked()  # [4, C, H, W] or [12, H, W] if channel_stacked
"""

import torch
from collections import deque
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image


class FrameBuffer:
    """Rolling buffer that maintains a fixed-size history of frames.

    Useful for inference where we need to maintain temporal context
    across multiple prediction calls.
    """

    def __init__(self, num_frames: int = 4, fill_mode: str = "repeat"):
        """
        Args:
            num_frames: Number of frames to maintain in history
            fill_mode: How to fill buffer when not enough frames:
                - "repeat": Repeat the first available frame
                - "zero": Fill with zeros
        """
        self.num_frames = num_frames
        self.fill_mode = fill_mode
        self.buffer = deque(maxlen=num_frames)

    def add(self, frame: torch.Tensor):
        """Add a frame to the buffer.

        Args:
            frame: Frame tensor of shape [C, H, W]
        """
        self.buffer.append(frame)

    def get_stacked(self, channel_stack: bool = False) -> torch.Tensor:
        """Get all frames as a stacked tensor.

        Args:
            channel_stack: If True, stack along channel dim [C*N, H, W]
                          If False, stack as separate dimension [N, C, H, W]

        Returns:
            Stacked frames tensor
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # Get available frames
        frames = list(self.buffer)

        # Fill missing frames
        while len(frames) < self.num_frames:
            if self.fill_mode == "repeat":
                # Repeat the oldest frame
                frames.insert(0, frames[0].clone())
            else:
                # Fill with zeros
                frames.insert(0, torch.zeros_like(frames[0]))

        # Stack frames
        stacked = torch.stack(frames, dim=0)  # [N, C, H, W]

        if channel_stack:
            # Reshape to [N*C, H, W]
            n, c, h, w = stacked.shape
            stacked = stacked.view(n * c, h, w)

        return stacked

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return len(self.buffer) >= self.num_frames


def channel_stack_frames(frames: torch.Tensor) -> torch.Tensor:
    """Stack frames along the channel dimension.

    Converts [N, C, H, W] to [N*C, H, W] for channel-stacked processing.

    Args:
        frames: Tensor of shape [N, C, H, W]

    Returns:
        Tensor of shape [N*C, H, W]
    """
    if frames.dim() == 4:
        n, c, h, w = frames.shape
        return frames.view(n * c, h, w)
    elif frames.dim() == 5:  # Batch dimension
        b, n, c, h, w = frames.shape
        return frames.view(b, n * c, h, w)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {frames.dim()}D")


def create_frame_mask(num_frames: int, available_frames: int, device="cuda") -> torch.Tensor:
    """Create a mask indicating which frames are real vs padded.

    Args:
        num_frames: Total number of frame slots
        available_frames: Number of actual frames (rest are padded)
        device: Device for the tensor

    Returns:
        Boolean tensor where True = frame is padded/dropped
    """
    mask = torch.zeros(num_frames, dtype=torch.bool, device=device)
    num_padded = num_frames - available_frames
    if num_padded > 0:
        mask[:num_padded] = True
    return mask


class FrameSequenceLoader:
    """Helper for loading frame sequences from a directory.

    Useful for preprocessing existing recordings into multi-frame format.
    """

    def __init__(self, frame_dir, num_frames: int = 4, frame_skip: int = 1):
        """
        Args:
            frame_dir: Directory containing frame images
            num_frames: Number of frames per sequence
            frame_skip: Skip every N frames (1 = consecutive)
        """
        self.frame_dir = frame_dir
        self.num_frames = num_frames
        self.frame_skip = frame_skip

        # Find all frames
        from pathlib import Path
        self.frames = sorted(Path(frame_dir).glob("*.png"))
        if not self.frames:
            self.frames = sorted(Path(frame_dir).glob("*.jpg"))

    def __len__(self):
        """Number of valid sequences (with enough history)."""
        required = (self.num_frames - 1) * self.frame_skip
        return max(0, len(self.frames) - required)

    def get_sequence(self, idx: int) -> List[Image.Image]:
        """Get a sequence of frames ending at the given index.

        Args:
            idx: Index of the final (current) frame

        Returns:
            List of PIL Images
        """
        required = (self.num_frames - 1) * self.frame_skip
        actual_idx = idx + required  # Shift to account for history

        frames = []
        for i in range(self.num_frames):
            frame_idx = actual_idx - (self.num_frames - 1 - i) * self.frame_skip
            img = Image.open(self.frames[frame_idx]).convert("RGB")
            frames.append(img)

        return frames


def compute_optical_flow_simple(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute simple frame difference as a proxy for motion.

    For actual optical flow, use cv2.calcOpticalFlowFarneback.
    This is a simple alternative that doesn't require OpenCV.

    Args:
        frame1: Previous frame [H, W, C]
        frame2: Current frame [H, W, C]

    Returns:
        Motion magnitude [H, W]
    """
    diff = np.abs(frame2.astype(float) - frame1.astype(float))
    motion = np.mean(diff, axis=-1)  # Average across channels
    return motion


def analyze_temporal_coherence(frames: List[torch.Tensor]) -> dict:
    """Analyze temporal coherence of a frame sequence.

    Useful for debugging and understanding if frame stacking is working.

    Args:
        frames: List of frame tensors [C, H, W]

    Returns:
        Dict with coherence metrics
    """
    if len(frames) < 2:
        return {"num_frames": len(frames), "motion": 0.0}

    # Compute average motion between consecutive frames
    motions = []
    for i in range(1, len(frames)):
        prev = frames[i - 1].cpu().numpy().transpose(1, 2, 0)
        curr = frames[i].cpu().numpy().transpose(1, 2, 0)
        motion = compute_optical_flow_simple(prev, curr)
        motions.append(motion.mean())

    return {
        "num_frames": len(frames),
        "avg_motion": np.mean(motions),
        "max_motion": np.max(motions),
        "min_motion": np.min(motions),
    }


# Convenience functions for model input preparation
def prepare_multiframe_input(
    frames: List[torch.Tensor],
    num_frames: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare multi-frame input for the model.

    Args:
        frames: List of available frames [C, H, W]
        num_frames: Target number of frames
        device: Device to place tensors

    Returns:
        Tuple of (stacked_frames, dropped_mask)
        - stacked_frames: [num_frames, C, H, W]
        - dropped_mask: [num_frames] boolean mask
    """
    available = len(frames)

    # Pad with first frame if needed
    while len(frames) < num_frames:
        frames.insert(0, frames[0].clone())

    stacked = torch.stack(frames, dim=0).to(device)
    dropped = create_frame_mask(num_frames, available, device)

    return stacked, dropped
