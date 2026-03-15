"""
Step 5: Merge NVIDIA data with user's own recordings
Creates a combined dataset for training
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from collections import OrderedDict

def count_frames(data_dir):
    """Count frames in a data directory"""
    frames_dir = data_dir / 'frames'
    if frames_dir.exists():
        return len(list(frames_dir.glob('*.png')))
    return 0

def validate_chunk(chunk_dir):
    """Check if a chunk has valid frames and actions"""
    frames_dir = chunk_dir / 'frames'
    actions_file = chunk_dir / 'actions.json'

    if not frames_dir.exists() or not actions_file.exists():
        return False

    frames = list(frames_dir.glob('*.png'))
    if len(frames) == 0:
        return False

    try:
        with open(actions_file) as f:
            actions = json.load(f)
        # Check actions aren't all zeros (placeholder)
        has_real_actions = any(
            sum(a.get('buttons', [])) > 0 or
            sum(abs(j) for j in a.get('joysticks', [])) > 0.1
            for a in actions[:100]  # Check first 100
        )
        return has_real_actions
    except:
        return False

def convert_nvidia_to_dagger_format(chunk_dir, output_dir):
    """
    Convert NVIDIA format (actions.json with buttons/joysticks arrays)
    to DAgger format (samples.json with full action dict per frame)
    """
    frames_dir = chunk_dir / 'frames'
    actions_file = chunk_dir / 'actions.json'

    if not frames_dir.exists() or not actions_file.exists():
        return 0

    with open(actions_file) as f:
        actions = json.load(f)

    # Get sorted frame files
    frame_files = sorted(frames_dir.glob('*.png'))

    if len(frame_files) == 0:
        return 0

    # Create output structure
    output_frames_dir = output_dir / 'frames'
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for i, frame_file in enumerate(frame_files):
        if i >= len(actions):
            break

        action_data = actions[i]
        buttons = action_data.get('buttons', [0] * 17)
        joysticks = action_data.get('joysticks', [0.0] * 4)

        # Copy frame
        new_frame_path = output_frames_dir / f"{i:06d}.png"
        shutil.copy2(frame_file, new_frame_path)

        # Convert to NitroGen action format
        # NVIDIA format: buttons[17], joysticks[4] (lx, ly, rx, ry)
        # Our format: OrderedDict with named keys

        action = OrderedDict([
            ("WEST", int(buttons[11]) if len(buttons) > 11 else 0),           # X
            ("SOUTH", int(buttons[10]) if len(buttons) > 10 else 0),          # A
            ("BACK", int(buttons[14]) if len(buttons) > 14 else 0),           # Back
            ("DPAD_DOWN", int(buttons[1]) if len(buttons) > 1 else 0),
            ("DPAD_LEFT", int(buttons[2]) if len(buttons) > 2 else 0),
            ("DPAD_RIGHT", int(buttons[3]) if len(buttons) > 3 else 0),
            ("DPAD_UP", int(buttons[0]) if len(buttons) > 0 else 0),
            ("GUIDE", int(buttons[16]) if len(buttons) > 16 else 0),
            ("AXIS_LEFTX", [int(joysticks[0] * 32767)] if len(joysticks) > 0 else [0]),
            ("AXIS_LEFTY", [int(joysticks[1] * 32767)] if len(joysticks) > 1 else [0]),
            ("LEFT_SHOULDER", int(buttons[4]) if len(buttons) > 4 else 0),    # LB
            ("LEFT_TRIGGER", [int(buttons[6] * 255)] if len(buttons) > 6 else [0]),  # LT
            ("AXIS_RIGHTX", [int(joysticks[2] * 32767)] if len(joysticks) > 2 else [0]),
            ("AXIS_RIGHTY", [int(joysticks[3] * 32767)] if len(joysticks) > 3 else [0]),
            ("LEFT_THUMB", int(buttons[5]) if len(buttons) > 5 else 0),       # L3
            ("RIGHT_THUMB", int(buttons[8]) if len(buttons) > 8 else 0),      # R3
            ("RIGHT_SHOULDER", int(buttons[7]) if len(buttons) > 7 else 0),   # RB
            ("RIGHT_TRIGGER", [int(buttons[9] * 255)] if len(buttons) > 9 else [0]),  # RT
            ("START", int(buttons[15]) if len(buttons) > 15 else 0),
            ("EAST", int(buttons[12]) if len(buttons) > 12 else 0),           # B
            ("NORTH", int(buttons[13]) if len(buttons) > 13 else 0),          # Y
        ])

        sample = {
            "frame_id": i,
            "frame_path": str(new_frame_path),
            "action": dict(action),  # Convert OrderedDict to regular dict for JSON
            "timestamp": i / 30.0,
            "is_human": True,
        }
        samples.append(sample)

    # Save samples.json
    samples_path = output_dir / 'samples.json'
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)

    return len(samples)

def main():
    parser = argparse.ArgumentParser(description='Merge NVIDIA data with user recordings')
    parser.add_argument('--nvidia-dir', default='nvidia_data/extracted', help='NVIDIA extracted data')
    parser.add_argument('--user-dir', required=True, help='User recordings directory')
    parser.add_argument('--output-dir', default='combined_training_data', help='Combined output')
    parser.add_argument('--nvidia-weight', type=float, default=1.0, help='Fraction of NVIDIA data to use (0-1)')
    parser.add_argument('--validate', action='store_true', help='Only include chunks with real labels')
    args = parser.parse_args()

    nvidia_dir = Path(args.nvidia_dir)
    user_dir = Path(args.user_dir)
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Dataset Merger")
    print(f"{'='*60}")
    print(f"  NVIDIA data: {nvidia_dir}")
    print(f"  User data: {user_dir}")
    print(f"  Output: {output_dir}")
    print(f"  NVIDIA weight: {args.nvidia_weight}")
    print(f"{'='*60}\n")

    # Clear output directory if exists
    if output_dir.exists():
        print("Clearing existing output directory...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect NVIDIA chunks
    nvidia_chunks = []
    if nvidia_dir.exists():
        for video_dir in nvidia_dir.iterdir():
            if video_dir.is_dir():
                for chunk_dir in video_dir.iterdir():
                    if chunk_dir.is_dir():
                        if args.validate:
                            if validate_chunk(chunk_dir):
                                nvidia_chunks.append(chunk_dir)
                        else:
                            actions_file = chunk_dir / 'actions.json'
                            frames_dir = chunk_dir / 'frames'
                            if actions_file.exists() and frames_dir.exists():
                                nvidia_chunks.append(chunk_dir)

    # Check if user_dir is a single recording or contains multiple
    user_samples_file = user_dir / 'samples.json'
    user_frames_dir = user_dir / 'frames'

    print(f"NVIDIA chunks found: {len(nvidia_chunks)}")

    # Copy user data
    print("\nProcessing user recordings...")
    user_frames_total = 0

    if user_samples_file.exists() and user_frames_dir.exists():
        # Single recording - copy directly
        user_output = output_dir / "user_0000"
        user_output.mkdir(parents=True, exist_ok=True)

        # Copy frames
        shutil.copytree(user_frames_dir, user_output / 'frames')

        # Copy samples.json
        shutil.copy2(user_samples_file, user_output / 'samples.json')

        user_frames_total = count_frames(user_output)
        print(f"  [OK] Copied user recording ({user_frames_total:,} frames)")
    else:
        print(f"  User recording not found at {user_dir}")
        print(f"  Looking for samples.json and frames/ directory")

    # Process NVIDIA data
    print(f"\nProcessing NVIDIA data (weight: {args.nvidia_weight})...")
    num_nvidia = int(len(nvidia_chunks) * args.nvidia_weight)

    nvidia_frames_total = 0
    if num_nvidia > 0 and len(nvidia_chunks) > 0:
        selected_nvidia = random.sample(nvidia_chunks, min(num_nvidia, len(nvidia_chunks)))

        for i, chunk_dir in enumerate(selected_nvidia):
            nvidia_output = output_dir / f"nvidia_{i:04d}"

            # Convert to DAgger format
            frames_converted = convert_nvidia_to_dagger_format(chunk_dir, nvidia_output)
            nvidia_frames_total += frames_converted

            if (i + 1) % 20 == 0 or i == len(selected_nvidia) - 1:
                print(f"  Processed {i+1}/{len(selected_nvidia)} chunks ({nvidia_frames_total:,} frames)")
    else:
        selected_nvidia = []

    # Create combined samples.json that references all subdirectories
    # (dagger_train.py will need to be updated to handle this structure)

    total_frames = user_frames_total + nvidia_frames_total

    print(f"\n{'='*60}")
    print(f"  Merge Complete")
    print(f"{'='*60}")
    print(f"  User frames: {user_frames_total:,}")
    print(f"  NVIDIA frames: {nvidia_frames_total:,}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Create a manifest file listing all subdirectories
    manifest = {
        'user_recordings': 1 if user_frames_total > 0 else 0,
        'nvidia_chunks': len(selected_nvidia) if nvidia_frames_total > 0 else 0,
        'total_frames': total_frames,
        'directories': [d.name for d in output_dir.iterdir() if d.is_dir()]
    }

    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  To train on combined data:")
    print(f"  python scripts/dagger_train.py --data-dir {output_dir}/user_0000 --epochs 10")
    print(f"\n  Or to train on a specific NVIDIA chunk:")
    print(f"  python scripts/dagger_train.py --data-dir {output_dir}/nvidia_0000 --epochs 5")

if __name__ == "__main__":
    main()
