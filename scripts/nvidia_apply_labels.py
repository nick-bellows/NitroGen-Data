"""
Step 4: Apply NVIDIA's action labels to extracted frames
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Apply NVIDIA labels to extracted frames')
    parser.add_argument('--game', default='hades', help='Game name')
    parser.add_argument('--frames-dir', default='nvidia_data/extracted', help='Extracted frames directory')
    parser.add_argument('--max-entries', type=int, default=200000, help='Max dataset entries to scan')
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)

    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  Applying NVIDIA Labels")
    print(f"{'='*60}")
    print(f"  Game: {args.game}")
    print(f"  Frames directory: {frames_dir}")
    print(f"{'='*60}\n")

    # Find all chunk directories
    chunk_dirs = {}
    for video_dir in frames_dir.iterdir():
        if video_dir.is_dir():
            for chunk_dir in video_dir.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith('chunk_'):
                    video_id = video_dir.name
                    chunk_id = chunk_dir.name.replace('chunk_', '')
                    key = f"{video_id}_{chunk_id}"
                    chunk_dirs[key] = {
                        'path': chunk_dir,
                        'video_id': video_id,
                        'chunk_id': chunk_id
                    }

    print(f"Found {len(chunk_dirs)} chunk directories to label\n")

    if len(chunk_dirs) == 0:
        print("No chunks found. Run nvidia_frame_extractor.py first!")
        return

    # Load dataset
    print("Loading NVIDIA dataset (streaming mode)...")
    print("This may take a while...\n")

    from datasets import load_dataset
    dataset = load_dataset("nvidia/NitroGen", streaming=True)

    # Scan dataset and apply labels
    labels_applied = 0
    entries_scanned = 0

    print("Scanning dataset for matching labels...\n")

    for item in dataset['train']:
        entries_scanned += 1

        if entries_scanned % 10000 == 0:
            print(f"  Scanned {entries_scanned:,} entries, applied {labels_applied} labels...")

        # Check if game matches
        game = item.get('game', '').lower()
        if args.game.lower() not in game:
            continue

        # Get video/chunk info
        video_info = item.get('original_video', {})
        video_id = video_info.get('video_id')
        chunk_id = str(item.get('chunk_id', ''))

        if not video_id or not chunk_id:
            continue

        key = f"{video_id}_{chunk_id}"

        if key in chunk_dirs:
            chunk_info = chunk_dirs[key]
            chunk_path = chunk_info['path']

            # Get actions from dataset
            actions_data = item.get('actions', [])

            if actions_data and len(actions_data) > 0:
                # Convert to our format
                formatted_actions = []
                for i, action in enumerate(actions_data):
                    buttons = action.get('buttons', [0] * 17)
                    joysticks = action.get('joysticks', [0.0] * 4)

                    # Ensure correct lengths
                    if len(buttons) < 17:
                        buttons = buttons + [0] * (17 - len(buttons))
                    if len(joysticks) < 4:
                        joysticks = joysticks + [0.0] * (4 - len(joysticks))

                    formatted_actions.append({
                        'frame': i,
                        'timestamp': i / 30.0,  # Assuming 30 FPS
                        'buttons': buttons[:17],
                        'joysticks': joysticks[:4]
                    })

                # Save actions
                actions_path = chunk_path / 'actions.json'
                with open(actions_path, 'w') as f:
                    json.dump(formatted_actions, f)

                labels_applied += 1

                # Remove from dict so we know what's left
                del chunk_dirs[key]

                if labels_applied % 10 == 0:
                    print(f"  [OK] Applied labels: {labels_applied}")

        # Early exit if all chunks labeled
        if len(chunk_dirs) == 0:
            print("\n  All chunks labeled!")
            break

        if entries_scanned >= args.max_entries:
            print(f"\n  Reached max entries limit ({args.max_entries:,})")
            break

    unlabeled = len(chunk_dirs)

    print(f"\n{'='*60}")
    print(f"  Labels Applied")
    print(f"{'='*60}")
    print(f"  Entries scanned: {entries_scanned:,}")
    print(f"  Labels applied: {labels_applied}")
    print(f"  Unlabeled chunks: {unlabeled}")
    print(f"{'='*60}\n")

    if unlabeled > 0:
        print(f"  Note: {unlabeled} chunks could not be matched to NVIDIA labels.")
        print(f"  These will use placeholder (zero) actions.")

    print(f"\n  Next step: Merge with your recordings")
    print(f"  python scripts/nvidia_merge_datasets.py --user-dir games/hades/recordings/Hades_20260110_185339")

if __name__ == "__main__":
    main()
