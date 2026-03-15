"""
Step 3: Extract frames from downloaded videos
Saves to nvidia_data/extracted/
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_chunk(video_path, chunk_info, output_dir, target_size=256):
    """
    Extract frames for a single chunk
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default assumption

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_time = chunk_info.get('start_time', 0)
    end_time = chunk_info.get('end_time', total_frames / fps)
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames)

    # Get overlay bbox for masking
    bbox = chunk_info.get('bbox_overlay')

    # Create output directory for this chunk
    chunk_id = chunk_info.get('chunk_id', '0')
    chunk_dir = output_dir / f"chunk_{chunk_id}"
    frames_dir = chunk_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = 0
    frames_saved = 0

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Mask controller overlay (black out the region)
        if bbox and len(bbox) >= 4:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            frame_h, frame_w = frame.shape[:2]

            # Handle both absolute and relative coordinates
            if x < 1 and y < 1:  # Relative coordinates (0-1)
                x = int(x * frame_w)
                y = int(y * frame_h)
                w = int(w * frame_w)
                h = int(h * frame_h)

            # Ensure within bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)

            if w > 0 and h > 0:
                frame[y:y+h, x:x+w] = 0  # Black out overlay

        # Resize to NitroGen input size
        frame_resized = cv2.resize(frame, (target_size, target_size))

        # Save frame
        frame_path = frames_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame_resized)

        frames_saved += 1
        frame_idx += 1

    cap.release()

    if frames_saved == 0:
        return None

    return {
        'chunk_id': chunk_id,
        'frames_extracted': frames_saved,
        'output_dir': str(chunk_dir),
        'fps': fps
    }

def create_placeholder_actions(chunk_dir, num_frames, fps=30):
    """Create placeholder actions file (to be replaced with NVIDIA labels)"""
    actions = []
    for i in range(num_frames):
        actions.append({
            'frame': i,
            'timestamp': i / fps,
            'buttons': [0] * 17,
            'joysticks': [0.0, 0.0, 0.0, 0.0]
        })

    actions_path = Path(chunk_dir) / 'actions.json'
    with open(actions_path, 'w') as f:
        json.dump(actions, f)

def main():
    parser = argparse.ArgumentParser(description='Extract frames from downloaded videos')
    parser.add_argument('--dataset-info', default='nvidia_data/dataset_info/hades_info.json', help='Dataset info file')
    parser.add_argument('--videos-dir', default='nvidia_data/videos', help='Downloaded videos directory')
    parser.add_argument('--output-dir', default='nvidia_data/extracted', help='Output directory')
    parser.add_argument('--target-size', type=int, default=256, help='Frame size')
    parser.add_argument('--max-chunks', type=int, default=100, help='Max chunks to process')
    parser.add_argument('--max-chunks-per-video', type=int, default=10, help='Max chunks per video')
    args = parser.parse_args()

    # Load dataset info
    dataset_path = Path(args.dataset_info)
    if not dataset_path.exists():
        print(f"Error: Dataset info not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        data = json.load(f)

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NVIDIA Frame Extractor")
    print(f"{'='*60}")
    print(f"  Videos directory: {videos_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Target frame size: {args.target_size}x{args.target_size}")
    print(f"  Max chunks: {args.max_chunks}")
    print(f"{'='*60}\n")

    # Find available videos
    available_videos = list(videos_dir.glob('*.mp4'))
    print(f"Found {len(available_videos)} downloaded videos\n")

    # Process each video
    processed_chunks = 0
    total_frames = 0

    for video_path in tqdm(available_videos, desc="Processing videos"):
        video_id = video_path.stem

        if video_id not in data.get('videos', {}):
            continue

        video_info = data['videos'][video_id]
        chunks = video_info.get('chunks', [])

        if not chunks:
            continue

        video_output_dir = output_dir / video_id
        chunks_processed_for_video = 0

        for chunk in chunks:
            if processed_chunks >= args.max_chunks:
                break
            if chunks_processed_for_video >= args.max_chunks_per_video:
                break

            # Extract frames
            result = extract_chunk(
                video_path,
                chunk,
                video_output_dir,
                args.target_size
            )

            if result:
                # Create placeholder actions (will be replaced by apply_labels)
                create_placeholder_actions(
                    result['output_dir'],
                    result['frames_extracted'],
                    result['fps']
                )

                processed_chunks += 1
                chunks_processed_for_video += 1
                total_frames += result['frames_extracted']

        if processed_chunks >= args.max_chunks:
            break

    print(f"\n{'='*60}")
    print(f"  Extraction Complete")
    print(f"{'='*60}")
    print(f"  Chunks processed: {processed_chunks}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    print(f"  Next step: Apply NVIDIA labels")
    print(f"  python scripts/nvidia_apply_labels.py")

if __name__ == "__main__":
    main()
