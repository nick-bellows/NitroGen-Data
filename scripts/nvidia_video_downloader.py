"""
Step 2: Download available videos from the dataset
Saves to nvidia_data/videos/
"""

import argparse
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def download_video(video_id, url, output_dir, resolution='720'):
    """Download a single video using yt-dlp"""
    output_path = output_dir / f"{video_id}.mp4"

    if output_path.exists():
        # Check if file is valid (not empty/corrupted)
        if output_path.stat().st_size > 1000000:  # > 1MB
            return {'video_id': video_id, 'status': 'exists', 'path': str(output_path)}

    cmd = [
        'yt-dlp',
        '-f', f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        '--merge-output-format', 'mp4',
        '-o', str(output_path),
        '--no-playlist',
        '--socket-timeout', '30',
        '--retries', '3',
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and output_path.exists():
            return {'video_id': video_id, 'status': 'downloaded', 'path': str(output_path)}
        else:
            return {'video_id': video_id, 'status': 'failed', 'error': result.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {'video_id': video_id, 'status': 'timeout'}
    except Exception as e:
        return {'video_id': video_id, 'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Download videos from NVIDIA dataset')
    parser.add_argument('--input', default='nvidia_data/dataset_info/hades_info.json', help='Dataset info file')
    parser.add_argument('--output-dir', default='nvidia_data/videos', help='Output directory')
    parser.add_argument('--max-videos', type=int, default=50, help='Max videos to download')
    parser.add_argument('--resolution', default='720', help='Max video resolution')
    parser.add_argument('--workers', type=int, default=2, help='Parallel downloads')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run nvidia_dataset_explorer.py first!")
        return

    with open(input_path) as f:
        data = json.load(f)

    videos = data.get('videos', {})

    print(f"\n{'='*60}")
    print(f"  NVIDIA Video Downloader")
    print(f"{'='*60}")
    print(f"  Videos in dataset: {len(videos)}")
    print(f"  Max to download: {args.max_videos}")
    print(f"  Resolution: {args.resolution}p")
    print(f"  Output: {output_dir}")
    print(f"  Parallel workers: {args.workers}")
    print(f"{'='*60}\n")

    # Filter to YouTube only and limit count
    download_queue = []
    for video_id, info in list(videos.items()):
        if len(download_queue) >= args.max_videos:
            break
        if info.get('source') == 'youtube':
            url = info.get('url')
            if not url:
                url = f"https://www.youtube.com/watch?v={video_id}"
            download_queue.append((video_id, url))

    print(f"Downloading {len(download_queue)} videos...\n")
    print("This may take a while. Progress:\n")

    results = {'downloaded': [], 'exists': [], 'failed': []}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_video, vid, url, output_dir, args.resolution): vid
            for vid, url in download_queue
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            status = result['status']

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0

            if status == 'downloaded':
                results['downloaded'].append(result)
                print(f"  [{i+1}/{len(download_queue)}] [OK] Downloaded: {result['video_id']} ({rate:.1f}/min)")
            elif status == 'exists':
                results['exists'].append(result)
                print(f"  [{i+1}/{len(download_queue)}] [--] Exists: {result['video_id']}")
            else:
                results['failed'].append(result)
                error = result.get('error', status)[:50]
                print(f"  [{i+1}/{len(download_queue)}] [X]  Failed: {result['video_id']} - {error}")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Download Complete")
    print(f"{'='*60}")
    print(f"  Downloaded: {len(results['downloaded'])}")
    print(f"  Already existed: {len(results['exists'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}\n")

    # Calculate disk usage
    total_size = sum(
        Path(r['path']).stat().st_size
        for r in results['downloaded'] + results['exists']
        if Path(r.get('path', '')).exists()
    )
    print(f"  Disk usage: {total_size / (1024**3):.2f} GB")

    # Save download results
    results_path = output_dir / 'download_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {results_path}")
    print(f"\n  Next step: Run frame extraction")
    print(f"  python scripts/nvidia_frame_extractor.py")

if __name__ == "__main__":
    main()
