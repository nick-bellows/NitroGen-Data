"""
Step 1: Explore NVIDIA dataset and find Hades videos
Saves results to nvidia_data/dataset_info/

The NitroGen dataset is stored as tar.gz archives on HuggingFace.
This script downloads and extracts metadata to find game-specific content.
"""

import argparse
import json
import tarfile
import tempfile
import os
from pathlib import Path
from collections import defaultdict
import subprocess

def check_video_exists(video_id, source='youtube'):
    """Check if video is still available using yt-dlp"""
    if source == 'youtube':
        url = f"https://www.youtube.com/watch?v={video_id}"
    else:
        return None

    try:
        result = subprocess.run(
            ['yt-dlp', '--skip-download', '--print', 'title', url],
            capture_output=True,
            timeout=15,
            text=True
        )
        if result.returncode == 0:
            return {'available': True, 'title': result.stdout.strip()}
        return {'available': False, 'error': result.stderr.strip()[:200]}
    except subprocess.TimeoutExpired:
        return {'available': None, 'error': 'timeout'}
    except Exception as e:
        return {'available': None, 'error': str(e)}

def download_and_scan_shard(shard_name, game_filter, cache_dir):
    """Download a shard tar.gz and scan for matching game entries"""
    from huggingface_hub import hf_hub_download

    results = []

    try:
        # Download the shard
        local_path = hf_hub_download(
            repo_id="nvidia/NitroGen",
            repo_type="dataset",
            filename=f"actions/{shard_name}",
            cache_dir=cache_dir
        )

        # Extract and scan metadata files
        with tarfile.open(local_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('metadata.json'):
                    try:
                        f = tar.extractfile(member)
                        if f:
                            metadata = json.load(f)
                            game = metadata.get('game', '').lower()

                            if game_filter.lower() in game:
                                results.append({
                                    'metadata': metadata,
                                    'shard': shard_name,
                                    'path': member.name
                                })
                    except Exception as e:
                        continue

    except Exception as e:
        print(f"    Error processing {shard_name}: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Explore NVIDIA NitroGen dataset')
    parser.add_argument('--game', default='hades', help='Game to filter for')
    parser.add_argument('--output', default='nvidia_data/dataset_info/hades_info.json', help='Output file')
    parser.add_argument('--max-shards', type=int, default=5, help='Max shards to scan (each ~1.6GB)')
    parser.add_argument('--check-availability', action='store_true', help='Check if videos exist')
    parser.add_argument('--check-limit', type=int, default=20, help='Max videos to check availability')
    parser.add_argument('--cache-dir', default=None, help='HuggingFace cache directory')
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NVIDIA NitroGen Dataset Explorer")
    print(f"{'='*60}")
    print(f"  Target game: {args.game}")
    print(f"  Max shards to scan: {args.max_shards}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run(['pip', 'install', 'huggingface_hub'])
        from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()

    # List available shards
    print("Listing available shards...")
    try:
        files = list(api.list_repo_tree(
            repo_id="nvidia/NitroGen",
            repo_type="dataset",
            path_in_repo="actions",
            recursive=False
        ))

        shard_files = sorted([f.path for f in files if f.path.endswith('.tar.gz')])
        print(f"Found {len(shard_files)} shards (each ~1.6GB compressed)\n")

    except Exception as e:
        print(f"Error listing shards: {e}")
        print("\nTrying hardcoded shard names...")
        shard_files = [f"actions/SHARD_{i:04d}.tar.gz" for i in range(100)]

    # Collect matching entries
    videos = defaultdict(lambda: {
        'chunks': [],
        'total_frames': 0,
        'total_duration': 0,
        'source': 'youtube',
        'url': None
    })

    entries_scanned = 0
    matching_entries = 0

    print(f"Scanning shards for '{args.game}' content...")
    print(f"WARNING: Each shard is ~1.6GB. Scanning {args.max_shards} shards.\n")

    cache_dir = args.cache_dir

    for i, shard_path in enumerate(shard_files[:args.max_shards]):
        shard_name = shard_path.split('/')[-1] if '/' in shard_path else shard_path
        print(f"  [{i+1}/{args.max_shards}] Downloading and scanning {shard_name}...")

        try:
            results = download_and_scan_shard(shard_path, args.game, cache_dir)
            entries_scanned += 1  # Count shards scanned

            for result in results:
                metadata = result['metadata']
                matching_entries += 1

                # Extract video info
                original_video = metadata.get('original_video', {})
                video_id = original_video.get('video_id', '')
                if not video_id:
                    video_id = metadata.get('video_id', f'unknown_{matching_entries}')

                # Store video metadata
                if videos[video_id]['url'] is None:
                    videos[video_id]['url'] = original_video.get('url')
                    videos[video_id]['source'] = original_video.get('source', 'youtube')
                    videos[video_id]['resolution'] = original_video.get('resolution')

                # Store chunk info
                chunk_info = {
                    'chunk_id': metadata.get('chunk_id'),
                    'uuid': metadata.get('uuid'),
                    'start_time': original_video.get('start_time'),
                    'end_time': original_video.get('end_time'),
                    'duration': original_video.get('duration', 20),
                    'start_frame': original_video.get('start_frame'),
                    'end_frame': original_video.get('end_frame'),
                    'chunk_size': metadata.get('chunk_size'),
                    'controller_type': metadata.get('controller_type'),
                    'bbox_overlay': metadata.get('bbox_controller_overlay'),
                    'shard': result['shard'],
                    'parquet_path': result['path'].replace('metadata.json', 'actions_processed.parquet')
                }

                videos[video_id]['chunks'].append(chunk_info)

                # Calculate frames
                chunk_size = metadata.get('chunk_size', 0)
                if chunk_size:
                    videos[video_id]['total_frames'] += chunk_size
                else:
                    duration = original_video.get('duration', 20)
                    videos[video_id]['total_frames'] += int(duration * 30)

                videos[video_id]['total_duration'] += original_video.get('duration', 20)

            if results:
                print(f"    Found {len(results)} {args.game} chunks in this shard")
            else:
                print(f"    No {args.game} content in this shard")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Calculate totals
    total_videos = len(videos)
    total_hours = sum(v['total_duration'] for v in videos.values()) / 3600
    total_frames = sum(v['total_frames'] for v in videos.values())

    print(f"\n{'='*60}")
    print(f"  Results for '{args.game}'")
    print(f"{'='*60}")
    print(f"  Shards scanned:      {entries_scanned}")
    print(f"  Matching chunks:     {matching_entries}")
    print(f"  Unique videos:       {total_videos}")
    print(f"  Total duration:      {total_hours:.1f} hours")
    print(f"  Total frames:        {total_frames:,}")
    print(f"{'='*60}\n")

    # Check video availability
    availability_results = {}
    if args.check_availability and total_videos > 0:
        print(f"Checking video availability (limit: {args.check_limit})...\n")

        available_count = 0
        unavailable_count = 0
        unknown_count = 0

        for i, (video_id, info) in enumerate(list(videos.items())[:args.check_limit]):
            result = check_video_exists(video_id, info['source'])
            availability_results[video_id] = result

            if result and result.get('available') is True:
                available_count += 1
                title = result.get('title', 'Available')[:50]
                print(f"  [OK] {video_id}: {title}")
            elif result and result.get('available') is False:
                unavailable_count += 1
                print(f"  [X]  {video_id}: Unavailable")
            else:
                unknown_count += 1
                error = result.get('error', 'Unknown') if result else 'Unknown'
                print(f"  [?]  {video_id}: {error}")

        print(f"\n  Availability: {available_count} OK | {unavailable_count} X | {unknown_count} ?")

        if available_count > 0:
            availability_rate = available_count / min(args.check_limit, total_videos)
            estimated_available = availability_rate * total_videos
            estimated_hours = availability_rate * total_hours
            print(f"  Estimated available: ~{estimated_available:.0f} videos (~{estimated_hours:.1f} hours)")

    # Save results
    output_data = {
        'game': args.game,
        'shards_scanned': entries_scanned,
        'matching_chunks': matching_entries,
        'total_videos': total_videos,
        'total_hours': total_hours,
        'total_frames': total_frames,
        'availability_checked': len(availability_results),
        'availability_results': availability_results,
        'videos': {k: dict(v) for k, v in videos.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    if total_videos > 0:
        print(f"\n  Next step: Run download script")
        print(f"  python scripts/nvidia_video_downloader.py --input {output_path}")
    else:
        print(f"\n  No {args.game} videos found in {entries_scanned} shards.")
        print(f"  Try scanning more shards with --max-shards 20")
        print(f"\n  Note: Hades content may be in later shards.")
        print(f"  The dataset covers 1000+ games across 100 shards.")

if __name__ == "__main__":
    main()
