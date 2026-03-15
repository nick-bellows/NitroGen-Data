"""
Download NitroGen shards from HuggingFace with resume support.
Downloads to: nvidia_index/downloads/

Each shard is ~1.6GB compressed (tar.gz).
100 shards total = ~160GB download.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Download NitroGen shards')
    parser.add_argument('--output-dir', default='nvidia_index/downloads', help='Download directory')
    parser.add_argument('--start-shard', type=int, default=0, help='Starting shard number')
    parser.add_argument('--end-shard', type=int, default=99, help='Ending shard number')
    parser.add_argument('--log-dir', default='nvidia_index/logs', help='Log directory')
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, HfApi

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'download_log.json'

    # Load existing log if resuming
    if log_file.exists():
        with open(log_file) as f:
            download_log = json.load(f)
    else:
        download_log = {
            'started': datetime.now().isoformat(),
            'completed_shards': [],
            'failed_shards': [],
            'in_progress': None,
            'total_bytes_downloaded': 0
        }

    print(f"\n{'='*60}")
    print(f"  NitroGen Shard Downloader")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Shard range: {args.start_shard} - {args.end_shard}")
    print(f"  Already completed: {len(download_log['completed_shards'])} shards")
    print(f"{'='*60}\n")

    api = HfApi()

    # Get list of all shards
    print("Fetching shard list from HuggingFace...")
    try:
        files = list(api.list_repo_tree(
            repo_id="nvidia/NitroGen",
            repo_type="dataset",
            path_in_repo="actions",
            recursive=False
        ))
        shard_files = sorted([f.path for f in files if f.path.endswith('.tar.gz')])
        print(f"Found {len(shard_files)} total shards\n")
    except Exception as e:
        print(f"Error listing shards: {e}")
        print("Using hardcoded shard names...")
        shard_files = [f"actions/SHARD_{i:04d}.tar.gz" for i in range(100)]

    # Filter to requested range
    shards_to_download = []
    for shard_path in shard_files:
        shard_name = shard_path.split('/')[-1]
        try:
            shard_num = int(shard_name.replace('SHARD_', '').replace('.tar.gz', ''))
            if args.start_shard <= shard_num <= args.end_shard:
                if shard_name not in download_log['completed_shards']:
                    shards_to_download.append(shard_path)
        except:
            continue

    print(f"Shards to download: {len(shards_to_download)}")
    if len(shards_to_download) > 0:
        est_size_gb = len(shards_to_download) * 1.6
        print(f"Estimated download size: ~{est_size_gb:.0f} GB\n")

    total_downloaded = 0
    start_time = time.time()
    session_bytes = 0

    for i, shard_path in enumerate(shards_to_download):
        shard_name = shard_path.split('/')[-1]

        print(f"[{i+1}/{len(shards_to_download)}] Downloading {shard_name}...")
        download_log['in_progress'] = shard_name

        # Save log before download (in case of crash)
        with open(log_file, 'w') as f:
            json.dump(download_log, f, indent=2)

        try:
            shard_start = time.time()

            # Download to our directory
            local_path = hf_hub_download(
                repo_id="nvidia/NitroGen",
                repo_type="dataset",
                filename=shard_path,
                local_dir=str(output_dir)
            )

            elapsed = time.time() - shard_start
            total_downloaded += 1

            # Get file size
            file_size_bytes = Path(local_path).stat().st_size
            file_size_gb = file_size_bytes / (1024**3)
            session_bytes += file_size_bytes
            download_log['total_bytes_downloaded'] = download_log.get('total_bytes_downloaded', 0) + file_size_bytes

            # Calculate speed
            speed_mbps = (file_size_bytes / elapsed) / (1024**2)

            print(f"    Downloaded: {file_size_gb:.2f} GB in {elapsed/60:.1f} min ({speed_mbps:.1f} MB/s)")

            download_log['completed_shards'].append(shard_name)
            download_log['in_progress'] = None

        except Exception as e:
            print(f"    Failed: {e}")
            download_log['failed_shards'].append({
                'shard': shard_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

        # Save log after each download
        with open(log_file, 'w') as f:
            json.dump(download_log, f, indent=2)

        # Progress update
        if total_downloaded > 0:
            avg_time = (time.time() - start_time) / total_downloaded
            remaining = len(shards_to_download) - (i + 1)
            eta_hours = (avg_time * remaining) / 3600
            total_gb = session_bytes / (1024**3)
            print(f"    Session total: {total_gb:.1f} GB | ETA for remaining {remaining}: {eta_hours:.1f} hours\n")

    # Final summary
    total_time = time.time() - start_time
    download_log['finished'] = datetime.now().isoformat()
    download_log['total_time_seconds'] = total_time

    with open(log_file, 'w') as f:
        json.dump(download_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Download Complete")
    print(f"{'='*60}")
    print(f"  Downloaded this session: {total_downloaded} shards")
    print(f"  Total completed: {len(download_log['completed_shards'])} shards")
    print(f"  Failed: {len(download_log['failed_shards'])} shards")
    print(f"  Session time: {total_time/3600:.1f} hours")
    print(f"  Session size: {session_bytes/(1024**3):.1f} GB")
    print(f"  Log saved to: {log_file}")
    print(f"{'='*60}\n")

    if download_log['failed_shards']:
        print("Failed shards:")
        for fail in download_log['failed_shards']:
            print(f"  - {fail['shard']}: {fail['error'][:50]}")
        print()

    print("Next step: Extract metadata")
    print("python scripts/nvidia_index/extract_metadata.py")

if __name__ == "__main__":
    main()
