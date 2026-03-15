"""
Extract metadata.json files from downloaded shards.
Skips large parquet files to save space.
Output: nvidia_index/metadata/

Each shard contains thousands of metadata.json files (~1-2KB each).
Total extracted metadata is ~100MB vs ~160GB for full dataset.
"""

import argparse
import json
import tarfile
import time
from pathlib import Path
from datetime import datetime

def extract_metadata_from_shard(shard_path, output_dir):
    """Extract only metadata.json files from a shard tar.gz"""

    shard_name = shard_path.stem.replace('.tar', '')  # SHARD_0000
    shard_output = output_dir / shard_name
    shard_output.mkdir(parents=True, exist_ok=True)

    metadata_count = 0
    errors = []

    try:
        with tarfile.open(shard_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('metadata.json'):
                    try:
                        f = tar.extractfile(member)
                        if f:
                            content = f.read()
                            metadata = json.loads(content)

                            # Create a clean filename from the path
                            # e.g., "video_id/video_id_chunk_0001/metadata.json"
                            parts = member.name.split('/')
                            if len(parts) >= 2:
                                chunk_name = parts[-2]  # video_id_chunk_0001
                            else:
                                chunk_name = f"chunk_{metadata_count:06d}"

                            # Save individual metadata file
                            output_file = shard_output / f"{chunk_name}.json"
                            with open(output_file, 'w') as out:
                                json.dump(metadata, out, indent=2)

                            metadata_count += 1

                    except Exception as e:
                        errors.append({
                            'file': member.name,
                            'error': str(e)
                        })

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'metadata_count': metadata_count
        }

    return {
        'status': 'success',
        'shard': shard_name,
        'metadata_count': metadata_count,
        'errors': errors,
        'output_dir': str(shard_output)
    }

def main():
    parser = argparse.ArgumentParser(description='Extract metadata from NitroGen shards')
    parser.add_argument('--downloads-dir', default='nvidia_index/downloads/actions', help='Downloaded shards directory')
    parser.add_argument('--output-dir', default='nvidia_index/metadata', help='Output directory for metadata')
    parser.add_argument('--log-dir', default='nvidia_index/logs', help='Log directory')
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'extraction_log.json'
    errors_file = log_dir / 'errors.json'

    # Load existing log if resuming
    if log_file.exists():
        with open(log_file) as f:
            extraction_log = json.load(f)
    else:
        extraction_log = {
            'started': datetime.now().isoformat(),
            'completed_shards': [],
            'failed_shards': [],
            'total_metadata': 0,
            'shard_stats': {}
        }

    # Find all downloaded shards
    shard_files = sorted(downloads_dir.glob('SHARD_*.tar.gz'))

    print(f"\n{'='*60}")
    print(f"  NitroGen Metadata Extractor")
    print(f"{'='*60}")
    print(f"  Downloads directory: {downloads_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Found shards: {len(shard_files)}")
    print(f"  Already processed: {len(extraction_log['completed_shards'])}")
    print(f"{'='*60}\n")

    if len(shard_files) == 0:
        print("No shard files found!")
        print(f"Expected location: {downloads_dir}")
        print("\nMake sure downloads completed successfully.")
        print("Run: python scripts/nvidia_index/download_shards.py")
        return

    # Filter to unprocessed shards
    shards_to_process = [
        s for s in shard_files
        if s.name not in extraction_log['completed_shards']
    ]

    print(f"Shards to process: {len(shards_to_process)}\n")

    start_time = time.time()
    total_metadata = extraction_log['total_metadata']
    all_errors = []

    for i, shard_path in enumerate(shards_to_process):
        print(f"[{i+1}/{len(shards_to_process)}] Processing {shard_path.name}...")

        shard_start = time.time()
        result = extract_metadata_from_shard(shard_path, output_dir)
        elapsed = time.time() - shard_start

        if result['status'] == 'success':
            print(f"    Extracted {result['metadata_count']:,} metadata files ({elapsed:.1f}s)")
            extraction_log['completed_shards'].append(shard_path.name)
            total_metadata += result['metadata_count']
            extraction_log['total_metadata'] = total_metadata
            extraction_log['shard_stats'][shard_path.name] = {
                'metadata_count': result['metadata_count'],
                'extraction_time': elapsed
            }

            if result['errors']:
                all_errors.extend(result['errors'])
                print(f"    (with {len(result['errors'])} minor errors)")
        else:
            print(f"    Failed: {result['error']}")
            extraction_log['failed_shards'].append({
                'shard': shard_path.name,
                'error': result['error']
            })

        # Save log after each shard
        with open(log_file, 'w') as f:
            json.dump(extraction_log, f, indent=2)

        # Progress
        processed = i + 1
        if processed > 0:
            avg_time = (time.time() - start_time) / processed
            remaining = len(shards_to_process) - processed
            eta_min = (avg_time * remaining) / 60
            print(f"    Total metadata so far: {total_metadata:,} | ETA: {eta_min:.0f} min\n")

    # Save errors to separate file
    if all_errors:
        with open(errors_file, 'w') as f:
            json.dump(all_errors, f, indent=2)

    # Final summary
    extraction_log['finished'] = datetime.now().isoformat()
    extraction_log['total_time_seconds'] = time.time() - start_time

    with open(log_file, 'w') as f:
        json.dump(extraction_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Extraction Complete")
    print(f"{'='*60}")
    print(f"  Shards processed: {len(extraction_log['completed_shards'])}")
    print(f"  Total metadata files: {total_metadata:,}")
    print(f"  Failed shards: {len(extraction_log['failed_shards'])}")
    print(f"  Minor errors: {len(all_errors)}")
    print(f"  Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"{'='*60}\n")

    # Calculate output size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*.json'))
    print(f"  Metadata size on disk: {total_size / (1024**2):.1f} MB")

    print("\nNext step: Generate index")
    print("python scripts/nvidia_index/generate_index.py")

if __name__ == "__main__":
    main()
