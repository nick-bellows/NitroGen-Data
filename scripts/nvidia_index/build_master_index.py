"""
Main orchestrator - runs the full pipeline.
1. Download shards from HuggingFace
2. Extract metadata.json files
3. Generate index files

Usage:
    python scripts/nvidia_index/build_master_index.py
    python scripts/nvidia_index/build_master_index.py --start-shard 0 --end-shard 49
    python scripts/nvidia_index/build_master_index.py --skip-download --skip-extract
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_step(script_path, args_list, description):
    """Run a pipeline step"""
    print(f"\n{'#'*60}")
    print(f"  {description}")
    print(f"{'#'*60}\n")

    cmd = [sys.executable, str(script_path)] + args_list

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n Step failed: {description}")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")
        return False

    print(f"\n  Step completed in {elapsed/60:.1f} minutes")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Build NitroGen master index (full pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all 100 shards)
  python scripts/nvidia_index/build_master_index.py

  # Download first 10 shards only (for testing)
  python scripts/nvidia_index/build_master_index.py --start-shard 0 --end-shard 9

  # Skip download, just regenerate index from existing metadata
  python scripts/nvidia_index/build_master_index.py --skip-download --skip-extract

  # Delete shard files after extraction to save space
  python scripts/nvidia_index/build_master_index.py --cleanup
        """
    )
    parser.add_argument('--start-shard', type=int, default=0, help='Starting shard (default: 0)')
    parser.add_argument('--end-shard', type=int, default=99, help='Ending shard (default: 99)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download step')
    parser.add_argument('--skip-extract', action='store_true', help='Skip extraction step')
    parser.add_argument('--cleanup', action='store_true', help='Delete shard files after extraction')
    args = parser.parse_args()

    base_dir = Path('nvidia_index')
    scripts_dir = Path('scripts/nvidia_index')

    # Ensure scripts exist
    required_scripts = ['download_shards.py', 'extract_metadata.py', 'generate_index.py']
    for script in required_scripts:
        if not (scripts_dir / script).exists():
            print(f"Error: Required script not found: {scripts_dir / script}")
            return 1

    total_shards = args.end_shard - args.start_shard + 1
    est_download_gb = total_shards * 1.6
    est_download_hours = total_shards * 0.05  # ~3 min per shard at 10MB/s

    print(f"\n{'#'*60}")
    print(f"  NitroGen Master Index Builder")
    print(f"{'#'*60}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Shard range: {args.start_shard} - {args.end_shard} ({total_shards} shards)")
    print(f"  Estimated download: ~{est_download_gb:.0f} GB")
    print(f"  Estimated time: ~{est_download_hours:.1f} hours (download) + ~1 hour (processing)")
    print(f"  Output: {base_dir}")
    print(f"{'#'*60}\n")

    pipeline_start = time.time()
    steps_completed = 0

    # Step 1: Download
    if not args.skip_download:
        success = run_step(
            scripts_dir / 'download_shards.py',
            ['--start-shard', str(args.start_shard), '--end-shard', str(args.end_shard)],
            'Step 1/3: Downloading Shards from HuggingFace'
        )
        if not success:
            print("\nDownload failed. You can resume later - progress is saved.")
            return 1
        steps_completed += 1
    else:
        print("\n[Skipping download step]")

    # Step 2: Extract
    if not args.skip_extract:
        success = run_step(
            scripts_dir / 'extract_metadata.py',
            [],
            'Step 2/3: Extracting Metadata from Shards'
        )
        if not success:
            print("\nExtraction failed. Check logs in nvidia_index/logs/")
            return 1
        steps_completed += 1
    else:
        print("\n[Skipping extraction step]")

    # Step 3: Generate Index
    success = run_step(
        scripts_dir / 'generate_index.py',
        [],
        'Step 3/3: Generating Index Files'
    )
    if not success:
        print("\nIndex generation failed.")
        return 1
    steps_completed += 1

    # Optional cleanup
    if args.cleanup:
        print(f"\n{'#'*60}")
        print(f"  Cleanup: Deleting Downloaded Shards")
        print(f"{'#'*60}\n")

        downloads_dir = base_dir / 'downloads'
        if downloads_dir.exists():
            # Calculate size before deletion
            total_size = sum(f.stat().st_size for f in downloads_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)

            import shutil
            shutil.rmtree(downloads_dir)
            print(f"  Deleted: {downloads_dir}")
            print(f"  Freed: {size_gb:.1f} GB")

    total_time = time.time() - pipeline_start

    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"  Steps completed: {steps_completed}/3")
    print(f"{'#'*60}\n")

    print(f"  Index files are in: {base_dir / 'index'}/")
    print(f"\n  Key files:")
    print(f"    - master_index.json  (complete details)")
    print(f"    - games_summary.csv  (spreadsheet-friendly)")
    print(f"    - game_shards.json   (find shards by game)")
    print(f"    - games_by_hours.md  (human-readable)")
    print()

    # Show quick usage example
    print("  Quick usage:")
    print("  ```python")
    print("  import json")
    print("  with open('nvidia_index/index/game_shards.json') as f:")
    print("      data = json.load(f)")
    print("  print(data['games'].get('Hades', []))")
    print("  ```")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
