"""
Optional: Delete downloaded shard files to reclaim disk space.
Run this AFTER extraction is complete.

The metadata files (~100MB) will be kept.
Only the raw shard downloads (~160GB) will be deleted.
"""

import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Cleanup downloaded shards to free disk space')
    parser.add_argument('--downloads-dir', default='nvidia_index/downloads', help='Downloads directory')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)

    if not downloads_dir.exists():
        print(f"Directory not found: {downloads_dir}")
        print("Nothing to clean up.")
        return

    # Calculate size
    total_size = 0
    file_count = 0
    for f in downloads_dir.rglob('*'):
        if f.is_file():
            total_size += f.stat().st_size
            file_count += 1

    size_gb = total_size / (1024**3)

    print(f"\n{'='*60}")
    print(f"  Cleanup Downloaded Shards")
    print(f"{'='*60}")
    print(f"  Directory: {downloads_dir}")
    print(f"  Files: {file_count:,}")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"{'='*60}\n")

    if size_gb < 0.001:
        print("Directory is essentially empty. Nothing to clean up.")
        return

    if not args.force:
        print("This will permanently delete the downloaded shard files.")
        print("The extracted metadata and index files will NOT be affected.")
        print()
        confirm = input("Delete these files? Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("\nCancelled. No files were deleted.")
            return

    print("\nDeleting files...")

    try:
        shutil.rmtree(downloads_dir)
        print(f"\n  Deleted: {downloads_dir}")
        print(f"  Freed: {size_gb:.2f} GB")
        print("\n  Cleanup complete!")
    except Exception as e:
        print(f"\n  Error during deletion: {e}")
        print("  Some files may not have been deleted.")

if __name__ == "__main__":
    main()
