"""
Generate master index files from extracted metadata.
Creates multiple output formats for different use cases.

Output files:
- master_index.json    Complete index with all details
- games_summary.csv    One row per game (spreadsheet-friendly)
- videos_list.csv      One row per video with URLs
- shard_map.json       Which games are in which shards
- game_shards.json     Which shards to download for each game
- games_by_hours.md    Human-readable ranking
- README.md            Documentation
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_all_metadata(metadata_dir):
    """Load all metadata files and organize by game/video"""

    games = defaultdict(lambda: {
        'videos': defaultdict(lambda: {
            'chunks': [],
            'total_duration': 0,
            'total_frames': 0,
            'url': None,
            'source': 'youtube',
            'resolution': None
        }),
        'total_hours': 0,
        'total_chunks': 0,
        'total_frames': 0,
        'shards': set()
    })

    shard_contents = defaultdict(set)  # shard -> set of games
    all_entries = []
    errors = []

    metadata_dir = Path(metadata_dir)
    shard_dirs = sorted(metadata_dir.glob('SHARD_*'))

    print(f"Loading metadata from {len(shard_dirs)} shard directories...\n")

    total_files = 0
    for shard_dir in shard_dirs:
        shard_name = shard_dir.name
        metadata_files = list(shard_dir.glob('*.json'))
        total_files += len(metadata_files)

        for meta_file in metadata_files:
            try:
                with open(meta_file, encoding='utf-8') as f:
                    metadata = json.load(f)

                # Extract game name
                game = metadata.get('game', 'Unknown')
                if not game or game.strip() == '':
                    game = 'Unknown'

                # Normalize game name (trim whitespace)
                game = game.strip()

                # Extract video info
                original_video = metadata.get('original_video', {})
                video_id = original_video.get('video_id', '')
                if not video_id:
                    video_id = metadata.get('video_id', meta_file.stem)

                # Get chunk details
                duration = original_video.get('duration', 20)
                if duration is None:
                    duration = 20
                chunk_size = metadata.get('chunk_size', int(duration * 30))
                if chunk_size is None:
                    chunk_size = int(duration * 30)

                # Update game stats
                games[game]['total_hours'] += duration / 3600
                games[game]['total_chunks'] += 1
                games[game]['total_frames'] += chunk_size
                games[game]['shards'].add(shard_name)

                # Update video stats
                video = games[game]['videos'][video_id]
                video['total_duration'] += duration
                video['total_frames'] += chunk_size
                if video['url'] is None:
                    video['url'] = original_video.get('url')
                    video['source'] = original_video.get('source', 'youtube')
                    video['resolution'] = original_video.get('resolution')

                video['chunks'].append({
                    'chunk_id': metadata.get('chunk_id'),
                    'uuid': metadata.get('uuid'),
                    'start_time': original_video.get('start_time'),
                    'end_time': original_video.get('end_time'),
                    'duration': duration,
                    'frames': chunk_size,
                    'shard': shard_name,
                    'controller_type': metadata.get('controller_type'),
                    'bbox_overlay': metadata.get('bbox_controller_overlay')
                })

                # Track shard contents
                shard_contents[shard_name].add(game)

                # Store raw entry for detailed exports
                all_entries.append({
                    'game': game,
                    'video_id': video_id,
                    'chunk_id': metadata.get('chunk_id'),
                    'shard': shard_name,
                    'duration': duration,
                    'frames': chunk_size,
                    'url': original_video.get('url'),
                    'controller_type': metadata.get('controller_type')
                })

            except Exception as e:
                errors.append({
                    'file': str(meta_file),
                    'error': str(e)
                })
                continue

        # Progress indicator
        print(f"  Processed {shard_name}: {len(metadata_files):,} files | Running total: {len(all_entries):,}")

    # Convert sets to lists for JSON serialization
    for game in games:
        games[game]['shards'] = sorted(list(games[game]['shards']))
        games[game]['video_count'] = len(games[game]['videos'])

    shard_contents = {k: sorted(list(v)) for k, v in shard_contents.items()}

    print(f"\nLoaded {total_files:,} metadata files")
    if errors:
        print(f"  ({len(errors)} files had errors)")

    return dict(games), shard_contents, all_entries, errors

def generate_master_index(games, shard_contents, all_entries, output_dir):
    """Generate all index files"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # Calculate totals
    total_games = len(games)
    total_videos = sum(g['video_count'] for g in games.values())
    total_hours = sum(g['total_hours'] for g in games.values())
    total_chunks = sum(g['total_chunks'] for g in games.values())
    total_frames = sum(g['total_frames'] for g in games.values())

    # 1. Master Index (JSON) - Complete details
    print("\nGenerating master_index.json...")
    master_index = {
        'generated': timestamp,
        'dataset': 'nvidia/NitroGen',
        'source': 'https://huggingface.co/datasets/nvidia/NitroGen',
        'total_games': total_games,
        'total_videos': total_videos,
        'total_hours': round(total_hours, 2),
        'total_chunks': total_chunks,
        'total_frames': total_frames,
        'games': {}
    }

    for game_name, game_data in games.items():
        master_index['games'][game_name] = {
            'video_count': game_data['video_count'],
            'total_hours': round(game_data['total_hours'], 2),
            'total_chunks': game_data['total_chunks'],
            'total_frames': game_data['total_frames'],
            'shards': game_data['shards'],
            'videos': {
                vid: {
                    'url': data['url'],
                    'source': data['source'],
                    'resolution': data['resolution'],
                    'duration_hours': round(data['total_duration'] / 3600, 4),
                    'total_frames': data['total_frames'],
                    'chunk_count': len(data['chunks'])
                }
                for vid, data in game_data['videos'].items()
            }
        }

    with open(output_dir / 'master_index.json', 'w', encoding='utf-8') as f:
        json.dump(master_index, f, indent=2, ensure_ascii=False)

    # 2. Games Summary (CSV) - One row per game, sorted by hours
    print("Generating games_summary.csv...")
    games_sorted = sorted(games.items(), key=lambda x: x[1]['total_hours'], reverse=True)

    with open(output_dir / 'games_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Game', 'Videos', 'Hours', 'Chunks', 'Frames', 'Shard_Count', 'Shards'])

        for rank, (game_name, data) in enumerate(games_sorted, 1):
            writer.writerow([
                rank,
                game_name,
                data['video_count'],
                round(data['total_hours'], 2),
                data['total_chunks'],
                data['total_frames'],
                len(data['shards']),
                ';'.join(data['shards'])
            ])

    # 3. Videos List (CSV) - One row per video
    print("Generating videos_list.csv...")
    with open(output_dir / 'videos_list.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Game', 'Video_ID', 'URL', 'Duration_Hours', 'Chunks', 'Frames', 'Source', 'Resolution'])

        for game_name, data in sorted(games.items()):
            for vid, vdata in data['videos'].items():
                res = vdata.get('resolution')
                res_str = f"{res[0]}x{res[1]}" if res and isinstance(res, (list, tuple)) and len(res) >= 2 else ''
                writer.writerow([
                    game_name,
                    vid,
                    vdata['url'] or '',
                    round(vdata['total_duration'] / 3600, 4),
                    len(vdata['chunks']),
                    vdata['total_frames'],
                    vdata['source'],
                    res_str
                ])

    # 4. Shard Map (JSON) - Which games in which shards
    print("Generating shard_map.json...")
    shard_map = {
        'generated': timestamp,
        'total_shards': len(shard_contents),
        'description': 'Maps each shard to the list of games it contains',
        'shards': shard_contents
    }

    with open(output_dir / 'shard_map.json', 'w', encoding='utf-8') as f:
        json.dump(shard_map, f, indent=2, ensure_ascii=False)

    # 5. Game to Shards Map (JSON) - Reverse lookup (most useful!)
    print("Generating game_shards.json...")
    game_shards = {
        'generated': timestamp,
        'total_games': total_games,
        'description': 'Maps each game to the list of shards containing its data',
        'usage': 'To download data for a specific game, download only the shards listed for that game',
        'games': {
            game: data['shards']
            for game, data in sorted(games.items())
        }
    }

    with open(output_dir / 'game_shards.json', 'w', encoding='utf-8') as f:
        json.dump(game_shards, f, indent=2, ensure_ascii=False)

    # 6. Games by Hours (Markdown) - Human readable
    print("Generating games_by_hours.md...")
    with open(output_dir / 'games_by_hours.md', 'w', encoding='utf-8') as f:
        f.write("# NitroGen Dataset - Games by Hours\n\n")
        f.write(f"*Generated: {timestamp}*\n\n")
        f.write("## Dataset Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Games | {total_games:,} |\n")
        f.write(f"| Total Videos | {total_videos:,} |\n")
        f.write(f"| Total Hours | {total_hours:,.1f} |\n")
        f.write(f"| Total Chunks | {total_chunks:,} |\n")
        f.write(f"| Total Frames | {total_frames:,} |\n\n")

        # Top 100
        f.write("## Top 100 Games by Hours\n\n")
        f.write("| Rank | Game | Videos | Hours | Chunks | Shards |\n")
        f.write("|------|------|--------|-------|--------|--------|\n")

        for rank, (game_name, data) in enumerate(games_sorted[:100], 1):
            # Escape pipe characters in game names
            safe_name = game_name.replace('|', '\\|')
            f.write(f"| {rank} | {safe_name} | {data['video_count']:,} | {data['total_hours']:.1f} | {data['total_chunks']:,} | {len(data['shards'])} |\n")

        # Games 101-500
        if len(games_sorted) > 100:
            f.write("\n## Games 101-500 by Hours\n\n")
            f.write("| Rank | Game | Videos | Hours |\n")
            f.write("|------|------|--------|-------|\n")
            for rank, (game_name, data) in enumerate(games_sorted[100:500], 101):
                safe_name = game_name.replace('|', '\\|')
                f.write(f"| {rank} | {safe_name} | {data['video_count']:,} | {data['total_hours']:.1f} |\n")

        # Full alphabetical list
        f.write("\n## All Games (Alphabetical)\n\n")
        f.write("<details>\n<summary>Click to expand full list</summary>\n\n")
        f.write("| Game | Videos | Hours | Shards |\n")
        f.write("|------|--------|-------|--------|\n")

        for game_name in sorted(games.keys()):
            data = games[game_name]
            safe_name = game_name.replace('|', '\\|')
            shards_str = ', '.join(data['shards'][:3])
            if len(data['shards']) > 3:
                shards_str += f" (+{len(data['shards'])-3})"
            f.write(f"| {safe_name} | {data['video_count']:,} | {data['total_hours']:.1f} | {shards_str} |\n")

        f.write("\n</details>\n")

    # 7. README
    print("Generating README.md...")
    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write("# NitroGen Dataset Master Index\n\n")
        f.write("A comprehensive, searchable index of NVIDIA's NitroGen gaming dataset.\n\n")
        f.write("## Why This Index?\n\n")
        f.write("The official NitroGen dataset on HuggingFace has no way to search by game.\n")
        f.write("You would have to download all 160GB just to find out which shards contain your game.\n")
        f.write("This index solves that problem.\n\n")

        f.write("## Dataset Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Games | {total_games:,} |\n")
        f.write(f"| Total Videos | {total_videos:,} |\n")
        f.write(f"| Total Hours | {total_hours:,.1f} |\n")
        f.write(f"| Total Chunks | {total_chunks:,} |\n")
        f.write(f"| Total Frames | {total_frames:,} |\n\n")

        f.write("## Files\n\n")
        f.write("| File | Description | Format |\n")
        f.write("|------|-------------|--------|\n")
        f.write("| `master_index.json` | Complete index with all video/chunk details | JSON |\n")
        f.write("| `games_summary.csv` | One row per game, sorted by hours | CSV |\n")
        f.write("| `videos_list.csv` | One row per video with URLs | CSV |\n")
        f.write("| `shard_map.json` | Shard → Games mapping | JSON |\n")
        f.write("| `game_shards.json` | Game → Shards mapping (most useful!) | JSON |\n")
        f.write("| `games_by_hours.md` | Human-readable ranking | Markdown |\n\n")

        f.write("## Quick Start\n\n")
        f.write("### Find which shards contain your game\n\n")
        f.write("```python\n")
        f.write("import json\n\n")
        f.write("with open('game_shards.json') as f:\n")
        f.write("    data = json.load(f)\n\n")
        f.write("# Find shards for Hades\n")
        f.write("hades_shards = data['games'].get('Hades', [])\n")
        f.write("print(f'Hades is in shards: {hades_shards}')\n")
        f.write("# Output: Hades is in shards: ['SHARD_0012', 'SHARD_0034', ...]\n")
        f.write("```\n\n")

        f.write("### Download only the shards you need\n\n")
        f.write("```python\n")
        f.write("from huggingface_hub import hf_hub_download\n\n")
        f.write("for shard in hades_shards:\n")
        f.write("    hf_hub_download(\n")
        f.write("        repo_id='nvidia/NitroGen',\n")
        f.write("        repo_type='dataset',\n")
        f.write("        filename=f'actions/{shard}.tar.gz',\n")
        f.write("        local_dir='./data'\n")
        f.write("    )\n")
        f.write("```\n\n")

        f.write("### Get video URLs for a game\n\n")
        f.write("```python\n")
        f.write("import json\n\n")
        f.write("with open('master_index.json') as f:\n")
        f.write("    index = json.load(f)\n\n")
        f.write("hades = index['games'].get('Hades', {})\n")
        f.write("for video_id, video_data in hades.get('videos', {}).items():\n")
        f.write("    print(f\"{video_id}: {video_data['url']}\")\n")
        f.write("```\n\n")

        f.write("## Source\n\n")
        f.write("Data extracted from: https://huggingface.co/datasets/nvidia/NitroGen\n\n")
        f.write("## License\n\n")
        f.write("The NitroGen dataset is released under CC BY-NC 4.0.\n")
        f.write("This index is provided for convenience and does not contain the actual training data.\n\n")
        f.write(f"*Index generated: {timestamp}*\n")

    print(f"\nAll index files generated in: {output_dir}")

    return {
        'total_games': total_games,
        'total_videos': total_videos,
        'total_hours': total_hours,
        'total_chunks': total_chunks,
        'total_frames': total_frames
    }

def main():
    parser = argparse.ArgumentParser(description='Generate NitroGen master index')
    parser.add_argument('--metadata-dir', default='nvidia_index/metadata', help='Metadata directory')
    parser.add_argument('--output-dir', default='nvidia_index/index', help='Output directory')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NitroGen Index Generator")
    print(f"{'='*60}")
    print(f"  Metadata directory: {args.metadata_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        print("\nRun the extraction step first:")
        print("python scripts/nvidia_index/extract_metadata.py")
        return

    # Load all metadata
    games, shard_contents, all_entries, errors = load_all_metadata(args.metadata_dir)

    print(f"\n{'='*60}")
    print(f"  Metadata Summary")
    print(f"{'='*60}")
    print(f"  Total games found: {len(games)}")
    print(f"  Total entries: {len(all_entries):,}")
    print(f"  Total shards: {len(shard_contents)}")
    if errors:
        print(f"  Parse errors: {len(errors)}")
    print(f"{'='*60}\n")

    if len(games) == 0:
        print("No games found! Check the metadata directory.")
        return

    # Generate index files
    stats = generate_master_index(games, shard_contents, all_entries, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Index Generation Complete")
    print(f"{'='*60}")
    print(f"  Games indexed: {stats['total_games']:,}")
    print(f"  Videos indexed: {stats['total_videos']:,}")
    print(f"  Hours indexed: {stats['total_hours']:,.1f}")
    print(f"  Chunks indexed: {stats['total_chunks']:,}")
    print(f"  Frames indexed: {stats['total_frames']:,}")
    print(f"{'='*60}\n")

    # Show output file sizes
    output_dir = Path(args.output_dir)
    print("Output files:")
    for f in sorted(output_dir.glob('*')):
        size_kb = f.stat().st_size / 1024
        if size_kb > 1024:
            print(f"  {f.name}: {size_kb/1024:.1f} MB")
        else:
            print(f"  {f.name}: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
