# NitroGen - Celeste

Game-specific configuration and data for Celeste.

## Quick Start

1. Start the inference server from project root:
   ```
   start_server.bat
   ```

2. Launch Celeste and get into gameplay

3. Run `play.bat` from this folder

## Scripts

| Script | Description |
|--------|-------------|
| `play.bat` | Main AI player (optimized) |
| `play_original.bat` | Original script (slower but compatible) |
| `dagger_collect.bat` | Record your corrections while AI plays |
| `dagger_train.bat` | Fine-tune model on collected data |

## Notes

Celeste is a platformer which is 18.4% of NitroGen's training data.
The model should have reasonable zero-shot performance.

## Folders

- `dagger_data/` - Collected training data
- `checkpoints/` - Fine-tuned model checkpoints

## Tips

1. **Precision platforming:** Celeste requires precise timing - DAgger training helps
2. **Dash mechanics:** The AI may struggle with dash chains initially
3. **Collection focus:** When collecting DAgger data, focus on specific jumps/sections
