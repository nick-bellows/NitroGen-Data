# NitroGen - Dark Souls III

Game-specific configuration and data for Dark Souls III.

## Quick Start

1. Start the inference server from project root:
   ```
   start_server.bat
   ```

2. Launch Dark Souls III and load into the game (past menus)

3. **Important:** Unplug your real Xbox controller!

4. Run `play.bat` from this folder

## Scripts

| Script | Description |
|--------|-------------|
| `play.bat` | Main AI player with optimized settings |
| `play_verbose.bat` | Shows detailed joystick values for debugging |
| `dagger_collect.bat` | Record your corrections while AI plays |
| `dagger_train.bat` | Fine-tune model on collected data |

## Settings

Dark Souls III specific optimizations:
- **Deadzone compensation:** 2.5x amplification, 0.35 minimum
- **Actions per chunk:** 4 (good balance of speed vs responsiveness)
- **Menu buttons:** Disabled (prevents accidental pause)

## Folders

- `dagger_data/` - Collected training data from your sessions
- `checkpoints/` - Fine-tuned model checkpoints

## Tips

1. **Window mode:** Use Borderless Windowed for best screen capture
2. **Controller:** Must unplug real controller - DS3 only sees one at a time
3. **Starting position:** Start in a safe area for initial testing
4. **DAgger collection:** Focus on correcting specific mistakes (dodging, movement)
