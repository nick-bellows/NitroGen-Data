# NitroGen - Hades

Game-specific configuration and data for Hades.

## Workflow

### 1. Record Your Gameplay (Behavior Cloning)
```
record.bat
```
- No AI running - just records your screen + controller
- Press F5 to start/stop recording
- Press F6 to quit
- Captures at 30 FPS for training

### 2. Train on Your Data
```
train.bat
```
- Fine-tunes NitroGen on your recorded gameplay
- Select which recording session to train on

### 3. Let AI Play
```
play.bat
```
- Runs the AI on Hades
- Requires inference server running

## Scripts

| Script | Description |
|--------|-------------|
| `record.bat` | Record YOUR gameplay (no AI) |
| `train.bat` | Fine-tune model on recordings |
| `play.bat` | Let AI play with trained model |

## Folders

- `recordings/` - Your recorded gameplay sessions
- `checkpoints/` - Fine-tuned model weights

## Tips for Recording

1. **God Mode**: Enable in Hades settings for longer runs
2. **Play naturally**: Don't try to play perfectly
3. **Variety**: Record multiple runs with different weapons/builds
4. **Duration**: 30-60 minutes of gameplay is a good start
5. **Quality over quantity**: Clean, consistent gameplay trains better

## Hades-Specific Notes

- **Genre**: Action Roguelike (similar to hack-and-slash in training data)
- **Camera**: Fixed isometric view (consistent for AI)
- **Combat**: Fast-paced, lots of dodging and attacking
- **Focus areas for training**:
  - Dash timing and direction
  - Attack patterns
  - Boon selection (if recording menu navigation)

## After Recording

Your recordings are saved to `recordings/Hades_TIMESTAMP/`:
- `frames/` - PNG images (256x256)
- `samples.json` - Frame + action pairs
- `metadata.json` - Recording info

The format is compatible with `dagger_train.py`.
