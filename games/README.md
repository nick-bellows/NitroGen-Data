# NitroGen Games

Game-specific configurations, scripts, and training data.

## Available Games

| Folder | Game | Training Data % |
|--------|------|-----------------|
| [darksouls3/](darksouls3/) | Dark Souls III | ~34.9% (Action-RPG) |
| [celeste/](celeste/) | Celeste | ~18.4% (Platformer) |

## Adding a New Game

1. Create a new folder: `games/yourgame/`

2. Copy the template structure from an existing game:
   - `play.bat` - Main player script
   - `dagger_collect.bat` - Data collection
   - `dagger_train.bat` - Fine-tuning
   - `README.md` - Game-specific notes

3. Update the process name in each script (e.g., `YourGame.exe`)

4. Create subfolders:
   - `dagger_data/` - For collected training data
   - `checkpoints/` - For fine-tuned models

## Game Process Names

Common games and their executables:

| Game | Executable |
|------|------------|
| Dark Souls III | `DarkSoulsIII.exe` |
| Elden Ring | `eldenring.exe` |
| Sekiro | `sekiro.exe` |
| Celeste | `Celeste.exe` |
| Hollow Knight | `hollow_knight.exe` |
| Cuphead | `Cuphead.exe` |

## Tips

- **Action-RPGs** (Dark Souls, Elden Ring): Focus DAgger on dodge timing, combat positioning
- **Platformers** (Celeste, Hollow Knight): Focus on precise jumps and movement
- **Racing games**: Less training data in base model, may need more DAgger data
