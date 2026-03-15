# NitroGen Project Structure

**Base Template | Per-Game Repos | Menu Handling**

---

## Repository Architecture

Three-tier structure: Base template (shared code) → Per-game repos (game-specific) → Shared weights.

| Repository | Purpose | Contents |
|------------|---------|----------|
| `nitrogen-base` | Template repo, shared utilities | Core inference, training scripts, utilities |
| `nitrogen-elden-ring` | Game-specific repo | Config, fine-tuned weights, menu templates |
| `nitrogen-sekiro` | Game-specific repo | Config, parry-trained weights, menu templates |
| `nitrogen-weights` | Shared large files (Git LFS) | Base model weights, shared across games |

---

## Base Template Structure

```
nitrogen-base/
├── src/
│   ├── inference/
│   │   ├── agent.py           # Main NitroGen agent class
│   │   ├── screen_capture.py  # Screen capture utilities
│   │   └── controller.py      # ViGEmBus controller output
│   ├── training/
│   │   ├── behavior_cloning.py
│   │   ├── dagger.py
│   │   └── rl_training.py
│   ├── menu_detection/
│   │   ├── detector.py        # Base menu detection class
│   │   └── handlers.py        # Menu exit handlers
│   └── utils/
│       ├── recording.py       # OBS/gameplay recording
│       └── action_extraction.py
├── configs/
│   └── base_config.yaml       # Default settings
├── scripts/
│   ├── run_inference.py       # Main entry point
│   ├── train.py               # Training entry point
│   └── setup_new_game.py      # Script to clone for new game
├── weights/                   # Symlink to nitrogen-weights/
├── requirements.txt
├── README.md                  # Template README for YouTube
└── .gitignore
```

---

## Per-Game Repository Structure

Example: `nitrogen-elden-ring/`

```
nitrogen-elden-ring/
├── src/                       # Inherited from base (git submodule)
├── game_specific/
│   ├── config.yaml            # Elden Ring specific settings
│   ├── menu_templates/        # Screenshots of menus to detect
│   │   ├── pause_menu.png
│   │   ├── death_screen.png
│   │   ├── level_up.png
│   │   └── bonfire.png
│   ├── menu_handlers.py       # Game-specific menu exit logic
│   └── reward_functions.py    # RL reward design (if training)
├── weights/
│   ├── base/ -> ../nitrogen-weights/  # Symlink to shared
│   └── fine_tuned/            # Game-specific fine-tuned weights
│       └── elden_ring_v1.pt
├── data/
│   ├── recordings/            # Your gameplay recordings
│   └── extracted/             # Processed training data
├── results/
│   ├── videos/                # Recorded AI gameplay
│   └── logs/                  # Training logs, metrics
├── README.md                  # Game-specific documentation
└── run.py                     # Quick start script
```

---

## Workflow: Adding a New Game

```bash
# 1. Clone from base template
python scripts/setup_new_game.py --name "sekiro"

# 2. Capture menu screenshots
# Take screenshots of: pause menu, death screen, etc.
# Save to: game_specific/menu_templates/

# 3. Configure game settings
# Edit: game_specific/config.yaml

# 4. Test zero-shot
python run.py --mode inference

# 5. (Optional) Fine-tune
python run.py --mode train --method dagger
```

---

## Menu Handling Without LLMs

Even "perfect" games have menus. Strategies to handle them with pure NitroGen:

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Template Detection** | OpenCV template matching on menu screenshots. When detected → scripted button press. | Death screens, pause menus, loading screens. Most reliable. |
| **Color Histogram** | Menus often have distinct color profiles (dark overlays, specific UI colors). | Pause menus with dark overlays, inventory screens. |
| **Button Spam** | NitroGen trained on data where players spam B/Circle to exit menus. | Simple popup messages, item pickups. Unreliable for complex menus. |
| **Timeout Detection** | If screen static for X seconds + no position change → assume menu → press B. | Fallback when other methods fail. |
| **Manual Mode** | Human handles menus, AI handles combat. Toggle with hotkey. | YouTube content (cleanest), complex menu situations. |

---

## Implementation: Template-Based Menu Detection

```python
# src/menu_detection/detector.py

import cv2
import numpy as np
from pathlib import Path

class MenuDetector:
    def __init__(self, templates_dir: Path, threshold: float = 0.8):
        self.threshold = threshold
        self.templates = {}
        
        # Load all template images
        for template_path in templates_dir.glob('*.png'):
            name = template_path.stem  # 'death_screen', 'pause_menu', etc.
            img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            self.templates[name] = img
    
    def detect(self, frame: np.ndarray) -> str | None:
        """Returns menu type if detected, None if gameplay."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for name, template in self.templates.items():
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > self.threshold:
                return name
        return None  # No menu detected, gameplay mode
```

---

## Menu Exit Handlers

```python
# src/menu_detection/handlers.py

# Define what buttons to press for each menu type
MENU_HANDLERS = {
    # Elden Ring specific
    'death_screen': ['A', 'A'],       # Press A twice to respawn
    'pause_menu': ['B'],              # B to exit pause
    'level_up': ['B', 'B'],           # B twice to exit level up
    'bonfire': ['B'],                 # B to exit bonfire menu
    'item_pickup': ['A'],             # A to dismiss item popup
    
    # Batman Arkham specific (override in game config)
    'retry_screen': ['A'],            # A to retry challenge
}
```

---

## Main Loop Integration

```python
# Main inference loop with menu handling

def run_agent(agent, menu_detector, controller, manual_override_key='F1'):
    manual_mode = False
    
    while True:
        # Check for manual override toggle
        if keyboard.is_pressed(manual_override_key):
            manual_mode = not manual_mode
            print(f"Manual mode: {manual_mode}")
            time.sleep(0.3)  # Debounce
        
        if manual_mode:
            continue  # Human plays, AI waits
        
        frame = capture_screen()
        
        # Check for menu
        menu_type = menu_detector.detect(frame)
        
        if menu_type:
            # Menu detected - use scripted handler
            buttons = MENU_HANDLERS.get(menu_type, ['B'])
            for button in buttons:
                controller.press(button)
                time.sleep(0.2)
        else:
            # Gameplay - use NitroGen
            action = agent.infer(frame)
            controller.execute(action)
```

---

## Game-Specific Configuration Example

```yaml
# game_specific/config.yaml (Elden Ring)

game:
  name: "Elden Ring"
  window_title: "ELDEN RING"
  resolution: [1920, 1080]

inference:
  timesteps: 4              # Balance FPS and quality
  weights: "weights/base/nitrogen.pt"

menu_detection:
  enabled: true
  threshold: 0.85
  templates_dir: "game_specific/menu_templates"

menu_handlers:
  death_screen: ['A', 'A']  # Respawn
  pause_menu: ['B']
  level_up: ['B', 'B']
  bonfire: ['B']

manual_override_key: "F1"   # Toggle human/AI control
```

---

## Elden Ring Menu Templates Needed

| Menu | Template File | Exit Buttons |
|------|---------------|--------------|
| Death/YOU DIED | `death_screen.png` | A, A |
| Pause Menu | `pause_menu.png` | B |
| Level Up (Grace) | `level_up.png` | B, B |
| Site of Grace | `bonfire.png` | B |
| Item Pickup | `item_pickup.png` | A |
| Boss Defeated | `boss_defeated.png` | Wait 5s |

---

## YouTube-Ready README Template

```markdown
# NitroGen - Elden Ring

AI plays Elden Ring using NVIDIA's NitroGen gaming AI.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download weights (if not already present)
python scripts/download_weights.py

# Launch Elden Ring, then run:
python run.py
```

## Controls
- **F1**: Toggle manual/AI control
- **ESC**: Stop the agent

## Video
Watch the AI in action: [YouTube Link]
```

---

## Best Practice: Focus on Combat for Content

For YouTube content, the **Manual Mode** approach is actually best:

1. ✅ Cleaner footage (no AI fumbling in menus)
2. ✅ You can narrate during menu sections
3. ✅ More professional content
4. ✅ Viewers want to see AI **combat**, not menu navigation

Save the automated menu handling for long autonomous runs or live streams.
