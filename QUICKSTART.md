# NitroGen Quick Start Guide

## Prerequisites

- Windows 11 with an NVIDIA GPU (tested on RTX 4080 Super)
- Python 3.10+ with CUDA support
- [ViGEmBus](https://github.com/nefarius/ViGEmBus) driver installed (virtual Xbox controller)
- NitroGen model weights downloaded from [HuggingFace](https://huggingface.co/nvidia/NitroGen)

## Setup

```cmd
git clone https://github.com/nick-bellows/NitroGen-Data.git
cd NitroGen-Data
python -m venv venv
venv\Scripts\pip install -e .
```

Download model weights:
```cmd
pip install huggingface-cli
huggingface-cli download nvidia/NitroGen ng.pt
```

## How to Run NitroGen

### Method 1: Using Batch Files (Easiest)

**Terminal 1 - Start the inference server:**
```cmd
start_server.bat
```

**Terminal 2 - Play any game:**
```cmd
play_any_game.bat
```

### Method 2: Manual Commands

**Terminal 1:**
```cmd
venv\Scripts\python.exe scripts\serve.py "<path_to_ng.pt>"
```

**Terminal 2:**
```cmd
venv\Scripts\python.exe scripts\play.py --process "Celeste.exe"
```

## Important Notes

1. **Launch order matters:**
   - First: Start the inference server (Terminal 1)
   - Second: Launch your game
   - Third: Run the play script (Terminal 2)

2. **Finding the .exe name:**
   - Open Task Manager (Ctrl+Shift+Esc)
   - Find your game process
   - Right-click -> Properties
   - Look for the .exe name in the General tab

3. **Output location:**
   - Videos: `out\ng\`
   - Debug videos: `*_DEBUG.mp4` (with AI visualization)
   - Clean videos: `*_CLEAN.mp4` (1080p gameplay only)
   - Actions: `*_ACTIONS.json` (action log)

4. **Menu actions disabled by default:**
   - START, BACK, GUIDE buttons are blocked
   - Add `--allow-menu` flag if you need them

## Testing Other Games

Replace `Celeste.exe` with your game's executable name:

```cmd
REM Elden Ring
venv\Scripts\python.exe scripts\play.py --process "eldenring.exe"

REM Dark Souls III
venv\Scripts\python.exe scripts\play.py --process "DarkSoulsIII.exe"

REM Batman Arkham Knight
venv\Scripts\python.exe scripts\play.py --process "BatmanAK.exe"
```

## Performance Settings

Default: `timesteps=4` (~20-25 FPS on an RTX 4080)

## Troubleshooting

**"Torch not compiled with CUDA enabled"**
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

**"Process not found"**
- Make sure the game is running
- Verify the exact .exe name in Task Manager

**Low FPS**
- Ensure GPU drivers are up to date
- Monitor GPU usage with `nvidia-smi`

**Screen capture issues**
- Game must be in windowed or borderless windowed mode
- DXCam works best with DirectX 11/12 games

## Documentation

See the `Documentation/` folder for:
- Complete game compatibility list (75+ games)
- Training methods (DAgger, Behavior Cloning)
- Hybrid LLM + NitroGen architecture

---

**Ready to test?** Run `start_server.bat` and let's see the AI play!
