"""
NitroGen Fast Inference Player

Pure inference with maximum performance optimizations:
- No recording/saving (zero I/O overhead)
- Action chunking (use N actions per inference)
- Real-time FPS counter
- Minimal code path

Usage:
    python scripts/play_fast.py --process "DarkSoulsIII.exe" --actions-per-chunk 4

Controls:
    F1: Pause/Resume AI
    F2: Exit
"""

import sys
import time
from collections import OrderedDict, deque

import cv2
import numpy as np
from PIL import Image

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("Warning: 'keyboard' not installed. Hotkeys disabled. Install: pip install keyboard")

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="NitroGen Fast Inference")
parser.add_argument("--process", type=str, required=True, help="Game executable (e.g., DarkSoulsIII.exe)")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument("--actions-per-chunk", type=int, default=4, help="Actions to use per inference (1-16)")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu buttons")
parser.add_argument("--game-speed", type=float, default=1.0, help="Game speed multiplier")
parser.add_argument("--joystick-amplify", type=float, default=2.5, help="Joystick amplification factor (default 2.5)")
parser.add_argument("--joystick-min", type=float, default=0.35, help="Minimum joystick output magnitude (default 0.35)")
parser.add_argument("--no-deadzone-comp", action="store_true", help="Disable deadzone compensation")
parser.add_argument("--verbose", action="store_true", help="Show detailed action values")
args = parser.parse_args()

# Clamp actions per chunk
args.actions_per_chunk = max(1, min(16, args.actions_per_chunk))

BUTTON_PRESS_THRES = 0.5
TOKEN_SET = BUTTON_ACTION_TOKENS
NO_MENU = not args.allow_menu

# Deadzone compensation settings
# Games typically have deadzones around 8000-10000 (out of 32767)
# The model often outputs small values that fall within this deadzone
JOYSTICK_DEADZONE = 0.01  # Ignore values below this (noise threshold) - lowered to capture small movements
JOYSTICK_MIN_OUTPUT = args.joystick_min  # Minimum output magnitude when above noise threshold
JOYSTICK_AMPLIFY = args.joystick_amplify  # Amplification factor for values above threshold
DEADZONE_COMP_ENABLED = not args.no_deadzone_comp

def apply_deadzone_compensation(x, y):
    """
    Compensate for game controller deadzones.

    The model often outputs small joystick values that fall within the game's
    deadzone (typically ~8000-10000 out of 32767, or ~0.25-0.30 normalized).

    This function:
    1. Ignores very small values (noise)
    2. Amplifies small-but-intentional movements to exceed the deadzone
    3. Preserves direction while boosting magnitude
    """
    # If disabled, return original values
    if not DEADZONE_COMP_ENABLED:
        return x, y

    magnitude = np.sqrt(x*x + y*y)

    # Below noise threshold - return zero
    if magnitude < JOYSTICK_DEADZONE:
        return 0.0, 0.0

    # Normalize direction
    if magnitude > 0:
        dir_x = x / magnitude
        dir_y = y / magnitude
    else:
        return 0.0, 0.0

    # Amplify and ensure minimum output
    new_magnitude = max(magnitude * JOYSTICK_AMPLIFY, JOYSTICK_MIN_OUTPUT)

    # Clamp to valid range
    new_magnitude = min(new_magnitude, 1.0)

    return dir_x * new_magnitude, dir_y * new_magnitude

def preprocess_img(img):
    """Fast image preprocessing."""
    arr = np.array(img)
    resized = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)

zero_action = OrderedDict([
    ("WEST", 0), ("SOUTH", 0), ("BACK", 0),
    ("DPAD_DOWN", 0), ("DPAD_LEFT", 0), ("DPAD_RIGHT", 0), ("DPAD_UP", 0),
    ("GUIDE", 0),
    ("AXIS_LEFTX", np.array([0], dtype=np.int64)),
    ("AXIS_LEFTY", np.array([0], dtype=np.int64)),
    ("LEFT_SHOULDER", 0),
    ("LEFT_TRIGGER", np.array([0], dtype=np.int64)),
    ("AXIS_RIGHTX", np.array([0], dtype=np.int64)),
    ("AXIS_RIGHTY", np.array([0], dtype=np.int64)),
    ("LEFT_THUMB", 0), ("RIGHT_THUMB", 0),
    ("RIGHT_SHOULDER", 0),
    ("RIGHT_TRIGGER", np.array([0], dtype=np.int64)),
    ("START", 0), ("EAST", 0), ("NORTH", 0),
])

class PerfCounter:
    """Lightweight performance counter."""
    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.inf_times = deque(maxlen=window)

    def tick(self, inf_time=None):
        self.times.append(time.perf_counter())
        if inf_time is not None:
            self.inf_times.append(inf_time)

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0] + 1e-9)

    @property
    def avg_inference_ms(self):
        if not self.inf_times:
            return 0.0
        return sum(self.inf_times) / len(self.inf_times) * 1000

# Connect to server
print(f"Connecting to server on port {args.port}...")
try:
    policy = ModelClient(port=args.port)
    policy.reset()
    info = policy.info()
except Exception as e:
    print(f"ERROR: Could not connect to server: {e}")
    print("Make sure start_server_optimized.bat is running!")
    sys.exit(1)

print(f"Connected!")

# Print config
print()
print("=" * 60)
print("NitroGen Fast Inference")
print("=" * 60)
print(f"Game: {args.process}")
print(f"Actions per chunk: {args.actions_per_chunk}")
print(f"Server optimizations: {info.get('optimizations', 'unknown')}")
print()
if DEADZONE_COMP_ENABLED:
    print(f"Deadzone compensation: ENABLED (amplify={JOYSTICK_AMPLIFY}x, min={JOYSTICK_MIN_OUTPUT})")
else:
    print("Deadzone compensation: DISABLED")
print()
print("Controls: F1=Pause/Resume  F2=Exit")
print("=" * 60)
print()
print("IMPORTANT: Keep Dark Souls III window in FOREGROUND!")
print("The AI sees whatever is on screen at the game window location.")
print("=" * 60)

# Countdown
for i in range(3, 0, -1):
    print(f"{i}... (switch to game window NOW)")
    time.sleep(1)

# Initialize game environment
try:
    env = GamepadEnv(
        game=args.process,
        game_speed=args.game_speed,
        env_fps=60,
        async_mode=True,
    )
except Exception as e:
    print(f"ERROR: Could not find game: {e}")
    print(f"Make sure {args.process} is running!")
    sys.exit(1)

env.reset()

# IMPORTANT: Don't pause the game - we want real-time play!
# The speedhack DLL freezes the game at speed 0.0
# Instead, set speed to 1.0 (normal) and keep it running
env.unpause()

# Get initial observation
obs = env.render()

perf = PerfCounter()
paused = False
total_actions = 0
start_time = time.time()
last_print = start_time

print("\nAI Playing...\n")

try:
    while True:
        # Hotkeys
        if HAS_KEYBOARD:
            if keyboard.is_pressed('F1'):
                paused = not paused
                print(f"\n{'[PAUSED - You play]' if paused else '[AI PLAYING]'}\n")
                time.sleep(0.3)
            if keyboard.is_pressed('F2'):
                break

        # Paused - let human play (reset gamepad to neutral, get fresh observation)
        if paused:
            env.gamepad_emulator.reset()
            obs = env.render()
            time.sleep(0.016)
            continue

        # Preprocess
        obs_img = preprocess_img(obs)

        # Inference
        t0 = time.perf_counter()
        pred = policy.predict(obs_img)
        inf_time = time.perf_counter() - t0

        j_left = pred["j_left"]
        j_right = pred["j_right"]
        buttons = pred["buttons"]

        # Use N actions from the chunk
        n_actions = min(args.actions_per_chunk, len(buttons))

        for i in range(n_actions):
            action = zero_action.copy()

            # Joysticks - apply deadzone compensation to boost small values
            xl, yl = j_left[i]
            xr, yr = j_right[i]

            # Apply deadzone compensation (amplify small movements)
            xl_comp, yl_comp = apply_deadzone_compensation(xl, yl)
            xr_comp, yr_comp = apply_deadzone_compensation(xr, yr)

            action["AXIS_LEFTX"] = np.array([int(xl_comp * 32767)], dtype=np.int64)
            action["AXIS_LEFTY"] = np.array([int(yl_comp * 32767)], dtype=np.int64)
            action["AXIS_RIGHTX"] = np.array([int(xr_comp * 32767)], dtype=np.int64)
            action["AXIS_RIGHTY"] = np.array([int(yr_comp * 32767)], dtype=np.int64)

            # Buttons - TOKEN_SET has 21 values from BUTTON_ACTION_TOKENS
            # Official HuggingFace format has 17 buttons; the extra 4 (RIGHT_BOTTOM, etc.) are padding
            # Mapping: model outputs buttons in TOKEN_SET order, we convert to gamepad actions
            for name, value in zip(TOKEN_SET, buttons[i]):
                # Skip padding tokens that don't map to real buttons
                if name in ['RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP']:
                    continue
                if "TRIGGER" in name:
                    action[name] = np.array([int(value * 255)], dtype=np.int64)
                else:
                    action[name] = 1 if value > BUTTON_PRESS_THRES else 0

            # Block menu buttons
            if NO_MENU:
                action["GUIDE"] = 0
                action["START"] = 0
                action["BACK"] = 0

            # Execute action - directly control gamepad without pausing game
            env.gamepad_emulator.step(action)
            total_actions += 1
            perf.tick(inf_time if i == 0 else None)

            # Small delay between actions in chunk
            time.sleep(0.016)  # ~60 FPS timing

        # Get new observation for next inference
        obs = env.render()

        # Print stats every second
        now = time.time()
        if now - last_print >= 1.0:
            elapsed = now - start_time
            # Show action values in verbose mode
            if args.verbose:
                # Show both raw model output and compensated values
                lx_model = int(xl * 32767)
                ly_model = int(yl * 32767)
                lx_out = int(xl_comp * 32767)
                ly_out = int(yl_comp * 32767)
                rx_out = int(xr_comp * 32767)
                ry_out = int(yr_comp * 32767)
                print(f"FPS: {perf.fps:5.1f} | Model L:({lx_model:+6d},{ly_model:+6d}) -> Out:({lx_out:+6d},{ly_out:+6d}) | R:({rx_out:+6d},{ry_out:+6d})")
            else:
                print(f"FPS: {perf.fps:5.1f} | Inference: {perf.avg_inference_ms:5.1f}ms | Actions: {total_actions:,} | Time: {elapsed:.0f}s")
            last_print = now

except KeyboardInterrupt:
    pass

finally:
    env.unpause()
    env.close()

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Session Complete")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total actions: {total_actions:,}")
    print(f"Average FPS: {perf.fps:.1f}")
    print(f"Average inference: {perf.avg_inference_ms:.1f}ms")
    print("=" * 60)
