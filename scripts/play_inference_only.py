"""
Lightweight Inference-Only Script for NitroGen

Pure inference with FPS counter - no recording, no saving.
Optimized for testing zero-shot performance on different games.

Usage:
    python scripts/play_inference_only.py --process "DarkSoulsIII.exe"
    python scripts/play_inference_only.py --process "Celeste.exe"

Controls:
    F1: Pause/Resume AI (you can play manually when paused)
    F2: Exit
"""

import os
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
    print("Warning: 'keyboard' package not installed. Hotkeys disabled.")

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="NitroGen Inference Only")
parser.add_argument("--process", type=str, required=True, help="Game executable name (e.g., DarkSoulsIII.exe)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu buttons (START, BACK, GUIDE)")
args = parser.parse_args()

# Connect to model server
print(f"Connecting to inference server on port {args.port}...")
policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

print(f"Connected! Action downsample ratio: {action_downsample_ratio}")

BUTTON_PRESS_THRES = 0.5
TOKEN_SET = BUTTON_ACTION_TOKENS
NO_MENU = not args.allow_menu

def preprocess_img(main_image):
    main_cv = cv2.cvtColor(np.array(main_image), cv2.COLOR_RGB2BGR)
    final_image = cv2.resize(main_cv, (256, 256), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

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

# FPS tracking
class FPSCounter:
    def __init__(self, window_size=60):
        self.timestamps = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)

    def tick(self, inference_time=None):
        self.timestamps.append(time.time())
        if inference_time:
            self.inference_times.append(inference_time)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def get_avg_inference_time(self):
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

fps_counter = FPSCounter()

print(f"\nStarting NitroGen on {args.process}")
print("=" * 60)
print("Controls:")
print("  F1: Pause/Resume AI")
print("  F2: Exit")
print("=" * 60)

for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

# Initialize environment
env = GamepadEnv(
    game=args.process,
    game_speed=1.0,
    env_fps=60,
    async_mode=True,
)

env.reset()
env.pause()

# Initial observation
obs, reward, terminated, truncated, info = env.step(action=zero_action)

paused = False
step_count = 0
last_status_time = time.time()

print("\nAI is playing! Press F1 to pause, F2 to exit.\n")

try:
    while True:
        # Check hotkeys
        if HAS_KEYBOARD:
            if keyboard.is_pressed('F1'):
                paused = not paused
                status = "PAUSED (you play)" if paused else "AI PLAYING"
                print(f"\n[{status}]")
                time.sleep(0.3)

            if keyboard.is_pressed('F2'):
                print("\nExiting...")
                break

        if paused:
            # Let human play - just step with zero action to keep game responsive
            obs, reward, terminated, truncated, info = env.step(action=zero_action)
            time.sleep(0.016)  # ~60fps
            continue

        # Preprocess observation
        obs_processed = preprocess_img(obs)

        # Run inference
        inference_start = time.time()
        pred = policy.predict(obs_processed)
        inference_time = time.time() - inference_start

        j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]
        n = len(buttons)

        # Convert predictions to actions
        env_actions = []
        for i in range(n):
            move_action = zero_action.copy()

            xl, yl = j_left[i]
            xr, yr = j_right[i]
            move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.int64)
            move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.int64)
            move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.int64)
            move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.int64)

            button_vector = buttons[i]
            for name, value in zip(TOKEN_SET, button_vector):
                if "TRIGGER" in name:
                    move_action[name] = np.array([int(value * 255)], dtype=np.int64)
                else:
                    move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0

            env_actions.append(move_action)

        # Execute actions
        for i, a in enumerate(env_actions):
            if NO_MENU:
                a["GUIDE"] = 0
                a["START"] = 0
                a["BACK"] = 0

            for _ in range(action_downsample_ratio):
                obs, reward, terminated, truncated, info = env.step(action=a)
                fps_counter.tick(inference_time if _ == 0 else None)

        step_count += 1

        # Print status every second
        if time.time() - last_status_time >= 1.0:
            fps = fps_counter.get_fps()
            avg_inf = fps_counter.get_avg_inference_time() * 1000
            print(f"Step: {step_count:5d} | FPS: {fps:5.1f} | Inference: {avg_inf:5.1f}ms | Actions/chunk: {n}")
            last_status_time = time.time()

except KeyboardInterrupt:
    print("\nInterrupted!")

finally:
    env.unpause()
    env.close()
    print("\nSession ended.")
    print(f"Total steps: {step_count}")
    print(f"Average FPS: {fps_counter.get_fps():.1f}")
    print(f"Average inference time: {fps_counter.get_avg_inference_time()*1000:.1f}ms")
