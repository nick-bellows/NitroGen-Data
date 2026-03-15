"""
Debug script to see what actions NitroGen is outputting.

This will show the actual joystick and button values being predicted.
"""

import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="Debug NitroGen Actions")
parser.add_argument("--process", type=str, required=True, help="Game executable")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument("--frames", type=int, default=10, help="Number of frames to debug")
args = parser.parse_args()

BUTTON_PRESS_THRES = 0.5
TOKEN_SET = BUTTON_ACTION_TOKENS

def preprocess_img(img):
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

# Connect
print(f"Connecting to server on port {args.port}...")
try:
    policy = ModelClient(port=args.port)
    policy.reset()
    print("Connected!")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Init game env
print(f"\nConnecting to {args.process}...")
try:
    env = GamepadEnv(
        game=args.process,
        game_speed=1.0,
        env_fps=60,
        async_mode=True,
    )
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

env.reset()
env.pause()

obs, _, _, _, _ = env.step(action=zero_action)

print(f"\nCapturing {args.frames} frames and showing predicted actions...\n")
print("=" * 80)

try:
    for frame_num in range(args.frames):
        # Capture and preprocess
        obs_img = preprocess_img(obs)

        # Save frame for inspection
        obs_img.save(f"debug_frame_{frame_num:02d}.png")

        # Get prediction
        pred = policy.predict(obs_img)

        j_left = pred["j_left"]
        j_right = pred["j_right"]
        buttons = pred["buttons"]

        print(f"\nFrame {frame_num + 1}:")
        print(f"  Actions in chunk: {len(buttons)}")

        # Show first action details
        print(f"\n  First action in chunk:")
        print(f"    Left Joystick:  X={j_left[0][0]:+.3f}, Y={j_left[0][1]:+.3f}")
        print(f"    Right Joystick: X={j_right[0][0]:+.3f}, Y={j_right[0][1]:+.3f}")

        # Convert joystick to actual values
        lx = int(j_left[0][0] * 32767)
        ly = int(j_left[0][1] * 32767)
        rx = int(j_right[0][0] * 32767)
        ry = int(j_right[0][1] * 32767)
        print(f"    Left Joystick (raw):  X={lx:+6d}, Y={ly:+6d}")
        print(f"    Right Joystick (raw): X={rx:+6d}, Y={ry:+6d}")

        # Show button activations
        print(f"\n    Buttons (threshold={BUTTON_PRESS_THRES}):")
        active_buttons = []
        for name, value in zip(TOKEN_SET, buttons[0]):
            if "TRIGGER" in name:
                trigger_val = int(value * 255)
                if trigger_val > 10:
                    active_buttons.append(f"{name}={trigger_val}")
            else:
                if value > BUTTON_PRESS_THRES:
                    active_buttons.append(f"{name}({value:.2f})")

        if active_buttons:
            print(f"      Active: {', '.join(active_buttons)}")
        else:
            print(f"      Active: NONE")

        # Show all button values
        print(f"\n    All button values:")
        for i, (name, value) in enumerate(zip(TOKEN_SET, buttons[0])):
            print(f"      {name:20s}: {value:.4f}")

        # Step to get next frame
        obs, _, _, _, _ = env.step(action=zero_action)
        time.sleep(0.5)

        print("=" * 80)

finally:
    env.unpause()
    env.close()

print(f"\nDebug frames saved as debug_frame_XX.png")
print("Check if the frames look correct (game is visible, not menu)")
