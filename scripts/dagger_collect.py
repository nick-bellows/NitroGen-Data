"""
DAgger Collection with Automatic Human Detection

Just grab your controller to take over - no buttons needed.
Release for 0.5s to let AI resume.

Usage:
    python scripts/dagger_collect.py --process "Hades.exe"

Requirements:
    pip install vgamepad XInput-Python mss

Note: Requires ViGEmBus driver installed for virtual controller.
      Download from: https://github.com/ViGEm/ViGEmBus/releases

Controller Flow:
    Physical Controller → This Script (reads via XInput)
                               ↓
                    Decides: human or AI?
                               ↓
                    Virtual Controller (ViGEmBus) → Game

Both AI and human input go through SAME virtual controller.
Game only sees virtual controller - no conflicts.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, deque

import cv2
import numpy as np
from PIL import Image

# For virtual controller (what game sees)
try:
    import vgamepad as vg
    HAS_VGAMEPAD = True
except ImportError:
    HAS_VGAMEPAD = False
    print("Warning: vgamepad not installed. Install with: pip install vgamepad")

# For reading physical controller
try:
    import XInput
    HAS_XINPUT = True
except ImportError:
    HAS_XINPUT = False
    # Fallback to inputs library
    try:
        import inputs
        HAS_INPUTS = True
    except ImportError:
        HAS_INPUTS = False
        print("Warning: Neither XInput nor inputs installed.")
        print("Install with: pip install XInput-Python")

# For screen capture
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    print("Warning: mss not installed. Install with: pip install mss")

import zmq

from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO

import argparse
parser = argparse.ArgumentParser(description="DAgger Collection with Auto-Handoff")
parser.add_argument("--process", type=str, default="Hades.exe", help="Game process name")
parser.add_argument("--port", type=int, default=5555, help="AI server port")
parser.add_argument("--host", type=str, default="localhost", help="AI server host")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
parser.add_argument("--deadzone", type=float, default=0.15, help="Joystick deadzone")
parser.add_argument("--idle-timeout", type=float, default=0.5, help="Seconds before AI takes over")
parser.add_argument("--fps", type=int, default=30, help="Target FPS")
args = parser.parse_args()

# Setup output directory
if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    game_name = args.process.replace(".exe", "").lower()
    OUTPUT_DIR = PATH_REPO / "games" / game_name / "dagger_sessions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create session directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = OUTPUT_DIR / f"dagger_{timestamp}"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = SESSION_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


class AutoHandoff:
    """Detects when human is actively controlling vs idle."""

    def __init__(self, deadzone: float = 0.15, idle_timeout: float = 0.5):
        self.deadzone = deadzone
        self.idle_timeout = idle_timeout
        self.last_human_input = 0
        self.human_active = False

    def check(self, controller_state: dict) -> bool:
        """Check if human is providing input."""
        # Check buttons
        buttons = controller_state.get('buttons', {})
        any_button = any(buttons.values()) if isinstance(buttons, dict) else any(buttons)

        # Check joysticks
        joysticks = controller_state.get('joysticks', {})
        if isinstance(joysticks, dict):
            stick_moved = (
                abs(joysticks.get('left_x', 0)) > self.deadzone or
                abs(joysticks.get('left_y', 0)) > self.deadzone or
                abs(joysticks.get('right_x', 0)) > self.deadzone or
                abs(joysticks.get('right_y', 0)) > self.deadzone
            )
        else:
            stick_moved = False

        # Check triggers
        triggers = controller_state.get('triggers', {})
        if isinstance(triggers, dict):
            trigger_pressed = (
                triggers.get('left', 0) > 0.1 or
                triggers.get('right', 0) > 0.1
            )
        else:
            trigger_pressed = False

        if any_button or stick_moved or trigger_pressed:
            self.last_human_input = time.time()
            self.human_active = True
        elif time.time() - self.last_human_input > self.idle_timeout:
            self.human_active = False

        return self.human_active


class PhysicalControllerReader:
    """Reads physical Xbox controller via XInput or inputs library."""

    BUTTON_NAMES = [
        'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT',
        'START', 'BACK', 'LEFT_THUMB', 'RIGHT_THUMB',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        'SOUTH', 'EAST', 'WEST', 'NORTH'
    ]

    def __init__(self, controller_id: int = 0):
        self.controller_id = controller_id
        self.use_xinput = HAS_XINPUT

    def read(self) -> dict:
        """Read current controller state."""
        if self.use_xinput:
            return self._read_xinput()
        elif HAS_INPUTS:
            return self._read_inputs()
        else:
            return self._empty_state()

    def _read_xinput(self) -> dict:
        """Read via XInput library."""
        try:
            state = XInput.get_state(self.controller_id)
        except Exception:
            return self._empty_state()

        gamepad = state.Gamepad

        buttons = {
            'DPAD_UP': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_DPAD_UP),
            'DPAD_DOWN': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_DPAD_DOWN),
            'DPAD_LEFT': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_DPAD_LEFT),
            'DPAD_RIGHT': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_DPAD_RIGHT),
            'START': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_START),
            'BACK': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_BACK),
            'LEFT_THUMB': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_LEFT_THUMB),
            'RIGHT_THUMB': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_RIGHT_THUMB),
            'LEFT_SHOULDER': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_LEFT_SHOULDER),
            'RIGHT_SHOULDER': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_RIGHT_SHOULDER),
            'SOUTH': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_A),
            'EAST': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_B),
            'WEST': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_X),
            'NORTH': bool(gamepad.wButtons & XInput.XINPUT_GAMEPAD_Y),
            'GUIDE': False,
        }

        joysticks = {
            'left_x': gamepad.sThumbLX / 32767.0,
            'left_y': gamepad.sThumbLY / 32767.0,
            'right_x': gamepad.sThumbRX / 32767.0,
            'right_y': gamepad.sThumbRY / 32767.0,
        }

        triggers = {
            'left': gamepad.bLeftTrigger / 255.0,
            'right': gamepad.bRightTrigger / 255.0,
        }

        return {'buttons': buttons, 'joysticks': joysticks, 'triggers': triggers}

    def _read_inputs(self) -> dict:
        """Read via inputs library (fallback)."""
        state = self._empty_state()
        try:
            events = inputs.get_gamepad()
            for event in events:
                self._process_inputs_event(event, state)
        except:
            pass
        return state

    def _process_inputs_event(self, event, state):
        """Process event from inputs library."""
        code = event.code
        value = event.state

        button_map = {
            'BTN_SOUTH': 'SOUTH', 'BTN_EAST': 'EAST',
            'BTN_WEST': 'WEST', 'BTN_NORTH': 'NORTH',
            'BTN_TL': 'LEFT_SHOULDER', 'BTN_TR': 'RIGHT_SHOULDER',
            'BTN_SELECT': 'BACK', 'BTN_START': 'START',
            'BTN_THUMBL': 'LEFT_THUMB', 'BTN_THUMBR': 'RIGHT_THUMB',
        }

        if code in button_map:
            state['buttons'][button_map[code]] = bool(value)
        elif code == 'ABS_X':
            state['joysticks']['left_x'] = value / 32767.0
        elif code == 'ABS_Y':
            state['joysticks']['left_y'] = value / 32767.0
        elif code == 'ABS_RX':
            state['joysticks']['right_x'] = value / 32767.0
        elif code == 'ABS_RY':
            state['joysticks']['right_y'] = value / 32767.0
        elif code == 'ABS_Z':
            state['triggers']['left'] = value / 255.0
        elif code == 'ABS_RZ':
            state['triggers']['right'] = value / 255.0

    def _empty_state(self) -> dict:
        return {
            'buttons': {k: False for k in self.BUTTON_NAMES + ['GUIDE']},
            'joysticks': {'left_x': 0, 'left_y': 0, 'right_x': 0, 'right_y': 0},
            'triggers': {'left': 0, 'right': 0},
        }


class VirtualController:
    """Virtual Xbox controller that the game sees."""

    def __init__(self):
        if not HAS_VGAMEPAD:
            raise RuntimeError("vgamepad not installed. Install with: pip install vgamepad")
        self.gamepad = vg.VX360Gamepad()
        print("Virtual controller created (ViGEmBus)")

    def send(self, action: dict):
        """Send action to virtual controller."""
        self.gamepad.reset()

        # Buttons
        buttons = action.get('buttons', {})
        if buttons.get('SOUTH'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        if buttons.get('EAST'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        if buttons.get('WEST'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        if buttons.get('NORTH'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        if buttons.get('LEFT_SHOULDER'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        if buttons.get('RIGHT_SHOULDER'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        if buttons.get('BACK'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
        if buttons.get('START'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        if buttons.get('LEFT_THUMB'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
        if buttons.get('RIGHT_THUMB'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)
        if buttons.get('DPAD_UP'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        if buttons.get('DPAD_DOWN'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        if buttons.get('DPAD_LEFT'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        if buttons.get('DPAD_RIGHT'): self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)

        # Joysticks
        joysticks = action.get('joysticks', {})
        self.gamepad.left_joystick_float(
            x_value_float=joysticks.get('left_x', 0),
            y_value_float=joysticks.get('left_y', 0)
        )
        self.gamepad.right_joystick_float(
            x_value_float=joysticks.get('right_x', 0),
            y_value_float=joysticks.get('right_y', 0)
        )

        # Triggers
        triggers = action.get('triggers', {})
        self.gamepad.left_trigger_float(triggers.get('left', 0))
        self.gamepad.right_trigger_float(triggers.get('right', 0))

        self.gamepad.update()


class AIPredictor:
    """Gets predictions from the inference server via ZMQ."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

    def predict(self, frame: Image.Image) -> dict:
        """Get AI prediction for frame."""
        try:
            request = {"type": "predict", "image": frame}
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())

            if response.get("status") == "ok":
                return self._convert_prediction(response["pred"])
            return None
        except zmq.Again:
            return None
        except Exception as e:
            print(f"AI error: {e}")
            return None

    def _convert_prediction(self, pred: dict) -> dict:
        """Convert AI prediction to controller format."""
        buttons_array = pred.get("buttons", [])
        j_left = pred.get("j_left", [0, 0])
        j_right = pred.get("j_right", [0, 0])

        # Button mapping
        button_names = BUTTON_ACTION_TOKENS

        buttons = {}
        for i, name in enumerate(button_names):
            if i < len(buttons_array):
                buttons[name] = buttons_array[i] > 0.5
            else:
                buttons[name] = False

        triggers = {
            'left': buttons_array[6] if len(buttons_array) > 6 else 0,
            'right': buttons_array[9] if len(buttons_array) > 9 else 0,
        }

        return {
            'buttons': buttons,
            'joysticks': {
                'left_x': j_left[0] if len(j_left) > 0 else 0,
                'left_y': j_left[1] if len(j_left) > 1 else 0,
                'right_x': j_right[0] if len(j_right) > 0 else 0,
                'right_y': j_right[1] if len(j_right) > 1 else 0,
            },
            'triggers': triggers,
        }

    def reset(self):
        """Reset AI session."""
        try:
            request = {"type": "reset"}
            self.socket.send(pickle.dumps(request))
            self.socket.recv()
        except:
            pass

    def close(self):
        self.socket.close()
        self.context.term()


class ScreenCapture:
    """Captures game screen."""

    def __init__(self, monitor_id: int = 1):
        if not HAS_MSS:
            raise RuntimeError("mss not installed. Install with: pip install mss")
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_id]

    def capture(self) -> Image.Image:
        """Capture and return PIL Image."""
        screenshot = self.sct.grab(self.monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def action_to_record_format(action: dict) -> dict:
    """Convert controller state to samples.json format."""
    buttons = action.get('buttons', {})
    joysticks = action.get('joysticks', {})
    triggers = action.get('triggers', {})

    return {
        'DPAD_UP': int(buttons.get('DPAD_UP', False)),
        'DPAD_DOWN': int(buttons.get('DPAD_DOWN', False)),
        'DPAD_LEFT': int(buttons.get('DPAD_LEFT', False)),
        'DPAD_RIGHT': int(buttons.get('DPAD_RIGHT', False)),
        'START': int(buttons.get('START', False)),
        'BACK': int(buttons.get('BACK', False)),
        'LEFT_THUMB': int(buttons.get('LEFT_THUMB', False)),
        'RIGHT_THUMB': int(buttons.get('RIGHT_THUMB', False)),
        'LEFT_SHOULDER': int(buttons.get('LEFT_SHOULDER', False)),
        'RIGHT_SHOULDER': int(buttons.get('RIGHT_SHOULDER', False)),
        'SOUTH': int(buttons.get('SOUTH', False)),
        'EAST': int(buttons.get('EAST', False)),
        'WEST': int(buttons.get('WEST', False)),
        'NORTH': int(buttons.get('NORTH', False)),
        'GUIDE': 0,
        'AXIS_LEFTX': [int(joysticks.get('left_x', 0) * 32767)],
        'AXIS_LEFTY': [int(joysticks.get('left_y', 0) * 32767)],
        'AXIS_RIGHTX': [int(joysticks.get('right_x', 0) * 32767)],
        'AXIS_RIGHTY': [int(joysticks.get('right_y', 0) * 32767)],
        'LEFT_TRIGGER': [int(triggers.get('left', 0) * 255)],
        'RIGHT_TRIGGER': [int(triggers.get('right', 0) * 255)],
    }


def main():
    print("=" * 60)
    print("DAgger Collection with Auto-Handoff")
    print("=" * 60)
    print(f"  Game: {args.process}")
    print(f"  AI Server: {args.host}:{args.port}")
    print(f"  Output: {SESSION_DIR}")
    print(f"  Deadzone: {args.deadzone}")
    print(f"  Idle timeout: {args.idle_timeout}s")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Grab controller = You control (H indicator)")
    print("  Release for 0.5s = AI resumes (. indicator)")
    print("  Ctrl+C = End session and save")
    print()

    # Check dependencies
    missing = []
    if not HAS_VGAMEPAD:
        missing.append("vgamepad")
    if not HAS_XINPUT and not HAS_INPUTS:
        missing.append("XInput-Python (or inputs)")
    if not HAS_MSS:
        missing.append("mss")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        print("\nNote: vgamepad requires ViGEmBus driver.")
        print("Download: https://github.com/ViGEm/ViGEmBus/releases")
        sys.exit(1)

    # Initialize components
    print("Initializing...")
    handoff = AutoHandoff(args.deadzone, args.idle_timeout)
    physical_controller = PhysicalControllerReader()
    virtual_controller = VirtualController()
    ai = AIPredictor(args.host, args.port)
    screen = ScreenCapture()

    # Session data
    session_data = []
    frame_count = 0
    human_frames = 0
    ai_frames = 0

    frame_time = 1.0 / args.fps
    last_status_time = time.time()

    print("\nMake sure the game is in focus!")
    print("Starting in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("GO!\n")

    try:
        while True:
            loop_start = time.time()

            # Capture screen
            frame = screen.capture()
            frame_resized = frame.resize((256, 256), Image.LANCZOS)

            # Read physical controller
            physical_state = physical_controller.read()
            human_active = handoff.check(physical_state)

            if human_active:
                # Human controls
                action = physical_state
                source = "human"
                human_frames += 1
                indicator = "H"
            else:
                # AI controls
                ai_action = ai.predict(frame_resized)
                if ai_action is not None:
                    action = ai_action
                    source = "ai"
                    ai_frames += 1
                    indicator = "."
                else:
                    action = physical_controller._empty_state()
                    source = "ai_fallback"
                    ai_frames += 1
                    indicator = "?"

            # Send to game via virtual controller
            virtual_controller.send(action)

            # Save frame
            frame_path = FRAMES_DIR / f"{frame_count:06d}.png"
            frame_resized.save(frame_path)

            # Record sample
            session_data.append({
                'frame_id': frame_count,
                'frame_path': str(frame_path),
                'action': action_to_record_format(action),
                'source': source,
                'timestamp': time.time(),
                'is_human': source == "human",
            })

            frame_count += 1
            print(indicator, end="", flush=True)

            # Status every 5 seconds
            if time.time() - last_status_time > 5:
                total = human_frames + ai_frames
                human_pct = (human_frames / total * 100) if total > 0 else 0
                print(f" [{frame_count} frames, {human_pct:.0f}% human]")
                last_status_time = time.time()

            # Maintain FPS
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopping collection...")

    # Save session
    print("\nSaving session...")

    samples_path = SESSION_DIR / "samples.json"
    with open(samples_path, 'w') as f:
        json.dump(session_data, f, indent=2)

    metadata = {
        'total_frames': frame_count,
        'human_frames': human_frames,
        'ai_frames': ai_frames,
        'human_percentage': (human_frames / frame_count * 100) if frame_count > 0 else 0,
        'game': args.process,
        'deadzone': args.deadzone,
        'idle_timeout': args.idle_timeout,
    }

    metadata_path = SESSION_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("DAgger Collection Complete!")
    print(f"{'='*60}")
    print(f"  Session: {SESSION_DIR}")
    print(f"  Total frames: {frame_count}")
    print(f"  Human frames: {human_frames} ({metadata['human_percentage']:.1f}%)")
    print(f"  AI frames: {ai_frames}")
    print(f"{'='*60}")
    print(f"\nTo train on this data:")
    print(f"  python scripts/dagger_train_weighted.py --corrections {SESSION_DIR}")

    ai.close()


if __name__ == "__main__":
    main()
