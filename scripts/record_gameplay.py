"""
High-FPS Gameplay Recorder for NitroGen Training

Records screen + controller inputs without AI inference.
Pure Behavior Cloning data collection at full FPS.

Usage:
    python scripts/record_gameplay.py --process "Hades.exe" --fps 30

Controls:
    F5: Start/Stop recording
    F6: Quit

Output is compatible with dagger_train.py
"""

import argparse
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np

# Try DXCam first (faster), fall back to mss
try:
    import dxcam
    USE_DXCAM = True
except ImportError:
    try:
        from mss import mss
        USE_DXCAM = False
    except ImportError:
        print("ERROR: Need either dxcam or mss for screen capture")
        print("Install: pip install dxcam  OR  pip install mss")
        exit(1)

try:
    import inputs
    HAS_INPUTS = True
except ImportError:
    HAS_INPUTS = False
    print("WARNING: 'inputs' package not installed. Controller recording disabled.")
    print("Install: pip install inputs")

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("WARNING: 'keyboard' package not installed. Using Ctrl+C to stop.")

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False
    print("WARNING: 'pygetwindow' not installed. Will capture primary monitor.")


class ControllerReader:
    """Read Xbox controller state in background thread"""

    # Map inputs library codes to NitroGen action keys
    BUTTON_MAP = {
        'BTN_SOUTH': 'SOUTH',      # A
        'BTN_EAST': 'EAST',        # B
        'BTN_WEST': 'WEST',        # X
        'BTN_NORTH': 'NORTH',      # Y
        'BTN_TL': 'LEFT_SHOULDER', # LB
        'BTN_TR': 'RIGHT_SHOULDER',# RB
        'BTN_THUMBL': 'LEFT_THUMB',# L3
        'BTN_THUMBR': 'RIGHT_THUMB',# R3
        'BTN_START': 'START',
        'BTN_SELECT': 'BACK',
        'BTN_MODE': 'GUIDE',       # Xbox button
    }

    def __init__(self):
        # Initialize state matching NitroGen format
        self.state = OrderedDict([
            ("WEST", 0), ("SOUTH", 0), ("BACK", 0),
            ("DPAD_DOWN", 0), ("DPAD_LEFT", 0), ("DPAD_RIGHT", 0), ("DPAD_UP", 0),
            ("GUIDE", 0),
            ("AXIS_LEFTX", [0]),
            ("AXIS_LEFTY", [0]),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", [0]),
            ("AXIS_RIGHTX", [0]),
            ("AXIS_RIGHTY", [0]),
            ("LEFT_THUMB", 0), ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", [0]),
            ("START", 0), ("EAST", 0), ("NORTH", 0),
        ])

        self.running = True
        self.lock = threading.Lock()

        if HAS_INPUTS:
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()

    def _read_loop(self):
        while self.running:
            try:
                events = inputs.get_gamepad()
                for event in events:
                    self._process_event(event)
            except inputs.UnpluggedError:
                time.sleep(0.1)
            except Exception:
                time.sleep(0.01)

    def _process_event(self, event):
        code = event.code
        value = event.state

        with self.lock:
            # Buttons
            if code in self.BUTTON_MAP:
                self.state[self.BUTTON_MAP[code]] = 1 if value else 0

            # D-Pad (special handling)
            elif code == 'ABS_HAT0X':
                self.state['DPAD_LEFT'] = 1 if value < 0 else 0
                self.state['DPAD_RIGHT'] = 1 if value > 0 else 0
            elif code == 'ABS_HAT0Y':
                self.state['DPAD_UP'] = 1 if value < 0 else 0
                self.state['DPAD_DOWN'] = 1 if value > 0 else 0

            # Triggers (0-255 range)
            elif code == 'ABS_Z':
                self.state['LEFT_TRIGGER'] = [value]
            elif code == 'ABS_RZ':
                self.state['RIGHT_TRIGGER'] = [value]

            # Joysticks (-32768 to 32767)
            elif code == 'ABS_X':
                self.state['AXIS_LEFTX'] = [value]
            elif code == 'ABS_Y':
                self.state['AXIS_LEFTY'] = [value]
            elif code == 'ABS_RX':
                self.state['AXIS_RIGHTX'] = [value]
            elif code == 'ABS_RY':
                self.state['AXIS_RIGHTY'] = [value]

    def get_state(self):
        """Get current controller state (thread-safe copy)"""
        with self.lock:
            # Deep copy for lists
            state_copy = {}
            for k, v in self.state.items():
                if isinstance(v, list):
                    state_copy[k] = v.copy()
                else:
                    state_copy[k] = v
            return state_copy

    def stop(self):
        self.running = False


class AsyncFrameSaver:
    """Save frames in background thread to prevent stuttering"""

    def __init__(self, output_dir, max_queue=300):
        self.frames_dir = Path(output_dir) / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.queue = queue.Queue(maxsize=max_queue)
        self.running = True
        self.saved_count = 0
        self.dropped_count = 0

        # Multiple worker threads for faster saving
        self.workers = []
        for _ in range(2):
            t = threading.Thread(target=self._save_loop, daemon=True)
            t.start()
            self.workers.append(t)

    def _save_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame_idx, frame = self.queue.get(timeout=0.1)
                filename = self.frames_dir / f"{frame_idx:06d}.png"
                # Use lower compression for speed
                cv2.imwrite(str(filename), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                self.saved_count += 1
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError saving frame: {e}")

    def save(self, frame_idx, frame):
        """Queue frame for async saving. Returns False if queue full."""
        try:
            self.queue.put_nowait((frame_idx, frame))
            return True
        except queue.Full:
            self.dropped_count += 1
            return False

    def get_queue_size(self):
        return self.queue.qsize()

    def stop(self):
        self.running = False
        # Wait for queue to drain
        try:
            self.queue.join()
        except:
            pass
        for t in self.workers:
            t.join(timeout=2)


def get_window_region(process_name):
    """Get game window position and size"""
    if not HAS_PYGETWINDOW:
        return None

    # Try exact match first
    title_search = process_name.replace('.exe', '').replace('.EXE', '')

    try:
        # Get all windows
        all_windows = gw.getAllWindows()

        for win in all_windows:
            if not win.title:
                continue
            # Case-insensitive partial match
            if title_search.lower() in win.title.lower():
                if win.width > 100 and win.height > 100:  # Skip tiny windows
                    return (win.left, win.top, win.width, win.height), win

        return None, None
    except Exception as e:
        print(f"Error finding window: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Record gameplay for NitroGen training')
    parser.add_argument('--process', default='Hades.exe', help='Game process name')
    parser.add_argument('--fps', type=int, default=30, help='Target capture FPS (default: 30)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: recordings/)')
    parser.add_argument('--resolution', type=int, default=256, help='Output resolution (default: 256)')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames to record (0=unlimited)')
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    game_name = args.process.replace('.exe', '').replace('.EXE', '')

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path("recordings")

    output_dir = base_dir / f"{game_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  NitroGen Gameplay Recorder")
    print("=" * 60)
    print(f"  Game: {args.process}")
    print(f"  Output: {output_dir}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Screen capture: {'DXCam' if USE_DXCAM else 'MSS'}")
    print(f"  Controller: {'Enabled' if HAS_INPUTS else 'DISABLED'}")
    print("=" * 60)

    if HAS_KEYBOARD:
        print("\n  Controls:")
        print("    F5 = Start/Stop recording")
        print("    F6 = Quit")
    else:
        print("\n  Press Ctrl+C to stop recording")

    print()

    # Initialize components
    controller = ControllerReader()
    frame_saver = AsyncFrameSaver(output_dir)

    # Setup screen capture
    camera = None
    sct = None

    if USE_DXCAM:
        camera = dxcam.create(output_color="BGR")
    else:
        sct = mss()

    # Find game window
    print("  Looking for game window...")
    region = None
    game_window = None

    for attempt in range(30):  # Try for 30 seconds
        result = get_window_region(args.process)
        if result and result[0]:
            region, game_window = result
            break
        time.sleep(1)
        print(f"  Waiting for {args.process}... ({attempt+1}s)")

    if region is None:
        print(f"\n  Could not find window for {args.process}")
        print("  Using full primary monitor capture instead.")
        if USE_DXCAM:
            # Get monitor dimensions
            import ctypes
            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            region = (0, 0, width, height)
        else:
            region = (0, 0, 1920, 1080)  # Default assumption

    left, top, width, height = region
    print(f"  Capture region: {width}x{height} at ({left}, {top})")
    print()
    print("=" * 60)
    print("  Ready! Press F5 to start recording." if HAS_KEYBOARD else "  Recording will start in 3 seconds...")
    print("=" * 60)
    print()

    if not HAS_KEYBOARD:
        for i in range(3, 0, -1):
            print(f"  Starting in {i}...")
            time.sleep(1)

    # Recording state
    recording = False if HAS_KEYBOARD else True
    samples = []
    frame_idx = 0
    frame_time = 1.0 / args.fps

    # FPS tracking
    fps_frames = 0
    fps_start = time.time()
    current_fps = 0

    # For DXCam region format
    if USE_DXCAM:
        capture_region = (left, top, left + width, top + height)

    try:
        while True:
            loop_start = time.time()

            # Check hotkeys
            if HAS_KEYBOARD:
                if keyboard.is_pressed('f5'):
                    recording = not recording
                    if recording:
                        print(f"\n  [RECORDING STARTED] Frame {frame_idx}")
                    else:
                        print(f"\n  [RECORDING PAUSED] {frame_idx} frames captured")
                    time.sleep(0.3)  # Debounce

                if keyboard.is_pressed('f6'):
                    print("\n  Stopping...")
                    break

            if recording:
                # Bring game window to front periodically
                if game_window and frame_idx % 60 == 0:
                    try:
                        if not game_window.isActive:
                            game_window.activate()
                    except:
                        pass

                # Capture frame
                if USE_DXCAM:
                    frame = camera.grab(region=capture_region)
                    if frame is None:
                        time.sleep(0.001)
                        continue
                else:
                    monitor = {"left": left, "top": top, "width": width, "height": height}
                    frame = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Resize to NitroGen input size
                frame_resized = cv2.resize(frame, (args.resolution, args.resolution),
                                          interpolation=cv2.INTER_AREA)

                # Get controller state
                action = controller.get_state()

                # Save frame async
                if frame_saver.save(frame_idx, frame_resized):
                    # Store sample in DAgger-compatible format
                    sample = {
                        "frame_id": frame_idx,
                        "frame_path": str(output_dir / "frames" / f"{frame_idx:06d}.png"),
                        "action": action,
                        "timestamp": time.time(),
                        "is_human": True,  # All BC data is human
                    }
                    samples.append(sample)
                    frame_idx += 1

                # Check max frames
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    print(f"\n  Reached max frames ({args.max_frames})")
                    break

                # FPS tracking
                fps_frames += 1
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_frames
                    fps_frames = 0
                    fps_start = time.time()

                # Status update
                queue_size = frame_saver.get_queue_size()
                dropped = frame_saver.dropped_count
                status = f"\r  Recording: {frame_idx:,} frames | {current_fps} FPS | Queue: {queue_size}"
                if dropped > 0:
                    status += f" | Dropped: {dropped}"
                print(status + "    ", end='', flush=True)
            else:
                # Not recording, just idle
                time.sleep(0.1)

            # Maintain target FPS
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n\n  Interrupted!")

    finally:
        print("\n  Finishing up...")

        # Stop components
        controller.stop()

        print("  Saving remaining frames...")
        frame_saver.stop()

        # Save samples.json (DAgger-compatible format)
        print("  Saving samples.json...")
        samples_path = output_dir / "samples.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)

        # Save metadata
        metadata = {
            "game": args.process,
            "fps": args.fps,
            "resolution": args.resolution,
            "total_frames": frame_idx,
            "duration_seconds": frame_idx / args.fps if args.fps > 0 else 0,
            "timestamp": timestamp,
            "capture_method": "dxcam" if USE_DXCAM else "mss",
            "dropped_frames": frame_saver.dropped_count,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print()
        print("=" * 60)
        print("  Recording Complete!")
        print("=" * 60)
        print(f"  Total frames: {frame_idx:,}")
        print(f"  Duration: {frame_idx / args.fps:.1f} seconds")
        print(f"  Dropped frames: {frame_saver.dropped_count}")
        print(f"  Output: {output_dir}")
        print("=" * 60)
        print()
        print("  To train on this data, run:")
        print(f"  python scripts/dagger_train.py --data-dir \"{output_dir}\" --epochs 10")
        print()


if __name__ == "__main__":
    main()
