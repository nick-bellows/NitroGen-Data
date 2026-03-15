"""
Test virtual controller connection to Dark Souls III.

This script will:
1. Create a virtual Xbox controller
2. Move the left joystick in a circle
3. Press some buttons

If you see the character moving in DS3, the controller is working.
"""

import time
import math
import vgamepad as vg

print("=" * 60)
print("Virtual Controller Test")
print("=" * 60)
print()
print("IMPORTANT: Unplug your REAL controller before running this test!")
print("DS3 only listens to one controller at a time.")
print()
print("This will test if the virtual controller works with DS3.")
print("Make sure Dark Souls III is running and you're in-game.")
print()
print("The test will:")
print("  1. Move left joystick in a circle (character should walk)")
print("  2. Press A button (roll/interact)")
print("  3. Move right joystick (camera should rotate)")
print()
input("Press Enter to start the test (UNPLUG REAL CONTROLLER FIRST)...")

# Create virtual Xbox controller
print("\nCreating virtual Xbox 360 controller...")
gamepad = vg.VX360Gamepad()

# Wake up the controller
print("Waking up controller...")
gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
gamepad.update()
time.sleep(0.2)
gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
gamepad.update()
time.sleep(0.5)

print("\n[TEST 1] Moving left joystick in a circle for 5 seconds...")
print("         (Character should walk in a circle)")
start = time.time()
while time.time() - start < 5:
    t = time.time() - start
    # Circle motion
    x = int(math.cos(t * 2) * 32767)
    y = int(math.sin(t * 2) * 32767)
    gamepad.left_joystick(x_value=x, y_value=y)
    gamepad.update()
    time.sleep(0.016)

# Reset
gamepad.reset()
gamepad.update()
time.sleep(0.5)

print("\n[TEST 2] Pressing A button 3 times...")
print("         (Character should roll or interact)")
for i in range(3):
    gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()
    time.sleep(0.15)
    gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()
    time.sleep(0.3)

time.sleep(0.5)

print("\n[TEST 3] Moving right joystick for 5 seconds...")
print("         (Camera should rotate)")
start = time.time()
while time.time() - start < 5:
    t = time.time() - start
    # Rotate camera
    x = int(math.cos(t) * 20000)
    y = int(math.sin(t * 0.5) * 10000)
    gamepad.right_joystick(x_value=x, y_value=y)
    gamepad.update()
    time.sleep(0.016)

# Reset
gamepad.reset()
gamepad.update()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
print()
print("Did the character move and the camera rotate?")
print()
print("If YES: The virtual controller is working. The issue is elsewhere.")
print("If NO:  Dark Souls III is not detecting the virtual controller.")
print()
print("Troubleshooting if it didn't work:")
print("  1. Make sure ViGEmBus is installed (run ViGEmBus_Setup.exe)")
print("  2. In DS3, go to Options > Key Config and set to 'Gamepad'")
print("  3. Try restarting Dark Souls III after running this script")
print("  4. Check Windows Settings > Devices > Controllers")
print()
