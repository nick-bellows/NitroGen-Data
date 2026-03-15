# NitroGen Debug Tools

Debugging and testing utilities for NitroGen.

## Scripts

| Script | Description |
|--------|-------------|
| `test_controller.bat` | Test if virtual Xbox controller works |
| `debug_actions.bat` | See what actions the model is predicting |
| `benchmark.bat` | Measure inference performance |

## Usage

### Test Controller
Tests virtual controller without needing the AI model.
1. **Unplug your real controller**
2. Start your game
3. Run `test_controller.bat`
4. Watch if character moves in-game

### Debug Actions
Captures frames and shows model predictions.
1. Start inference server
2. Start your game
3. Run `debug_actions.bat`
4. Check saved `debug_frame_XX.png` files and console output

### Benchmark
Measures raw inference speed.
1. Start inference server
2. Run `benchmark.bat`
3. Review FPS and latency numbers

## Troubleshooting

### Controller not working
1. Install ViGEmBus driver
2. Unplug real controller
3. Check Windows Settings > Devices > Controllers
4. In-game, set controls to "Gamepad"

### Wrong screen captured
- Make sure game window is in foreground
- Use Borderless Windowed mode
- Check saved debug frames

### Low FPS
- Use `start_server_optimized.bat` or `start_server_ultra.bat`
- Close other GPU-heavy applications
- Check VRAM usage
