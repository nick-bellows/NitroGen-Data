"""
NitroGen Inference Benchmark

Measures actual inference performance with various settings.

Usage:
    python scripts/benchmark.py

This connects to a running server and measures:
- Inference latency (min/max/avg)
- Effective FPS
- Throughput with action chunking
"""

import sys
import time
import argparse
import statistics

import numpy as np
from PIL import Image

from nitrogen.inference_client import ModelClient

parser = argparse.ArgumentParser(description="NitroGen Benchmark")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument("--iterations", type=int, default=100, help="Number of test iterations")
parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
args = parser.parse_args()

print("=" * 60)
print("NitroGen Inference Benchmark")
print("=" * 60)
print()

# Connect to server
print(f"Connecting to server on port {args.port}...")
try:
    client = ModelClient(port=args.port)
    client.reset()
    info = client.info()
    print("Connected!")
    print(f"Server info: {info.get('optimizations', 'N/A')}")
except Exception as e:
    print(f"ERROR: Could not connect: {e}")
    print("Make sure the server is running!")
    sys.exit(1)

# Create test image (random noise simulating game frame)
print("\nCreating test image...")
test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

# Warmup
print(f"\nWarming up ({args.warmup} iterations)...")
for i in range(args.warmup):
    _ = client.predict(test_img)
    print(f"  Warmup {i+1}/{args.warmup}", end="\r")
print()

# Benchmark
print(f"\nRunning benchmark ({args.iterations} iterations)...")
times = []

for i in range(args.iterations):
    start = time.perf_counter()
    result = client.predict(test_img)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

    # Progress
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{args.iterations} ({elapsed*1000:.1f}ms)", end="\r")

print()

# Calculate statistics
times_ms = [t * 1000 for t in times]
avg_ms = statistics.mean(times_ms)
min_ms = min(times_ms)
max_ms = max(times_ms)
std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
p50_ms = statistics.median(times_ms)
p95_ms = sorted(times_ms)[int(len(times_ms) * 0.95)]
p99_ms = sorted(times_ms)[int(len(times_ms) * 0.99)]

# Calculate FPS metrics
actions_per_chunk = len(result.get("buttons", [])) if result else 16
raw_fps = 1000 / avg_ms
effective_fps_1 = raw_fps  # Using 1 action per inference
effective_fps_4 = raw_fps * 4  # Using 4 actions per inference
effective_fps_8 = raw_fps * 8  # Using 8 actions per inference

# Results
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print()
print("Latency (ms):")
print(f"  Average:  {avg_ms:6.1f} ms")
print(f"  Min:      {min_ms:6.1f} ms")
print(f"  Max:      {max_ms:6.1f} ms")
print(f"  Std Dev:  {std_ms:6.1f} ms")
print(f"  P50:      {p50_ms:6.1f} ms")
print(f"  P95:      {p95_ms:6.1f} ms")
print(f"  P99:      {p99_ms:6.1f} ms")
print()
print("Throughput:")
print(f"  Raw inference FPS:     {raw_fps:5.1f}")
print(f"  Actions per chunk:     {actions_per_chunk}")
print()
print("Effective FPS (with action chunking):")
print(f"  1 action/inference:    {effective_fps_1:5.1f} FPS")
print(f"  4 actions/inference:   {effective_fps_4:5.1f} FPS  <-- Recommended")
print(f"  8 actions/inference:   {effective_fps_8:5.1f} FPS")
print()

# Performance rating
if effective_fps_4 >= 60:
    rating = "EXCELLENT - Real-time gameplay possible"
elif effective_fps_4 >= 30:
    rating = "GOOD - Playable performance"
elif effective_fps_4 >= 15:
    rating = "MODERATE - Slow but functional"
else:
    rating = "POOR - Consider more optimizations"

print(f"Performance Rating: {rating}")
print()
print("=" * 60)

# Recommendations
print("\nRecommendations:")
if avg_ms > 100:
    print("  - Try --timesteps 2 for faster inference")
    print("  - Enable --fp16 if not already")
    print("  - Use --actions-per-chunk 8 in play_fast.py")
elif avg_ms > 50:
    print("  - Current settings are good")
    print("  - Use --actions-per-chunk 4 for best balance")
else:
    print("  - Excellent performance!")
    print("  - Can use --actions-per-chunk 2 for more responsiveness")
