# Hades AI Improvement Plan

## Current Status
- ✅ Model trained (loss: 0.0004)
- ✅ AI can control the game (~19 FPS)
- ⚠️ Gets stuck in corners - lacks temporal awareness
- ⚠️ No objective understanding

## Implementation Progress
- ✅ **Phase 1: Frame Stacking** - Scripts created, ready for training
  - `scripts/dagger_train_multiframe.py` - Multi-frame training script
  - `scripts/serve_multiframe.py` - Multi-frame inference server
  - `scripts/frame_stacking.py` - Utility module
  - `games/hades/train_multiframe.bat` - Easy training launcher
  - `start_server_multiframe.bat` - Easy server launcher
- ❌ **Phase 2: NVIDIA Data** - SKIPPED (redundant - already in pre-trained weights)
- ✅ **Phase 3: DAgger** - Scripts created, ready for collection
  - `scripts/dagger_collect.py` - Auto-handoff collection
  - `scripts/dagger_train_weighted.py` - Weighted training
  - `games/hades/dagger_collect.bat` - Easy collection launcher
  - `games/hades/dagger_train.bat` - Easy training launcher
  - `start_server_dagger.bat` - Serve DAgger-improved model

## Improvement Phases

---

## Phase 1: Frame Stacking (Temporal Context)

**Goal:** Give the model awareness of motion and direction by seeing multiple frames.

**Current State:**
- Model sees 1 frame at a time
- No awareness of movement direction
- Can't distinguish "moving toward corner" vs "stuck in corner"

**Target State:**
- Model sees 4 consecutive frames
- Can perceive motion, velocity, and direction
- Better spatial reasoning

### Tasks

#### 1.1 ✅ Create Frame-Stacking Training Script
**File:** `scripts/dagger_train_multiframe.py`

Created with:
- `MultiFrameDataset` class that loads N consecutive frames per sample
- Session-aware frame grouping for proper temporal sequences
- Full model input construction with proper dropped_frames handling
- Saves `multiframe_config` in checkpoint for inference

#### 1.2 ✅ Create Multi-Frame Inference Server
**File:** `scripts/serve_multiframe.py`

Created with:
- `MultiFrameServer` class with frame buffer
- Warmup with full buffer fill
- Stats tracking (FPS, buffer size)
- Checkpoint config detection (reads training frame count)

#### 1.3 ✅ Create Frame Stacking Utilities
**File:** `scripts/frame_stacking.py`

Created with:
- `FrameBuffer` class for rolling frame history
- `channel_stack_frames()` for channel-based stacking
- `prepare_multiframe_input()` for model input preparation
- Helper functions for path handling and analysis

#### 1.4 ⏳ Retrain Model (Ready to Run)
```bash
# Option 1: Use batch file
games\hades\train_multiframe.bat

# Option 2: Command line
python scripts/dagger_train_multiframe.py \
    --data-dir games/hades/recordings/Hades_20260110_185339 \
    --num-frames 4 \
    --epochs 10
```

### Expected Improvement
- Model can see "I've been in this corner for 4 frames" → dodge out
- Better prediction of enemy movement trajectories
- Smoother action sequences (less jitter)

---

## Phase 2: Augment with NVIDIA Data (353 Hours of Hades)

**Goal:** Expose model to diverse gameplay situations from expert players.

**Current State:**
- Trained on ~7.8 hours of your recordings (14,037 samples)
- Limited gameplay variety

**Target State:**
- Training on 350+ hours of diverse Hades gameplay
- Exposure to all weapons, bosses, rooms, builds
- Better generalization

### Tasks

#### 2.1 Extract Hades Data from Downloaded Shards
**File:** `scripts/nvidia_index/extract_hades_data.py`

We already have all 100 shards downloaded. Need to:
- Filter for Hades metadata files (already identified: 97 shards)
- Extract corresponding parquet files with action labels
- Convert to DAgger training format

#### 2.2 Download Hades Videos
Since we have metadata with YouTube URLs, we can:
- Use yt-dlp to download the actual gameplay videos
- Extract frames at 30 FPS
- Match frames to action labels from parquet files

#### 2.3 Create Combined Dataset
**File:** `scripts/nvidia_merge_datasets.py` (already exists, needs update)

- Merge NVIDIA Hades data with your recordings
- Apply frame stacking to NVIDIA data too
- Create unified `samples.json`

#### 2.4 Train on Combined Data
```bash
python scripts/dagger_train_multiframe.py \
    --data-dir combined_training_data \
    --num-frames 4 \
    --epochs 15 \
    --batch-size 8
```

### Expected Improvement
- Knows how to handle all enemy types
- Better room navigation (has seen thousands of rooms)
- Weapon-specific combat patterns
- Boss fight strategies

---

## Phase 3: DAgger (Interactive Correction)

**Goal:** Fix specific failure modes by correcting AI mistakes in real-time.

**Current State:**
- AI trained on demonstrations only
- When AI makes mistakes, it doesn't know how to recover
- Distribution shift: AI's mistakes create states not seen in training

**Target State:**
- AI can recover from its own mistakes
- Corrections teach recovery strategies
- Iterative improvement loop

### Tasks

#### 3.1 ✅ Create DAgger Collection Script
**File:** `scripts/dagger_collect.py`

Created with:
- Auto-handoff detection (grab controller = you control)
- Virtual controller via ViGEmBus (no input conflicts)
- XInput reader for physical controller
- Screen capture with mss
- Records both AI and human frames with source tags

Controller Flow:
```
Physical Controller → Script (XInput) → Decision → Virtual Controller → Game
```

#### 3.2 ✅ Create Weighted Training Script
**File:** `scripts/dagger_train_weighted.py`

Created with:
- WeightedRandomSampler for correction priority
- Supports multiple DAgger sessions
- Human-only mode option
- Combined with frame stacking (4 frames)

#### 3.3 ✅ Create Batch Files
- `games/hades/dagger_collect.bat` - Start correction collection
- `games/hades/dagger_train.bat` - Train on corrections
- `start_server_dagger.bat` - Serve DAgger-improved model

#### 3.4 ⏳ DAgger Iteration Loop (Ready to Run)
1. Run `start_server_multiframe.bat` (AI server)
2. Run `games\hades\dagger_collect.bat` (collect corrections)
3. Play for 10-15 minutes, correcting mistakes
4. Run `games\hades\dagger_train.bat` (train)
5. Repeat with improved model

### Expected Improvement
- Learns to escape corners (you show it how)
- Better boss fight reactions (you correct bad dodges)
- Recovers from mistakes instead of getting stuck

---

## Implementation Order

### Week 1: Frame Stacking
```
Day 1-2: Create dagger_train_multiframe.py
Day 3:   Update serve_optimized.py for multi-frame inference
Day 4:   Retrain model with 4-frame stacking
Day 5:   Test and tune frame count (try 4, 8 frames)
```

### Week 2: NVIDIA Data Integration
```
Day 1:   Create extract_hades_data.py
Day 2-3: Download Hades videos (may take time)
Day 4:   Convert to training format
Day 5:   Train on combined dataset
```

### Week 3: DAgger
```
Day 1-2: Create dagger_collect.py with human override
Day 3:   First DAgger session (30 min corrections)
Day 4:   Retrain with corrections
Day 5:   Iterate: play → correct → train
```

---

## File Structure After Implementation

```
NitroGen/
├── scripts/
│   ├── dagger_train.py              # Original (1 frame)
│   ├── dagger_train_multiframe.py   # ✅ Multi-frame training
│   ├── dagger_train_weighted.py     # ✅ DAgger weighted training
│   ├── dagger_collect.py            # ✅ DAgger collection with auto-handoff
│   ├── serve_multiframe.py          # ✅ Multi-frame inference server
│   ├── frame_stacking.py            # ✅ Frame stacking utilities
│   ├── serve_optimized.py           # Original optimized server
│   └── nvidia_index/
│       └── ...                      # Index building tools
├── games/hades/
│   ├── recordings/                  # Your recordings
│   ├── dagger_sessions/             # DAgger correction sessions
│   ├── train_multiframe.bat         # ✅ Frame stacking training
│   ├── dagger_collect.bat           # ✅ DAgger collection
│   └── dagger_train.bat             # ✅ DAgger training
├── start_server_multiframe.bat      # ✅ Serve multi-frame model
├── start_server_dagger.bat          # ✅ Serve DAgger-improved model
└── checkpoints/
    ├── nitrogen_hades_best.pt       # Current best (1 frame)
    ├── nitrogen_hades_multiframe_best.pt  # After frame stacking training
    └── nitrogen_hades_dagger_best.pt     # After DAgger training
```

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | Corner escape rate | >80% (vs current ~20%) |
| Phase 1 | Action smoothness | Less jitter in movement |
| Phase 2 | Room completion | Consistently clear rooms |
| Phase 2 | Boss damage dealt | Measurable damage before death |
| Phase 3 | Recovery from mistakes | <5 seconds to recover |
| Phase 3 | Play time before stuck | >5 minutes continuous |

---

## Commands Quick Reference

### Phase 1: Frame Stacking
```bash
# Train with frame stacking (use batch file - recommended)
games\hades\train_multiframe.bat

# Or manually:
python scripts/dagger_train_multiframe.py --data-dir games/hades/recordings/Hades_20260110_185339 --num-frames 4 --epochs 10

# Serve with frame stacking (use batch file - recommended)
start_server_multiframe.bat

# Or manually:
python scripts/serve_multiframe.py --ckpt checkpoints/nitrogen_hades_multiframe_best.pt --ctx 4 --timesteps 4 --fp16

# Play
python scripts/play_fast.py --host localhost --port 8000
```

### Phase 2: NVIDIA Data
```bash
# Extract Hades data from shards
python scripts/nvidia_index/extract_hades_data.py --game hades --output games/hades/nvidia_data

# Merge datasets
python scripts/nvidia_merge_datasets.py --user-dir games/hades/recordings/Hades_20260110_185339 --nvidia-dir games/hades/nvidia_data --output games/hades/combined_data

# Train on combined
python scripts/dagger_train_multiframe.py --data-dir games/hades/combined_data --num-frames 4 --epochs 15
```

### Phase 3: DAgger
```bash
# Step 1: Start the AI server
start_server_multiframe.bat

# Step 2: Collect corrections (grab controller to correct, release to let AI play)
games\hades\dagger_collect.bat

# Step 3: Train on corrections
games\hades\dagger_train.bat

# Or manually:
python scripts/dagger_collect.py --process "Hades.exe" --port 5555

python scripts/dagger_train_weighted.py \
    --base-data games/hades/recordings/Hades_20260110_185339 \
    --corrections games/hades/dagger_sessions \
    --correction-weight 2.0 \
    --epochs 5

# Serve the improved model
start_server_dagger.bat
```

---

## Ready to Train?

All scripts are complete! Here's the recommended workflow:

### Step 1: Frame Stacking Training
```bash
games\hades\train_multiframe.bat
```
This trains with 4-frame temporal context on your existing recordings.

### Step 2: Test the Multi-Frame Model
```bash
start_server_multiframe.bat
# In another terminal:
python scripts/play_fast.py --host localhost --port 8000
```

### Step 3: DAgger - Correct AI Mistakes
```bash
# Terminal 1: AI server
start_server_multiframe.bat

# Terminal 2: Play and correct
games\hades\dagger_collect.bat
```
Grab controller when AI makes mistakes. Release to let AI resume.

### Step 4: Train on Corrections
```bash
games\hades\dagger_train.bat
```

### Step 5: Test the Improved Model
```bash
start_server_dagger.bat
```

Repeat Steps 3-5 to iteratively improve the model!
