# NitroGen: An Open Foundation Model for Generalist Gaming Agents

**Date:** 2025-12-19

**Authors:** Loïc Magne¹*, Anas Awadalla¹²*, Guanzhi Wang¹³*†, Yinzhen Xu¹, Joshua Belofsky⁴, Fengyuan Hu¹, Joohwan Kim¹, Ludwig Schmidt², Georgia Gkioxari³, Jan Kautz¹, Yisong Yue³†, Yejin Choi¹²†, Yuke Zhu¹⁵†, Linxi "Jim" Fan¹†

**Affiliations:** ¹NVIDIA, ²Stanford, ³Caltech, ⁴UChicago, ⁵UT Austin

*Co-lead, †Co-advise

**Website:** https://nitrogen.minedojo.org

---

## Abstract

We introduce NitroGen, a vision-action foundation model for generalist gaming agents that is trained on 40,000 hours of gameplay videos across more than 1,000 games. We incorporate three key ingredients:

1. An internet-scale video-action dataset constructed by automatically extracting player actions from publicly available gameplay videos
2. A multi-game benchmark environment that can measure cross-game generalization
3. A unified vision-action model trained with large-scale behavior cloning

NitroGen exhibits strong competence across diverse domains, including combat encounters in 3D action games, high-precision control in 2D platformers, and exploration in procedurally generated worlds. It transfers effectively to unseen games, achieving up to **52% relative improvement** in task success rates over models trained from scratch. We release the dataset, evaluation suite, and model weights to advance research on generalist embodied agents.

---

## 1. Introduction

Building generally capable embodied agents that can operate in unknown environments has long been considered a holy grail of AI research. While computer vision and large language models (LLMs) have achieved this generalization through large-scale pre-training on internet data, comparable progress in embodied AI has been impeded by the lack of large, diverse, and labeled action datasets.

Video games present an ideal domain for advancing embodied AI since they offer visually rich interactive environments and tasks that span a wide range of complexities and temporal horizons.

### Prior Approach Limitations

| Approach | Limitations |
|----------|-------------|
| **LLM-based methods** | Require hand-crafted programmatic APIs or complicated perception modules for text extraction and object detection |
| **Reinforcement learning** | Achieved superhuman performance (StarCraft II, Dota 2) but agents are narrow, costly to train, and depend on specialized simulators |
| **Behavior-cloning** | Relied on expensive-to-collect demonstrations, constraining training to only a few game titles |

### Three Major Contributions

1. **Internet-scale dataset of action-labeled videos** - Using publicly available videos where content creators overlay their input commands in real time. Dataset of 40,000 hours spanning more than 1,000 games.

2. **Multi-task multi-game evaluation suite** - 30 tasks from 10 commercial games, covering combat, navigation, decision-making, platforming, exploration, and puzzle-solving. Universal Gymnasium API.

3. **Large-scale behavior-cloning pre-training** - Vision-action transformer model showing up to 52% relative improvement in success rates over models trained from scratch.

---

## 2. Approach

NitroGen consists of three novel components:
1. An internet-scale video dataset with action labels
2. A multi-game benchmark with a Gymnasium environment wrapper
3. A vision-action model pre-trained through large-scale behavior cloning

### 2.1 Internet-scale Multi-game Video-action Dataset

#### Annotation Challenge

A central challenge in training policies from internet videos is recovering the corresponding actions, since most gameplay recordings typically do not include the player's inputs. 

**Solution:** Use videos that feature **input overlay software** that displays a real-time visualization of the player's actions, typically as a 2D image of a gamepad in a corner of the screen with pressed buttons highlighted.

#### Dataset Curation

- Collected **71,000 hours** of raw video containing gamepad overlays
- Originally used primarily within the speedrunning community, now expanded to many action games
- Used keyword-based searches and curation guided by content diversity
- Covers **more than 1,000 unique games**
- Contains **38,739 videos** from **818 different content creators**
- Average video duration: **1 hour and 50 minutes**

#### Dataset Distribution

**Hours per Game:**
- 233 games: 0-1 hours
- 452 games: 1-10 hours
- 303 games: 10-100 hours
- 76 games: 100-1,000 hours
- 15 games: 1,000-10,000 hours

**Genre Distribution by Total Hours:**
| Genre | Percentage |
|-------|------------|
| Action-RPG | 34.9% |
| Platformer | 18.4% |
| Action-Adventure | 9.2% |
| Sports | 5.8% |
| Metroidvania | 5.4% |
| Roguelike | 4.9% |
| RPG | 4.7% |
| Battle Royale | 4.0% |
| Racing | 3.3% |
| Other | 9.4% |

#### Action Extraction Pipeline

**Stage 1: Template Matching**
- Apply template matching using ~300 common controller templates
- Sample 25 frames per video
- Perform feature matching with **SIFT** and **XFeat** against all curated templates
- Estimate affine transformation from paired keypoints
- Require at least 20 inliers for a valid match

**Stage 2: Gamepad Action Parsing**
- Use fine-tuned **SegFormer** segmentation model
- Process pairs of consecutive frames (concatenated along spatial dimension)
- Output segmentation mask for joystick positions on discrete **11×11 grid**
- Output binary button states
- Training data: 8M labeled synthetic frames with varied overlay opacity, controller size, and video compression
- Optimizer: AdamW with LR 0.0001, weight decay 0.1, batch size 256

**Joystick Position Processing:**
- Detect contours for each joystick over entire video
- Average positions from frames where joystick is centered
- Normalize to range [-1.0, 1.0] using 99th percentile

**Stage 3: Quality Filtering**
- Discard segments based on action density
- Keep only chunks where **≥50% of timesteps** have non-zero button or joystick actions
- Results in **55% of data being kept**
- Mask on-screen controller to prevent models from exploiting it as shortcut

### 2.2 Evaluation Suite

#### Universal Simulator

- Wraps any game title with a **Gymnasium API** for model development
- Intercepts game engine's system clock to control simulation time
- Enables frame-by-frame interaction without modifying game code
- Works with any title using system clock for physics/interactions

#### Unified Observation and Action Space

**Observations:** Single RGB frames

**Actions:**
- 16-dimensional binary vector for gamepad buttons:
  - 4 d-pad buttons
  - 4 face buttons
  - 2 shoulders
  - 2 triggers
  - 2 joystick thumb buttons
  - Start, Back
- 4-dimensional continuous vector for joystick positions

#### Evaluation Tasks

- **10 games** across diverse visual styles and genres
- **30 tasks total**
- **5 2D games:** 3 side-scrollers, 2 top-down roguelikes with procedural generation
- **5 3D games:** 2 open-world, 2 combat-focused action-RPGs, 1 sports

**Task Categories:**
- 11 combat tasks (boss fights, enemy encounters)
- 10 navigation tasks (reaching locations, traversing environments)
- 9 game-specific tasks (unique mechanics)

### 2.3 NitroGen Foundation Model

#### Architecture

- Based on **flow matching** to generate chunks of future actions conditioned on visual observations
- Adapted from **GR00T N1** with language and state encoders removed
- Single action head

**Vision Encoder:**
- **SigLIP 2** vision transformer
- Input: 256×256 RGB
- Output: 256 image tokens per frame

**Action Generation:**
- **Diffusion Transformer (DiT)**
- Outputs multiple actions per forward pass
- Noisy action chunks encoded by MLP into one action token per timestep
- Processed through DiT blocks (alternating self-attention and cross-attention)
- Cross-attention conditions on encoded frame tokens
- Final tokens decoded into continuous action vectors via MLP

#### Design Choices

- Single context frame (no benefit from multiple past frames)
- Generate **16-action chunks** for temporal consistency
- **k=16 denoising steps** at inference (additional steps yield no improvement)

#### Training

- **Conditional flow-matching objective** applied to 16-action chunks
- One 256×256 frame of context

**Image Augmentations:**
- Random brightness, contrast, saturation, hue
- Random rotation between -5 and 5 degrees
- Random crops

**Optimizer:** AdamW with weight decay 0.001

**Learning Rate Schedule:** Warmup-stable-decay (WSD) with constant LR phase of 0.0001

**EMA:** Exponential moving average of model weights with decay 0.9999 (consistently outperforms non-EMA)

---

## 3. Experiments

### Gamepad Action Extraction Performance

Evaluated by recording gameplay from 6 video games with randomized opacity, gamepad size, and type.

**Results by Controller Family:**

| Controller | Joystick R² | Button Accuracy |
|------------|-------------|-----------------|
| Xbox One | 0.92 | 98% |
| Xbox 360 | 0.91 | 98% |
| PS3 | 0.85 | 97% |
| PS4 | 0.84 | 91% |
| Xbox Series X | 0.79 | 97% |
| PS5 | 0.77 | 93% |
| **Average** | **0.84** | **96%** |

### Pre-training Results (Zero-Shot)

**Average Task Completion by Game Type:**

| Visual Style | Combat | Navigation | Game Specific |
|--------------|--------|------------|---------------|
| 3D | 61.2% | 55.0% | 56.3% |
| 2D Top-down | 46.0% | 52.0% | 61.5% |
| 2D Side-scrolling | 44.8% | 37.9% | 54.0% |

**Key Finding:** NitroGen performs well on both memorizable tasks and tasks requiring zero-shot generalization (procedural generation).

### Fine-tuning vs Training from Scratch

**Isometric Roguelike Game (varying data quantity):**

| Training Hours | Fine-tuned | From Scratch | Relative Improvement |
|----------------|------------|--------------|---------------------|
| 60h | 53.0% | 48.1% | +10% |
| 120h | 65.6% | 57.8% | +14% |
| 240h | 81.0% | 76.0% | +7% |

**3D Action-RPG (30h data, varying task type):**

| Task Type | Fine-tuned | From Scratch | Relative Improvement |
|-----------|------------|--------------|---------------------|
| Combat | 73.3% | 48.3% | **+52%** |
| Navigation | 60.0% | 48.0% | +25% |
| Game Specific | 66.6% | 63.3% | +5% |

**Key Insights:**
- Fine-tuning achieves average **10% relative improvement** on isometric roguelike
- Fine-tuning achieves average **25% relative improvement** on 3D action-RPG
- Generic tasks (combat, navigation) benefit most from pre-training
- Game-specific mechanics show marginal gains

### Dataset Noise Resilience

The model trains successfully despite multiple noise sources:
- **(a)** Actions are not strictly ground truth (input overlay software introduces delays, parsing adds inaccuracies)
- **(b)** Video frames contain creator-specific artifacts (livestream chats, subscribe prompts, progress trackers)
- **(c)** Controller configurations vary across players (sensitivity settings, custom button mappings)

---

## 4. Limitations and Future Work

### Design Limitations

- **System-1 only:** Fast-reacting sensory model
- **Cannot plan** over long horizons
- **Cannot follow language instructions**
- Only reacts to short context it sees

**Future Direction:** Post-training for language-following and reinforcement learning to enhance planning capabilities.

### Dataset Bias

- Biased toward **action games**
- Biased toward games typically played with a **gamepad**
- Keyboard-only games or those involving complex manipulation are less represented
- May limit generalization to strategy or simulation games

---

## 5. Related Works

### Gaming Agents

| Approach | Examples | Notes |
|----------|----------|-------|
| Reinforcement Learning | DQN, AlphaGo, AlphaStar, OpenAI Five | Rely on engineered rewards, hand-crafted features, specialized simulators |
| LLM-based | Voyager, Cradle | Depend on hand-crafted interfaces |
| Behavior Cloning | MineRL, VPT, SIMA, GATO, Dreamer 4, Lumine | Rely on expensive human demonstrations or RL-generated data |
| **NitroGen** | This work | Scales behavior cloning to internet-scale without costly collection |

### Concurrent Work

**Game-TARS** (Wang et al., 2025): Also trains multi-game agent using 20,000 hours combining contractor data and multi-modal reasoning data.

---

## Appendix A: Model Details

### A.1 Training Objective

Given:
- Ground-truth action chunk: a ∈ ℝ^(16×24)
- Observation: o ∈ ℝ^(256×256)
- Flow-matching timestep: t ∈ [0, 1]
- Gaussian noise: ε ~ 𝒩(0, ℐ)

Noisy action construction:
```
aₜ = (1 - t) · ε + t · a
```

Conditional velocity field:
```
u_cond(x, t, a, ε, o) = a - ε
```

Conditional flow-matching loss:
```
ℒ_CFM(θ, φ) = 𝔼_{t,a,ε} [‖πθ(aₜ, ψφ(o), t) - (a - ε)‖²]
```

Where πθ is the DiT and ψφ is the image encoder. Timestep t sampled from shifted beta distribution prioritizing small timesteps.

### A.2 Inference

Initialize: a₀ ~ 𝒩(0, ℐ)

Iteratively denoise for k steps using Euler integration:
```
a_{t+1/k} = aₜ + (1/k) · πθ(aₜ, ψφ(o), t)
```

Use **k = 16 denoising steps** (additional steps yield no measurable improvement).

---

## Appendix B: Evaluation

### B.1 Synchronous Inference

The Gymnasium API freezes the game while the model predicts the next action.

**Validation:** Recorded videos and actions of humans playing, then replayed:
- (a) In real time without pausing
- (b) While pausing and resuming with random pause durations

**Result:** Replayed sequences diverge after:
- ~1 minute for games with continuous actions
- ~3 minutes for games with discrete actions only

Same divergence pattern for both (a) and (b), confirming divergence is due to error accumulation, not the pause/resume mechanism.

---

## Key Resources

| Resource | URL |
|----------|-----|
| Website | https://nitrogen.minedojo.org |
| Model Weights | https://huggingface.co/nvidia/NitroGen |
| Dataset | https://huggingface.co/datasets/nvidia/NitroGen |
| Paper | https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf |

## Input Overlay Software Used for Training Data

- Open Joystick Display: https://github.com/AkikoKumagara/open-joystick-display
- Input Overlay: https://github.com/univrsal/input-overlay
- GamePad Viewer: https://beta.gamepadviewer.com/

## Recording Tools

- OBS (Open Broadcaster Software): https://obsproject.com/
- Input Recording Tool: https://github.com/loicmagne/input-rec

---

## Quick Reference Summary

| Spec | Value |
|------|-------|
| Training Data | 40,000 hours (filtered from 71,000) |
| Games Covered | 1,000+ |
| Videos | 38,739 |
| Content Creators | 818 |
| Model Size | ~500M parameters |
| Input Resolution | 256×256 RGB |
| Action Chunk Size | 16 actions |
| Denoising Steps | 16 (default) |
| Vision Encoder | SigLIP 2 |
| Action Head | Diffusion Transformer (DiT) |
| Best Zero-Shot Performance | 3D games (55-61%) |
| Fine-tuning Benefit | Up to 52% relative improvement |
