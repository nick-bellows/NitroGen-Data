<img src="assets/github_banner.gif" width="100%" />

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> |
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/NitroGen"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2601.02427"><strong>Paper</strong></a>
  </p>
</div>

# NitroGen — Extended Edition

> **Fork of [NVIDIA's NitroGen](https://github.com/MineDojo/NitroGen)** with custom game configurations, training pipelines, data tooling, and documentation added on top of the original foundation model.

NitroGen is an open foundation model for generalist gaming agents. This multi-game model takes pixel input and predicts gamepad actions, trained through behavior cloning on the largest video-action gameplay dataset assembled exclusively from internet videos.

---

## What's Different From the Official Repo?

This fork extends NVIDIA's base NitroGen release with substantial custom work:

### Game-Specific Configurations
Pre-built launch scripts and configurations for specific titles, each with tuned parameters:
- **Hades** — DAgger collection, multi-frame training, gameplay recording
- **Dark Souls 3** — inference with verbose logging, DAgger pipelines
- **Celeste** — precision platformer setup with action tuning

### Custom Training Pipelines
- **DAgger (Dataset Aggregation)** — iterative data collection and fine-tuning scripts (`dagger_collect.py`, `dagger_train.py`)
- **Multi-frame training** — temporal context via frame stacking for improved action prediction (`dagger_train_multiframe.py`, `frame_stacking.py`)
- **Weighted training** — loss weighting strategies for imbalanced action distributions (`dagger_train_weighted.py`)

### NVIDIA Dataset Tooling
End-to-end pipelines for working with NVIDIA's gameplay dataset:
- **Index pipeline** (`nvidia_index/`) — download shards, extract metadata, build a searchable master index
- **Data pipeline** (`nvidia_pipeline/`) — explore, download, extract frames, apply labels, and merge datasets
- Standalone tools: video downloader, frame extractor, dataset explorer, label applicator

### Optimized Inference
- `serve_optimized.py` — performance-tuned model server
- `serve_multiframe.py` — multi-frame context inference server
- `play_fast.py` — low-latency gameplay loop
- `play_inference_only.py` — headless inference without game attachment

### Debug & Benchmarking Tools
- Controller input tester and visualizer
- Action space debugger
- Inference benchmark suite

### Documentation
Comprehensive guides covering architecture, training workflows, dataset handling, and project structure in the `Documentation/` directory.

---

## Installation

### Prerequisites

We **do not distribute game environments** — you must use your own copies of the games. This repository only supports running the agent on **Windows games**. You can serve the model from a Linux machine for inference, but the game must run on Windows. Tested on Windows 11 with Python ≥ 3.12.

### Setup

```bash
git clone https://github.com/nick-bellows/NitroGen-Data.git
cd NitroGen-Data
pip install -e .
```

Download the NitroGen checkpoint from [HuggingFace](https://huggingface.co/nvidia/NitroGen):
```bash
hf download nvidia/NitroGen ng.pt
```

## Quick Start

**Start the inference server:**
```bash
python scripts/serve.py <path_to_ng.pt>
```

**Run the agent on any game:**
```bash
python scripts/play.py --process '<game_executable_name>.exe'
```

The `--process` parameter must be the exact executable name of the game. Find it via Windows Task Manager (Ctrl+Shift+Esc) → right-click process → Properties → General tab.

**Or use a pre-configured game script:**
```bash
# Example: Hades
games/hades/play.bat
```

See [QUICKSTART.md](QUICKSTART.md) for a full walkthrough and the [Documentation/](Documentation/) directory for in-depth guides.

---

## Project Structure

```
├── nitrogen/              # Core model code (from NVIDIA, with minor fixes)
├── scripts/               # Training, inference, and data processing scripts
│   ├── serve.py           # Base model server
│   ├── play.py            # Base gameplay loop (from NVIDIA)
│   ├── dagger_*.py        # DAgger training pipelines (custom)
│   ├── serve_*.py         # Optimized/multi-frame servers (custom)
│   ├── play_*.py          # Optimized gameplay loops (custom)
│   ├── record_gameplay.py # Gameplay recording (custom)
│   └── nvidia_index/      # Dataset index tooling (custom)
├── games/                 # Per-game configs and launch scripts (custom)
├── nvidia_index/          # Dataset index pipeline scripts (custom)
├── nvidia_pipeline/       # Dataset processing pipeline (custom)
├── tools/                 # Debug and benchmarking utilities (custom)
└── Documentation/         # Architecture and workflow guides (custom)
```

## Paper and Citation

Original paper by the NVIDIA NitroGen team:

```bibtex
@misc{magne2026nitrogen,
      title={NitroGen: An Open Foundation Model for Generalist Gaming Agents},
      author={Loïc Magne and Anas Awadalla and Guanzhi Wang and Yinzhen Xu and Joshua Belofsky and Fengyuan Hu and Joohwan Kim and Ludwig Schmidt and Georgia Gkioxari and Jan Kautz and Yisong Yue and Yejin Choi and Yuke Zhu and Linxi "Jim" Fan},
      year={2026},
      eprint={2601.02427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.02427},
}
```

**Disclaimer**: This project is strictly for research and educational purposes and is not an official NVIDIA product. The original NitroGen model and dataset are the work of the NVIDIA team. Custom extensions in this fork are my own work.
