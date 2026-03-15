# NitroGen Project Implementation Plan

**Last Updated:** January 9, 2026 | **Version:** 9

---

## ✅ Verified System Status

All prerequisites verified and working as of January 9, 2026.

| Component | Version | Status |
|-----------|---------|--------|
| GPU | NVIDIA GeForce RTX 4080 (16GB) | ✅ Verified |
| CUDA | 12.9 | ✅ Verified |
| NVIDIA Driver | 577.00 | ✅ Verified |
| Python | 3.13.7 | ✅ Works (monitor for compat) |
| Git | 2.52.0.windows.1 | ✅ Verified |
| Git LFS | Initialized | ✅ Verified |
| ViGEmBus | Nefarius Virtual Gamepad Emulation Bus | ✅ Verified |
| RAM | 128GB DDR5 | ✅ Confirmed |
| CPU | AMD Ryzen 9 7900X (12C/24T) | ✅ Confirmed |
| OS | Windows 11 | ✅ Confirmed |

> **Note:** Python 3.13.7 is very new. If you encounter dependency issues, install Python 3.11 alongside using pyenv-win or direct installer.

---

## Claude Code Introductory Prompt

Copy and paste this prompt when starting a new Claude Code session:

```
# NitroGen Gaming AI Project Context

## Project Overview
I'm building a gaming AI system for YouTube content using NVIDIA's NitroGen model.
Goal: Create "AI plays games" videos showcasing AI learning various games.

## Key Repositories
- NitroGen main: https://github.com/MineDojo/NitroGen
- DAgger server: https://github.com/fmthola/NitroGen-Monitor-Server
- Weights: https://huggingface.co/nvidia/NitroGen
- Dataset: https://huggingface.co/datasets/nvidia/NitroGen

## Verified Hardware (Jan 9, 2026)
- GPU: NVIDIA RTX 4080 Super (16GB VRAM)
- CUDA: 12.9 | Driver: 577.00
- CPU: AMD Ryzen 9 7900X (12C/24T)
- RAM: 128GB DDR5
- OS: Windows 11
- Python: 3.13.7 (may need 3.11 for some deps)
- Git: 2.52.0, Git LFS: Initialized
- ViGEmBus: Installed (virtual Xbox controller)

## NitroGen Architecture
- Vision: SigLIP 2 (256x256 -> 256 tokens)
- Action head: Diffusion Transformer
- Output: 16-action chunks, Xbox controller only
- Params: ~500M

## Three Training Methods
1. Pure RL (PPO/SAC): Learn from rewards, superhuman potential
2. DAgger: 30-60 min human corrections via Monitor-Server
3. Behavior Cloning: 30-60h recordings, teaches playstyle

## Expected Performance (RTX 4080)
- timesteps=4: 20-25 FPS (start here)
- timesteps=8: 11-14 FPS

## Current Phase: [UPDATE THIS]
Phase 1 - Environment Setup & Zero-Shot Testing

## What I Need Help With: [UPDATE THIS]
Getting NitroGen running for the first time
```

---

## Project Phases

### Phase 1: Environment Setup & Zero-Shot Testing (Weeks 1-4)

#### Week 1: Environment Setup
- [x] Verify GPU, CUDA, drivers
- [x] Install Git, Git LFS
- [x] Install ViGEmBus
- [ ] Clone NitroGen repository
- [ ] Install Python dependencies
- [ ] Download model weights from HuggingFace

#### Week 2: First Tests
- [ ] Test on Celeste (default game in NitroGen code)
- [ ] Experiment with timesteps parameter (start at 4)
- [ ] Set up OBS for recording gameplay
- [ ] Record first "AI plays" footage

#### Weeks 3-4: Zero-Shot on Multiple Games
- [ ] Test Elden Ring (top priority, 34.9% of training data)
- [ ] Test Dark Souls III (explicitly in paper)
- [ ] Test Batman Arkham series (freeflow combat)
- [ ] **Deliverable: First YouTube video**

### Phase 2: Training Methods (Weeks 5-8)
- Clone NitroGen-Monitor-Server for DAgger
- Implement DAgger training on Sekiro (parry timing)
- Set up OBS + Gamepad Viewer for behavior cloning
- Record 40-60h gameplay, train behavior cloning
- **Deliverable: Before/After comparison video**

### Phase 3: Hybrid LLM + NitroGen (Weeks 9-12) - Optional
- Build frame classifier (combat vs menu vs dialogue)
- Integrate Claude API for menu/dialogue handling
- Test on Yakuza: Like a Dragon (turn-based + exploration)
- **Deliverable: Claude + NitroGen Play Yakuza video**

---

## Game Categories Quick Reference

| Category | Works With | Example Games |
|----------|------------|---------------|
| 🟢 Perfect | Pure NitroGen | Elden Ring, DS3, Sekiro, Batman, Celeste, Hades |
| 🟡 Partial | NitroGen + human help | Yakuza 0, Skyrim, Witcher 3, Tomb Raider |
| 🟠 Hybrid | Claude + NitroGen | Like a Dragon, Pokemon, Persona |
| 🔴 Won't Work | Mouse-only games | EU4, Civ, Cities Skylines, WoW |

---

## Quick Links

- **NitroGen repo:** github.com/MineDojo/NitroGen
- **DAgger server:** github.com/fmthola/NitroGen-Monitor-Server
- **Model weights:** huggingface.co/nvidia/NitroGen
- **ViGEmBus:** github.com/ViGEm/ViGEmBus
- **Gamepad Viewer:** gamepadviewer.com

---

## Project Documentation Index

| Document | Contents |
|----------|----------|
| `Project_Plan.md` | This document. System status, Claude Code prompt, phases, quick reference. |
| `Complete_Guide.md` | 75+ games categorized, hybrid architecture details, research references. |
| `Content_Strategy.md` | YouTube video ideas, 12-week content calendar, thumbnail formulas. |
| `Project_Structure.md` | Multi-repo architecture, menu handling strategies, per-game configs. |
