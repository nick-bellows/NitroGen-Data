# NitroGen Project Documentation

Documentation for the NitroGen Gaming AI YouTube content project.

## Quick Start

1. Read **Project_Plan.md** first - contains verified system status and Claude Code intro prompt
2. Reference **Complete_Guide.md** for game compatibility
3. Use **Content_Strategy.md** when planning videos
4. Use **Project_Structure.md** when ready for multi-game repos

## Documents

| Document | Purpose | When to Use |
|----------|---------|-------------|
| `Project_Plan.md` | Implementation guide, system specs, Claude Code prompt | Starting each work session |
| `Complete_Guide.md` | 75+ games categorized, hybrid architecture details | Choosing games, understanding capabilities |
| `Content_Strategy.md` | YouTube video ideas, content calendar | Planning content, video titles |
| `Project_Structure.md` | Multi-repo architecture, menu handling | Setting up per-game repositories |

## For Claude Code

Claude Code can easily read all `.md` files. Just ask:
```
Read all the markdown files in the Documentation folder to understand the project context.
```

## Verified System Status (Jan 9, 2026)

| Component | Version | Status |
|-----------|---------|--------|
| GPU | RTX 4080 Super (16GB) | ✅ Verified |
| CUDA | 12.9 | ✅ Verified |
| Driver | 577.00 | ✅ Verified |
| Python | 3.13.7 | ✅ Works |
| Git | 2.52.0 | ✅ Verified |
| Git LFS | Initialized | ✅ Verified |
| ViGEmBus | Installed | ✅ Verified |

## Key Links

- **NitroGen Repo**: https://github.com/MineDojo/NitroGen
- **DAgger Server**: https://github.com/fmthola/NitroGen-Monitor-Server
- **Model Weights**: https://huggingface.co/nvidia/NitroGen
- **Dataset**: https://huggingface.co/datasets/nvidia/NitroGen

## Project Phases

1. **Phase 1 (Weeks 1-4)**: Environment setup, zero-shot testing on Celeste/Elden Ring
2. **Phase 2 (Weeks 5-8)**: DAgger training, behavior cloning
3. **Phase 3 (Weeks 9-12)**: Hybrid LLM + NitroGen architecture (optional)
4. **Phase 4 (Weeks 13-16)**: Pokemon project comparison (optional)

## Game Categories

- 🟢 **Perfect** (50+ games): Elden Ring, Dark Souls, Sekiro, Batman Arkham, Celeste, Hades
- 🟡 **Partial** (15+ games): Yakuza 0, Skyrim, Witcher 3 (combat works, skip dialogue)
- 🟠 **Hybrid** (10+ games): Like a Dragon, Pokemon, Persona (need Claude + NitroGen)
- 🔴 **Won't Work**: EU4, Civilization, Cities Skylines (mouse-only)

---

*Last updated: January 9, 2026*
