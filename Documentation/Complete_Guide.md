# NitroGen Complete Guide

**75+ Games Analyzed | Hybrid Architecture | Research References**

---

## Table of Contents
1. [NitroGen Capabilities](#nitrogen-capabilities)
2. [Game Categories](#game-categories)
3. [Hybrid LLM + NitroGen Architecture](#hybrid-llm--nitrogen-architecture)
4. [Research References](#research-references)

---

## NitroGen Capabilities

### What NitroGen CAN Do
- ✅ Real-time combat (Souls-likes, action games)
- ✅ Platforming (Celeste, Hollow Knight)
- ✅ Racing (Forza Horizon)
- ✅ Twin-stick shooters (Enter the Gungeon)
- ✅ Boss fights
- ✅ Controller-based gameplay

### What NitroGen CANNOT Do
- ❌ Read text/dialogue
- ❌ Navigate complex menus
- ❌ Inventory management
- ❌ Long-term planning/strategy
- ❌ Mouse-only games
- ❌ Turn-based combat (without hybrid approach)

---

## Game Categories

### 🟢 Category A: Perfect NitroGen Games (50+ games)

These work out-of-the-box with pure NitroGen.

#### Souls-likes & Action RPGs (8 games)
| Game | Platform | Notes |
|------|----------|-------|
| **ELDEN RING** | Steam | 34.9% of training data, TOP PRIORITY |
| Dark Souls III | Steam | Explicitly in NitroGen paper |
| Sekiro | Steam | "Can AI Learn to Parry?" content |
| Black Myth: Wukong | Steam | 95.89% rating, 2024 hit |
| Lies of P | Game Pass | Excellent Souls-like |
| Wo Long: Fallen Dynasty | Game Pass | Team Ninja, deflect combat |
| Nioh 2 Complete | Steam | Stance system |
| Sifu | Steam | Roguelike structure |

#### Action/Combat Games (8 games)
| Game | Platform | Notes |
|------|----------|-------|
| Batman: Arkham Asylum | Steam | PERFECT freeflow combat |
| Batman: Arkham City | Steam | Combat Challenges ideal |
| Batman: Arkham Knight | Steam | Most refined combat |
| Shadow of Mordor | Steam | Arkham-style, Nemesis system |
| Shadow of War | Steam | Expanded Nemesis |
| Mad Max | Steam | 90.92% rating, vehicle + melee |
| Ghost of Tsushima | Steam | Samurai combat, standoffs |
| Marvel's Spider-Man | Steam | Web-swinging + combat |

#### Roguelikes & Platformers (12 games)
| Game | Platform | Notes |
|------|----------|-------|
| **Celeste** | Game Pass | DEFAULT in NitroGen code! Start here |
| Hades | Game Pass | Perfect roguelike structure |
| Hollow Knight | Game Pass | In training data |
| Dead Cells | Game Pass | Fast combat, in training data |
| Cuphead | Game Pass | Boss-focused, visual showcase |
| Enter the Gungeon | Steam | 93.84% rating, twin-stick |
| Katana ZERO | Steam | 96.34% rating, fast action |
| Ori and the Will of the Wisps | Game Pass | Beautiful, has combat |
| Ghostrunner 1 & 2 | Steam | One-hit deaths, parkour |
| Death's Door | Steam | Zelda-like combat |
| Blasphemous 2 | Steam | Dark Metroidvania |
| Little Nightmares I & II | Steam | Horror + stealth + platforming |

#### FPS & Shooters (4 games)
| Game | Platform | Notes |
|------|----------|-------|
| DOOM Eternal | Steam | Fast FPS, resource loop |
| Trepang2 | Steam | F.E.A.R. successor, slow-mo |
| Metal: Hellsinger | Steam | FPS + rhythm hybrid |
| Resident Evil 2/Village/7 | Steam | Horror + combat |

#### Additional Perfect Games
- Forza Horizon 5 (Game Pass) - Racing
- Tomb Raider trilogy (Steam)
- Assassin's Creed Origins/Odyssey/Valhalla
- NieR: Automata/Replicant (Steam)
- A Plague Tale series (Steam)
- Bomb Rush Cyberfunk (Steam)
- Hi-Fi RUSH (Both) - 95.14% rating

---

### 🟡 Category B: Partial Support (15+ games)

Combat works, but dialogue/menus need manual skipping.

#### Yakuza Action Games
| Game | Hours Owned | Notes |
|------|-------------|-------|
| Yakuza 0 | 63.6h | Beat 'em up, 40-50% dialogue |
| Yakuza Kiwami 1/2 | 30h+ | Heavy dialogue |
| Like a Dragon: Ishin! | 40.5h | Samurai action |
| Like a Dragon Gaiden | - | 93% rating, more action-focused |
| Judgment/Lost Judgment | - | Two fighting styles |

#### Open World RPGs
| Game | Notes |
|------|-------|
| Skyrim Special Edition | Combat/exploration works, dialogue trees manual |
| The Witcher 3 | Combat + horse riding works, in training data |
| Red Dead Redemption 2 | Combat/riding works, slow pacing |
| Cyberpunk 2077 | FPS combat/driving works, heavy dialogue |
| BioShock series | FPS combat works, weapon wheel = menus |

**Workaround:** Manually skip dialogue/cutscenes, let AI handle combat sections.

---

### 🟠 Category C: Hybrid Architecture Required (10+ games)

These NEED LLM brain (Claude/GPT-4V) for menu navigation, planning, text reading.

| Game | Challenge | Hybrid Solution |
|------|-----------|-----------------|
| Like a Dragon | Turn-based menu combat | LLM reads battle UI, selects actions |
| Infinite Wealth | Turn-based combat | Same approach |
| Pokemon (Emulator) | Text + menus | LLM reads text/selects moves |
| Persona 3/4/5 | Turn-based + social sim | LLM handles dialogue/combat menus |
| Final Fantasy (turn-based) | Menu-based combat | LLM reads HP/MP/weaknesses |
| Stardew Valley | Planning required | LLM decides what to plant/harvest |
| The Escapists 1/2 | Multi-step planning | LLM handles schedules, crafting |

---

### 🔴 Category D: Won't Work

Mouse-only games incompatible with NitroGen's controller output.

| Game | Hours | Reason |
|------|-------|--------|
| Europa Universalis IV/V | 408h | Mouse-only grand strategy |
| Crusader Kings II/III | 292h | Mouse-only dynasty sim |
| Stellaris | 182.5h | Mouse-only 4X |
| Hearts of Iron IV | 179.5h | Mouse-only WW2 strategy |
| Civilization V/VI/VII | 209h | Turn-based, mouse-only |
| Cities: Skylines | 56.9h | Mouse-only city builder |
| Baldur's Gate 3 | 85.2h | Turn-based isometric, mouse-based |

---

## Hybrid LLM + NitroGen Architecture

### The Core Problem

| Approach | Strength | Fatal Weakness |
|----------|----------|----------------|
| Pure NitroGen | Fast (20+ FPS), handles action | Can't read text, navigate menus |
| Pure LLM (Claude) | Can read, understand, strategize | Too slow (~2-5 sec/action), expensive |
| Pure RL (Pokemon Red) | Can learn anything | Takes 50,000+ hours |

### The Solution: "Brain + Hands"

```
GAME SCREEN
    ↓
MODE DETECTOR (combat vs menu vs dialogue)
    ↓                              ↓
Combat Mode                   Menu/Dialogue Mode
    ↓                              ↓
NitroGen (20+ FPS)            Claude API
- Dodge                       - Reads text
- Attack                      - Selects optimal actions
- Navigate                    - "Use Thunderbolt"
```

### Implementation: Mode Switching

```python
while game_running:
    frame = capture_screen()
    mode = detect_mode(frame)  # 'combat', 'menu', 'dialogue'
    
    if mode == 'combat':
        action = nitrogen.infer(frame)  # Fast, 20+ FPS
    elif mode == 'menu' or mode == 'dialogue':
        action = claude.decide(frame)   # Slow but smart
    
    execute(action)
```

### Local LLM vs API for Hybrid

| Factor | Local (Ollama) | Claude API |
|--------|----------------|------------|
| Latency | 100-500ms ✅ | 1-3 sec |
| Capability | Good | Excellent ✅ |
| Cost | Free ✅ | ~$0.003-0.015/call |
| VRAM Impact | 8-14GB (competes with game) | 0 GB ✅ |

**Recommendation:** Use Claude API for strategic decisions (menus, battles) since latency doesn't matter for turn-based content.

---

## Research References

| Project | Game | Approach |
|---------|------|----------|
| Pokemon Red RL (Peter Whidden) | Pokemon Red | Pure RL, 50,000+ hours. Got stuck on "Deliver parcel" - can't read text. |
| PokeLLMon | Pokemon battles | GPT-4 reads battle state, human-parity in battles. |
| Voyager | Minecraft | LLM generates code for tasks, builds skill library. |
| JARVIS-1 | Minecraft | Memory-augmented multimodal LLM. |
| PORTAL (March 2025) | 1000s of 3D games | LLM generates behavior trees, neural nets execute. |
| Cradle | RDR2, others | Foundation agents for general computer control. |

### Pokemon: The Perfect Hybrid Test Case

Peter Whidden's viral approach:
- Pure RL, no text understanding
- 50,000+ hours of training
- AI got stuck because it couldn't read "Deliver this parcel"

Hybrid approach could solve in hours:

| Task | Handler | Why |
|------|---------|-----|
| Walking around | NitroGen | Real-time navigation |
| Reading dialogue | Claude | Can understand text |
| Battle decisions | Claude | Knows type matchups |
| Menu navigation | Claude | Can read options |

---

## Game Count Summary

| Category | Count | Notes |
|----------|-------|-------|
| 🟢 Perfect | 50+ | Pure NitroGen |
| 🟡 Partial | 15+ | Combat works, skip dialogue |
| 🟠 Hybrid | 10+ | Innovative content opportunity |
| 🔴 Won't Work | Many | Mouse-only strategy games |
| **Total Viable** | **75+** | |
