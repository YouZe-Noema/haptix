# haptix Reading List

Ongoing literature review for tactile data infrastructure.
Papers are tagged by relevance and status.

## Legend

- 📖 To read | 📚 Reading | ✅ Read | 🗄️ Archived
- 🔑 Key paper | 📊 Dataset | 🔧 Method | 📋 Survey | 🏭 Industry

---

## Foundational Surveys

| Status | Tag | Paper | Key Insight |
|---|---|---|---|
| ✅ | 🔑📋 | Luo et al. (2017) "Robotic tactile perception of object properties: A review." *Mechatronics* | Three-tier sensor taxonomy; interpretation gap |
| 📖 | 📋 | Kappassov et al. (2015) "Tactile sensing in dexterous robot hands — Review." *Robotics and Autonomous Systems* | Dexterous hand sensor integration |
| 📖 | 📋 | Dahiya et al. (2010) "Tactile sensing — from humans to humanoids." *IEEE T-RO* | Foundational sensor survey |

## Datasets & Benchmarks

| Status | Tag | Paper/Dataset | Sensor | Notes |
|---|---|---|---|---|
| 📖 | 📊 | Touch and Go (Stanford, 2023) | GelSight | Tactile-driven navigation |
| 📖 | 📊 | YCB-Slide / YCB-Touch (MIT/CMU) | GelSight | YCB objects with tactile |
| 📖 | 📊 | RoboTouch (CMU) | DIGIT | Manipulation dataset |
| 📖 | 📊 | Stanford NeuTouch (2021) | NeuTouch | Event-driven tactile |
| 📖 | 🔑📊 | TactiDex (2026.07) | Multiple | Latest tactile dexterity benchmark |
| 📖 | 📊 | Lab-CORO TactileDataset | Capacitive | Real+sim grasping contacts |

## Methods & Architectures

| Status | Tag | Paper | Key Contribution |
|---|---|---|---|
| 📖 | 🔧 | UniForce (2026.02) | Unified latent force across diverse sensors |
| ✅ | 🔑🔧 | **UniForce** (Chen, Ni, Luo, Lepora et al., 2026.02) | Learns shared latent force space across GelSight/TacTip/uSkin. Authors include Shan Luo (survey author) + Nathan Lepora (TacTip). "Code and datasets will be released." |
| 📖 | 🔧 | DECO / Plugin Tactile Adapter (2026.02) | Drop-in tactile adapter for diffusion policies |
| 📖 | 🔧 | Human-Centric Tactile Pre-Training (2026.07) | Human→robot tactile transfer learning |
| 📖 | 🔧 | DexViTac (2026.03) | Visuo-tactile-kinematic data collection |

## Sensor Hardware

| Status | Tag | Sensor | Lab / Company | Notes |
|---|---|---|---|---|
| ✅ | 🏭 | DIGIT / DIGIT v2 | Meta FAIR | First haptix adapter target |
| 📖 | 🏭 | GelSight | MIT / GelSight Inc | Next adapter target |
| 📖 | 🏭 | AnySkin | Meta / NYU | Replaceable elastomer |
| 📖 | 🏭 | BioTac | SynTouch | Commercial biomimetic |
| 📖 | 🏭 | TacTip | Bristol Robotics Lab | Open-source biomimetic |

## Formats & Standards

| Status | Tag | Paper/Standard | Notes |
|---|---|---|---|
| 📖 | 🔑 | MPEG Haptics (ISO 23090-31) | Human haptic coding, not ML data |
| 📖 | 🔑 | IEEE P1918.1.1 Haptic Codecs | Tactile internet codecs |
| 📖 | 🔧 | tlabel schema v2.1 | Closest existing unified schema |

## Potential Collaborators

| Researcher | Institution | Relevance |
|---|---|---|
| Shan Luo | Harvard Medical School | Tactile perception, survey author, **UniForce co-author** |
| Nathan Lepora | University of Bristol | TacTip inventor, **UniForce co-author** |
| Zhuo Chen | (UniForce lead) | Unified tactile representations |
| Roberto Calandra | Meta FAIR / TU Dresden | DIGIT, tactile RL |
| Edward Adelson | MIT CSAIL | GelSight inventor |
| Nathan Lepora | Bristol / University of Bristol | TacTip, tactile robotics |
| Wenzhen Yuan | CMU / UIUC | GelSight, RoboTouch |

---

*Last updated: 2026-07-27. This file is maintained by haptix-dev during overnight development sessions.*
