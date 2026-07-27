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

| Status | Tag | Paper/Dataset | Sensor | Notes | arXiv |
|---|---|---|---|---|---|
| 📖 | 📊 | **Touch and Go** (Yang, Ma, Zhang, Zhu, Yuan, Owens, 2022) | GelSight | Paired egocentric video + tactile data in the wild. NeurIPS 2022 Datasets & Benchmarks. | [2211.12498](https://arxiv.org/abs/2211.12498) |
| 📖 | 📊 | YCB-Slide / YCB-Touch (MIT/CMU) | GelSight | YCB objects with tactile | — |
| 📖 | 📊 | RoboTouch (CMU) | DIGIT | Manipulation dataset | — |
| 📖 | 📊 | Stanford NeuTouch (2021) | NeuTouch | Event-driven tactile | — |
| 📖 | 🔑📊 | **TactiDex** (Ni, Zhang, Wei, Chen et al., 2026.07) | Multiple | Real-world tactile-guided benchmark for human-like dexterous manipulation. Introduces TactiSkill (tri-component tactile reward). 7 authors, 18 pages. | [2607.09190](https://arxiv.org/abs/2607.09190) |
| 📖 | 📊 | Lab-CORO TactileDataset | Capacitive | Real+sim grasping contacts | — |
| 📖 | 📊 | **OPENTOUCH** (Song, Li, Fu et al., MIT/CMU 2025.12) | Wearable tactile | **First in-the-wild egocentric full-hand tactile dataset.** 5.1 hrs of synchronized video-touch-pose + 2,900 annotated clips. Retrieval & classification benchmarks. Authors include Torralba, Matusik, Du, Liang. | [2512.16842](https://arxiv.org/abs/2512.16842) |
| 📖 | 🔑📊 | **ViTacWorld** (Huang, Sang, Lu, Ni et al., 2026.07) | Multiple | Scaling visuo-tactile world models for contact-rich robot manipulation. Shares authors with TactiDex (Ni, Shi, Wang). Project page: https://vitacworld.github.io/ | [2607.22530](https://arxiv.org/abs/2607.22530) |

## Methods & Architectures

| Status | Tag | Paper | Key Contribution | arXiv |
|---|---|---|---|---|
| 📖 | 🔧 | **UniForce** (Chen, Ni, Luo, Lepora et al., 2026.02) | Learns shared latent force space across GelSight/TacTip/uSkin. Inverse + forward dynamics with force equilibrium. Full author list: Zhuo Chen, Fei Ni, Kaiyao Luo, Zhiyuan Wu, Xuyang Zhang, Emmanouil Spyrakos-Papastavridis, Lorenzo Jamone, Nathan F. Lepora, Jiankang Deng, Shan Luo. "Code and datasets will be released." | [2602.01153](https://arxiv.org/abs/2602.01153) |
| ✅ | 🔑🔧 | **UniForce** — abstract confirmed | Unified latent force cross-sensor. Heterogeneous sensors (optical vs magnetic) aligned via static equilibrium force pairing. VTLA model integration for robotic wiping task. | [2602.01153](https://arxiv.org/abs/2602.01153) |
| 📖 | 🔧 | DECO / Plugin Tactile Adapter (2026.02) | Drop-in tactile adapter for diffusion policies | — |
| 📖 | 🔧 | Human-Centric Tactile Pre-Training (2026.07) | Human→robot tactile transfer learning | — |
| 📖 | 🔧 | DexViTac (2026.03) | Visuo-tactile-kinematic data collection | — |
| 📖 | 🔧 | **Multi-Resolution Tactile Imitation Learning** (Krohn, Helmut, Funk et al., 2026.06) | Tactile imitation learning for contact-rich manipulation. 20 pages. | [2606.06281](https://arxiv.org/abs/2606.06281) |
| 📖 | 🔧 | **DreamTacVLA** (Ye et al., 2025.12) | "Learning to Feel the Future" — tactile VLA for contact-rich manipulation. | [2512.23864](https://arxiv.org/abs/2512.23864) |
| 📖 | 🔧 | **VTLoc** (Wu, Chen, Luo, 2026.07) | Learning-based tactile contact localization in visual point clouds. From UniForce group. | [2607.16146](https://arxiv.org/abs/2607.16146) |

## Sensor Hardware

| Status | Tag | Sensor | Lab / Company | Notes |
|---|---|---|---|---|
| ✅ | 🏭 | DIGIT / DIGIT v2 | Meta FAIR | First haptix adapter target |
| 📖 | 🏭 | GelSight | MIT / GelSight Inc / CMU/UIUC | Next adapter target |
| 📖 | 🔑🏭 | **GelSight Modular Design** (Agarwal, Mirzaee, Sun, Yuan, 2025) | CMU / UIUC | Modularized design approach for GelSight family. OptiSense Studio toolbox. Accepted to IJRR. | [2504.14739](https://arxiv.org/abs/2504.14739) |
| 📖 | 🏭 | **GelSight FlexiRay** (Wang, Wu, Guo, Dong, 2024.11) | — | Flexible, full-coverage multimodal sensing beyond planar limits. | [2411.18979](https://arxiv.org/abs/2411.18979) |
| 📖 | 🏭 | AnySkin | Meta / NYU | Replaceable elastomer |
| 📖 | 🏭 | BioTac | SynTouch | Commercial biomimetic |
| 📖 | 🏭 | TacTip | Bristol Robotics Lab | Open-source biomimetic |
| 📖 | 🏭 | uSkin | — | Magnetic-based tactile sensor used in UniForce |

## Formats & Standards

| Status | Tag | Paper/Standard | Notes |
|---|---|---|---|
| 📖 | 🔑 | MPEG Haptics (ISO 23090-31) | Human haptic coding, not ML data |
| 📖 | 🔑 | IEEE P1918.1.1 Haptic Codecs | Tactile internet codecs |
| 📖 | 🔧 | tlabel schema v2.1 | Closest existing unified schema |

## Potential Collaborators

| Researcher | Institution | Relevance |
|---|---|---|
| Shan Luo | Harvard / King's College London | Tactile perception, **UniForce co-author**, VTLoc co-author |
| Nathan Lepora | University of Bristol | TacTip inventor, **UniForce co-author** |
| Zhuo Chen | (UniForce lead) | Unified tactile representations, VTLoc |
| Roberto Calandra | Meta FAIR / TU Dresden | DIGIT, tactile RL |
| Edward Adelson | MIT CSAIL | GelSight inventor |
| Wenzhen Yuan | CMU / UIUC | GelSight, RoboTouch, Touch and Go, GelSight modular design |
| Fengyu Yang | (Touch and Go lead) | Visuo-tactile learning, Touch and Go dataset |
| Andrew Owens | (Touch and Go) | Touch and Go co-author |
| Yuxin Ray Song | MIT | OPENTOUCH lead — full-hand tactile dataset |
| Paul Pu Liang | MIT / CMU | OPENTOUCH — multimodal egocentric perception |
| Suting Ni | (TactiDex lead) | Tactile-guided dexterous manipulation, ViTacWorld |
| Ye Shi / Jingya Wang | (TactiDex / ViTacWorld) | Tactile world models + benchmarks |
| Arpit Agarwal | CMU | GelSight modular design (IJRR 2025) |
| Yiyue Luo | MIT | OPENTOUCH, tactile sensing |

---

## Recent Discoveries (2026.07 round)

1. **TactiDex** (2607.09190) — confirmed as major 2026 benchmark. Uses multiple sensors for human-like dexterous manipulation. Introduces TactiSkill tri-component tactile reward.
2. **ViTacWorld** (2607.22530) — from same group as TactiDex. Scales visuo-tactile world models. Brand new (Jul 24 2026).
3. **OPENTOUCH** (2512.16842) — first full-hand egocentric tactile dataset. MIT/CMU collaboration. 5.1hr, 2900 clips. Highly relevant for haptix data format work.
4. **UniForce** (2602.01153) — confirmed cross-sensor force representation. Heterogeneous sensor alignment via static equilibrium.
5. **GelSight Modular Design** (2504.14739) — accepted IJRR. OptiSense Studio toolbox for sensor design optimization.
6. **VTLoc** (2607.16146) — from UniForce group (Wu, Chen, Luo). Tactile contact localization in visual point clouds.

*Last updated: 2026-07-27. This file is maintained by haptix-dev during overnight development sessions.*
