# 🦾 VLASH-Piper: VLA Fine-Tuning & Deployment for PIPER Arm

[![Docs](https://img.shields.io/badge/Docs-Zensical-blue?logo=readthedocs)](https://alvin0523.github.io/vlash-piper/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20AGX%20Orin-76b900?logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-orin/)
[![Pixi](https://img.shields.io/badge/Pixi-Package%20Manager-brightgreen?logo=conda-forge)](https://pixi.sh)
[![VLASH](https://img.shields.io/badge/VLASH-arXiv%3A2512.01031-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.01031)
[![LeRobot](https://img.shields.io/badge/🤗%20LeRobot-v0.4.1-ffcc00)](https://github.com/huggingface/lerobot)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-orange)](https://huggingface.co/lerobot)
[![Policy](https://img.shields.io/badge/Policy-π0.5-8b5cf6)](https://github.com/mit-han-lab/vlash)
[![Robot](https://img.shields.io/badge/Robot-AgileX%20PIPER-0ea5e9)](https://github.com/agilexrobotics/piper_sdk)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros)](https://docs.ros.org/en/humble/)
[![Course](https://img.shields.io/badge/HKUST-COMP4901D%20Embedded%20AI-003366?logo=academia)](https://www.cse.ust.hk/)

> Fine-tune π0.5 on your AgileX PIPER robotic arm and deploy with real-time async inference — powered by [VLASH](https://github.com/mit-han-lab/vlash) and [LeRobot](https://github.com/huggingface/lerobot).

[![Read the Docs](https://img.shields.io/badge/📖%20Read%20the%20Docs-Operation%20Guide%20%26%20Reference-blue?style=for-the-badge)](https://alvin0523.github.io/vlash-piper/)

---

## 📖 About

This workspace combines two git submodules into a single [Pixi](https://pixi.sh)-managed workflow for data collection, fine-tuning, and deployment of VLA policies on a PIPER robotic arm running on **Jetson AGX Orin**.

| Submodule | Role | Original Repo |
|---|---|---|
| `vlash/` | Async VLA fine-tuning & inference framework | [mit-han-lab/vlash](https://github.com/mit-han-lab/vlash) |
| `lerobot_piper/` | LeRobot adapted for PIPER arm (CAN bus) | [WeGo-Robotics/lerobot_piper](https://github.com/WeGo-Robotics/lerobot_piper) |

---

## 📁 Project Structure

```
vlash-piper/
├── data/                    # datasets (place LeRobot datasets here)
├── docs/                    # integration guides, diagnostics & reports
│   ├── index.md
│   ├── guide.md
│   ├── integration.md
│   ├── problems.md
│   └── report.md
├── lerobot_piper/           # submodule — LeRobot fork for PIPER arm
├── models/                  # model checkpoints (output_dir goes here)
├── scripts/                 # helper shell scripts
│   ├── activate.sh
│   ├── init_orin_can.sh
│   ├── record.sh
│   └── teleop.sh
├── vlash/                   # submodule — VLASH async inference framework
├── pixi.toml                # Pixi task definitions & dependencies
└── zensical.toml            # Zensical docs site config
```

---

## 🚀 Quick Start

**1. Install [Pixi](https://pixi.sh) (one-time):**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**2. Clone and install:**

```bash
git clone --recursive https://github.com/Alvin0523/vlash-piper.git
cd vlash-piper
pixi install
```

Already cloned without `--recursive`? Run `git submodule update --init --recursive` first.

👉 For hardware setup, data collection, training, inference, and Pixi task reference — see the **[Operation Guide](https://alvin0523.github.io/vlash-piper/operation_guide/)**.

---

## 🎬 Demo

<video src="assets/demo.mp4" controls width="100%"></video>

> π0.5 deployed on AgileX PIPER via async VLASH inference on Jetson AGX Orin.

---

## 📚 Citation

If this workspace is useful to you, please cite the underlying frameworks:

**VLASH** (async VLA inference framework):

```bibtex
@article{tang2025vlash,
  title   = {VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference},
  author  = {Tang, Jiaming and Sun, Yufei and Zhao, Yilong and Yang, Shang and
             Lin, Yujun and Zhang, Zhuoyang and Hou, James and Lu, Yao and Liu, Zhijian and others},
  journal = {arXiv preprint arXiv:2512.01031},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.01031}
}
```

**LeRobot** (robot learning framework):

```bibtex
@misc{cadene2024lerobot,
  author       = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and
                  Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and
                  Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and
                  Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
  title        = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished = {\url{https://github.com/huggingface/lerobot}},
  year         = {2024}
}
```

---

## 👥 Authors

**HKUST COMP4901-D Embedded AI — Course Project**

[<img src="https://github.com/Alvin0523.png" width="80" style="border-radius:50%">](https://github.com/Alvin0523)
[<img src="https://github.com/frieddeli.png" width="80" style="border-radius:50%">](https://github.com/frieddeli)
[<img src="https://github.com/HappyEthan.png" width="80" style="border-radius:50%">](https://github.com/HappyEthan)

| Name | GitHub |
|---|---|
| Wong Wei Ming | [@Alvin0523](https://github.com/Alvin0523) |
| Shao Ying Zhan | [@frieddeli](https://github.com/frieddeli) |
| Chen Yusen | [@HappyEthan](https://github.com/HappyEthan) |
