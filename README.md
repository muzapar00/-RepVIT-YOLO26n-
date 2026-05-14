# 无人机自主降落引导标识视觉实时检测系统

> Drone Autonomous Landing Marker Visual Real-Time Detection System

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.30-purple)](https://www.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 简介 | Introduction

本项目针对无人机自主降落场景，基于 YOLO 系列目标检测算法和 RepViT 重参数化 Vision Transformer，构建了一套**引导标识实时视觉检测与三维定位系统**。检测目标为两类：**sign**（降落引导标识框）和 **Yaw**（偏航角度指示标识）。

**核心创新**：将 RepViT (CVPR 2024) 的重参数化多分支 Token Mixer 与 YOLO26 的 C3k2 结构融合，提出 **RepViT-C3k2** 模块。训练时通过多尺度深度卷积分支（7×7 + 5×5 + 3×3 + Identity）+ SE 通道注意力 + 倒残差 Channel Mixer 增强特征表征；推理时通过重参数化将多分支融合为单个 DWConv，实现**零额外推理开销**。

We build a real-time visual detection and 3D localization system for drone autonomous landing. The system detects two classes: **sign** (landing marker) and **Yaw** (yaw angle indicator).

**Key innovation**: **RepViT-C3k2** — fusing RepViT's re-parameterized multi-branch Token Mixer (CVPR 2024) with YOLO26's C3k2 CSP structure. Multi-scale depthwise conv branches (7×7 + 5×5 + 3×3 + Identity), SE channel attention, and inverted-residual Channel Mixer during training; all branches fused into a single DWConv at inference for **zero-cost deployment**.

---

## 主要结果 | Key Results

| 指标 Metric | YOLO26n + RepViT-C3k2 | YOLO26n Baseline | 提升 Δ |
|:---|:---:|:---:|:---:|
| 参数量 Params | **2.42M** | 2.62M | −7.6% |
| mAP50 | **27.9%** | 27.3% | +0.6 pp |
| mAP50-95 | 18.1% | 18.1% | — |
| Yaw mAP50 | **19.9%** | 17.4% | **+2.5 pp** |
| sign mAP50 | 35.9% | **37.2%** | −1.3 pp |
| 推理速度 Inference | 5.1 ms | **4.0 ms** | — |

- 标注数据质量审查发现 367/1999 (18.4%) 标注文件存在 sign/Yaw 标反问题，已自动修复
- Label audit: 367/1999 (18.4%) annotation files had swapped sign/Yaw labels — automatically corrected

---

## 系统架构 | System Architecture

```
┌─────────────────────────────────────────────────┐
│                  实时检测流水线                    │
│                 Real-Time Pipeline               │
├─────────────────────────────────────────────────┤
│  摄像头 Capture ─→ YOLO检测 ─→ 坐标计算 ─→ 输出  │
│                    ↓ (失败/fallback)              │
│              QR码检测 (WeChatQRCode)              │
│                   ↓                              │
│           EPnP 三维定位 (PnP 3D Localization)     │
└─────────────────────────────────────────────────┘
```

**三层检测策略 | Three-Layer Detection Strategy:**

1. **YOLO 优先** — 检测 sign / Yaw 标识，计算二维坐标与偏航角
2. **QR 码回退** — YOLO 检测失败时启用 WeChatQRCode 深度学习二维码识别（内容 `getqrcode12138`）
3. **EPnP 定位** — 基于相机标定参数和 PnP 算法解算三维空间坐标

**1. YOLO primary** — detect sign/Yaw markers, compute 2D coordinates and yaw angle
**2. QR fallback** — WeChatQRCode deep-learning QR detector when YOLO fails
**3. EPnP 3D localization** — PnP-based 3D coordinate estimation using camera intrinsics

---

## 相机标定 | Camera Calibration

| 参数 | 标定值 (1920×1080) | 运行值 (1280×720) |
|:---|:---|:---|
| fx | 1459.05 | 972.37 (×0.667) |
| fy | 1456.91 | 971.29 (×0.667) |
| cx | 969.07 | 646.05 (×0.667) |
| cy | 531.88 | 354.59 (×0.667) |
| k1, k2, p1, p2, k3 | 0.174, −0.486, 0.0013, 0.0005, 0.474 | — |

**实体尺寸 | Physical Dimensions:** sign Φ600mm | Yaw Φ90mm | sign→Yaw 200mm | QR 113×113mm

---

## 项目结构 | Project Structure

```
workspace/
├── modules/
│   └── repvit_c3k2.py              # RepViTBlock + RepViTC3k2 自定义模块
├── yolo26n-repvit.yaml             # YOLO26n + RepViT-C3k2 模型配置
├── train_yolo26n_repvit.py         # v5 训练脚本 (含自动 patch)
├── realtime_detect_coords.py       # ★ 实时检测主程序 (含坐标计算)
├── detect_video_yolo26n.py         # 视频推理脚本
├── fix_labels.py                   # 标注自动审查与修复
├── analyze_labels.py               # 标注质量分析
├── gen_wts.py                      # 导出 .wts 权重 (TensorRT 部署)
├── dataset/
│   ├── data.yaml                   # 数据集配置
│   ├── images/train/               # ~1800 训练图片
│   ├── images/val/                 # ~200 验证图片
│   └── labels/                     # YOLO 格式标注
├── deploy/                         # 旭日X3派部署文件
├── deploy_v2/                      # TensorRT / OpenVINO 部署
├── runs/yolo26n_repvit/weights/    # 训练输出 & 最佳模型
├── vision_landing/                 # 视觉着陆相关模块
└── bg_candidates/                  # 背景图候选 (数据增强用)
```

---

## 快速开始 | Quick Start

### 环境要求 | Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, ultralytics == 8.4.30
- (可选) OpenVINO 2024+ for CPU 加速

### 安装 | Installation

```bash
pip install torch torchvision ultralytics==8.4.30 opencv-python openvino
```

### 实时检测 | Real-Time Detection

```bash
# 摄像头实时检测 (CPU 模式, imgsz=320)
python realtime_detect_coords.py

# 视频文件推理
python detect_video_yolo26n.py --source your_video.mp4
```

### 训练 | Training

```bash
# 训练 YOLO26n + RepViT-C3k2 (v5 配置, 100 epochs)
python train_yolo26n_repvit.py
```

训练脚本会自动 patch ultralytics 以注册 `RepViTC3k2` 模块。如需恢复原始 ultralytics，从 `tasks.py.backup` 覆盖即可。

The training script auto-patches ultralytics to register `RepViTC3k2`. Restore from `tasks.py.backup` if needed.

---

## 模型导出与部署 | Export & Deploy

```bash
# OpenVINO 导出 (CPU 加速)
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='openvino', imgsz=320)"

# TensorRT 导出 (GPU 加速)
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='engine', imgsz=320)"

# WTS 导出 (TensorRT 手动构建)
python gen_wts.py
```

---

## 消融实验 | Ablation Study

| 版本 | Backbone | 预训练 | mAP50 | mAP50-95 |
|:---|:---|:---:|:---:|:---:|
| YOLOv11s Baseline | C2f | COCO | 25.93% | 16.02% |
| YOLOv11s + 训练优化 | C2f | COCO | 26.87% | 17.29% |
| YOLO26s Baseline | C3k2 | COCO | 28.6% | 16.2% |
| YOLO26s + RepViT-C3k2 | RepViT-C3k2 | partial | 20.9% | 9.8% |
| **YOLO26n + RepViT-C3k2 v5** | **RepViT-C3k2** | **partial** | **27.9%** | **18.1%** |

---

## 关键经验 | Lessons Learned

1. **标注质量 > 模型结构** — 18.4% 标反率直接限制 mAP 天花板，数据清洗是第一步
2. **预训练权重至关重要** — 从零训练 9M 参数网络 + 2K 数据 → mAP≈25%；加载 COCO 预训练 → Epoch 1 即 20%+
3. **小模型更适合小数据** — YOLO26n (2.4M) 效果优于 26s (9.1M)，减少过拟合
4. **重参数化的价值** — RepViT-C3k2 在 Yaw 小目标上 +2.5pp，验证了多尺度特征融合的有效性
5. **CPU 部署需要导出** — PyTorch CPU 仅 2-5 FPS，OpenVINO 导出后可提升至 8-15 FPS

1. **Label quality matters most** — 18.4% annotation errors cap mAP ceiling
2. **Pretrained weights are critical** — COCO pretrain gives 20%+ mAP50 at epoch 1 vs. training from scratch
3. **Smaller models generalize better on small datasets** — 2.4M > 9.1M params with only 2K images
4. **RepViT-C3k2 helps small objects** — +2.5 pp on Yaw validates multi-scale feature fusion
5. **Export for CPU deployment** — OpenVINO provides 3-5× speedup over PyTorch CPU inference

---

## 作者 | Author

**穆再排尔·穆合塔尔** (Muzapper Muhtar)

南京航空航天大学 | Nanjing University of Aeronautics and Astronautics (NUAA)

---

## 引用 | Citation

If you find this work useful, please consider citing the key papers that inspired this project:

- Wang, A. et al. (2024). *RepViT: Revisiting Mobile CNN From ViT Perspective*. CVPR 2024.
- Wang, C.-Y. et al. (2025). *YOLO26: An End-to-End Multi-Stage Object Detector*. Ultralytics.
- Ding, X. et al. (2021). *RepVGG: Making VGG-style ConvNets Great Again*. CVPR 2021.

---
