# HybridFormerUNet – Brain Tumor Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Albumentations-1.3+-brightgreen" alt="Albumentations">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/last-commit/YOUR_USERNAME/hybridformer-unet?style=flat-square" alt="Last Commit">
</p>

Advanced U-Net variant combining **ResNeXt-style multi-branch convolutions**, **State Space Model (SSM) bottleneck** with bidirectional GRUs, **ASPP**, **dual attention gates**, and **deep supervision** for precise low-grade glioma (LGG) tumor segmentation in brain MRI.



## Architecture at a Glance

- **Encoder**: ResNeXt-style blocks (parallel 3×3 + 5×5 grouped convs)
- **Bottleneck**: ASPP → bidirectional GRU-based SSM Token Mixer → fusion conv
- **Skip connections**: Dual Attention Gates (Squeeze-Excitation + Spatial Attention)
- **Decoder**: Bilinear upsampling + deep supervision on 4 levels
- **Loss**: Hybrid CE + Dice + deep supervision auxiliary losses
- **Optimizer & schedule**: AdamW + CosineAnnealingWarmRestarts + mixed precision (AMP)



## Explainability Visuals

**Grad-CAM** (adapted for segmentation) + **Decoder-level spatial attention maps**


# 2. Run full pipeline (downloads dataset automatically via kagglehub)
python main.py

