# Depth Estimation in Computer Vision
### From First Principles to Production Deployment

> **Scope:** Basic intuition → mathematical foundations → model architectures → open/closed source
> landscape → industrial applications → practical code → production API → relevant links.

---

## Table of Contents

1. [What is Depth Estimation?](#1-what-is-depth-estimation)
2. [Why is it Hard?](#2-why-is-it-hard)
3. [Types of Depth Estimation](#3-types-of-depth-estimation)
4. [How Humans Perceive Depth — Visual Cues](#4-how-humans-perceive-depth--visual-cues)
5. [Classical (Non-DL) Approaches](#5-classical-non-dl-approaches)
6. [Deep Learning Approach — Architecture Evolution](#6-deep-learning-approach--architecture-evolution)
7. [Loss Functions Used in Training](#7-loss-functions-used-in-training)
8. [Training Datasets](#8-training-datasets)
9. [Open Source Models — In-Depth Guide](#9-open-source-models--in-depth-guide)
10. [Closed Source / Commercial Models](#10-closed-source--commercial-models)
11. [Model Comparison Table](#11-model-comparison-table)
12. [Building a Model from Scratch — Architecture Walkthrough](#12-building-a-model-from-scratch--architecture-walkthrough)
13. [Practical Code — Basic to Advanced](#13-practical-code--basic-to-advanced)
14. [The Production API — This Project](#14-the-production-api--this-project)
15. [Industrial Use Cases](#15-industrial-use-cases)
16. [Evaluation Metrics](#16-evaluation-metrics)
17. [Current Research Frontiers](#17-current-research-frontiers)
18. [All Relevant Links](#18-all-relevant-links)

---

## 1. What is Depth Estimation?

**Depth estimation** is the computer vision task of predicting how far each pixel in an image is
from the camera — transforming a flat 2D photograph into a 3D distance map called a **depth map**.

```
Input:  RGB image   (H × W × 3)
Output: Depth map   (H × W × 1)  ← one distance value per pixel
```

Think of it like X-ray vision: the model reads a photo and tells you "this coffee cup is
0.4 m away, that wall behind it is 3.2 m away".

A depth map is usually visualised as a heatmap where warm colours (red/yellow) = **close**
and cool colours (blue/purple) = **far**, or vice versa depending on the colormap.

---

## 2. Why is it Hard?

From a single image, depth is **geometrically ambiguous**. An object can appear large because
it is large and far away, or small and close-up. No unique 3D solution exists.

Consider this: a 10 cm coin held 20 cm from the camera looks the same size as a 50 cm plate
held 1 m away. Humans resolve this using **context** — we know roughly how big coins are.
Neural networks must learn the same priors from data.

Key challenges:
- **Scale ambiguity** — monocular images carry no absolute scale signal
- **Texture-less surfaces** — walls, floors have few depth cues
- **Transparent surfaces** — windows look through to another depth
- **Reflections** — mirrors show a virtual scene at a different depth
- **Moving objects** — video depth is complicated by optical flow
- **Domain shift** — indoor models fail outdoors and vice versa

---

## 3. Types of Depth Estimation

### 3.1 By sensor inputs

| Type | Inputs | Notes |
|---|---|---|
| **Monocular** | Single RGB camera | Hardest; no geometric constraint |
| **Stereo** | Two calibrated cameras | Well-constrained; standard in automotive |
| **Multi-view** | 3+ cameras or video | SfM / MVS pipelines |
| **RGB-D** | RGB + depth sensor (LiDAR, ToF) | Sensor fusion; completion tasks |

This project focuses on **monocular depth estimation** — one image in, depth map out.

### 3.2 By output type

**Relative depth estimation**
- Outputs an ordinal ranking: pixel A is closer than pixel B
- No real-world units; depth values are normalised or arbitrary
- Works zero-shot across many domains
- Models: Depth Anything V2, MiDaS, Marigold

**Absolute / Metric depth estimation**
- Outputs real distances in **metres** (or feet)
- Requires training on datasets with ground-truth scale (LiDAR, stereo)
- Often specialised per scene type (indoor vs outdoor)
- Models: ZoeDepth, UniDepth, Metric3D, Depth Pro

---

## 4. How Humans Perceive Depth — Visual Cues

Neural networks implicitly learn to exploit the same cues human vision uses:

| Cue | Description | Example |
|---|---|---|
| **Linear perspective** | Parallel lines converge to a vanishing point | Railway tracks converging |
| **Texture gradient** | Texture elements get smaller with distance | A cobblestone road |
| **Occlusion** | Near objects block far objects | A tree in front of a building |
| **Relative size** | Familiar object size implies distance | A person far away looks small |
| **Atmospheric haze** | Distant objects are hazier, bluer | Mountains in the background |
| **Shading & shadows** | Light direction implies 3D shape | A sphere vs a flat circle |
| **Defocus blur** | Bokeh from shallow depth of field | Portrait photography |

---

## 5. Classical (Non-DL) Approaches

Before neural networks, depth was estimated via geometry:

### 5.1 Stereo Matching
Two cameras separated by a known baseline. For each pixel in the left image, find the
matching pixel in the right image. The horizontal offset (disparity `d`) gives depth:

```
depth = (focal_length × baseline) / disparity
```

Algorithm: Semi-Global Matching (SGM), used in automotive LiDAR-free setups even today.

### 5.2 Structure from Motion (SfM)
A moving camera takes multiple photos of the same scene. Feature matches across frames
triangulate 3D points. Tools: COLMAP, VisualSFM.

### 5.3 Shape from Shading
Infer surface normals from shading under known lighting. Very sensitive to lighting assumptions.

### 5.4 Limitations of classical methods
- Stereo requires calibrated camera pairs
- SfM requires multiple views + motion
- Fail on textureless surfaces (can't match features)
- Cannot generalise across scenes

---

## 6. Deep Learning Approach — Architecture Evolution

### Generation 1: CNN Encoder–Decoder (2014–2018)

**Eigen et al. (2014)** — the first paper to use CNNs for monocular depth, using AlexNet
as encoder and a simple upsampling decoder. Trained on NYU Depth v2.

```
RGB image → AlexNet encoder → coarse depth → refinement network → depth map
```

**Key idea:** Deep networks can learn scene statistics from millions of images and resolve
the geometric ambiguity through learned priors.

### Generation 2: Skip connections + U-Net (2017–2020)

Skip connections from encoder to decoder preserve fine spatial details lost during
downsampling. This pattern (U-Net style) became standard.

```
             ┌────────────────────────────────────────┐
Input RGB    │  E1 → E2 → E3 → E4 → bottleneck       │  Encoder
             │  ↕    ↕    ↕    ↕                      │  (skip connections)
             │  D4 ← D3 ← D2 ← D1 ← bottleneck       │  Decoder
Output depth └────────────────────────────────────────┘
```

### Generation 3: Dense Prediction Transformer — DPT (2021)

Intel's **DPT** (Vision Transformers for Dense Prediction) replaced the CNN backbone with
a **Vision Transformer (ViT)**. ViTs capture long-range dependencies — understanding that
the floor connects the near foreground to the far background.

Key innovation: fuse features from **multiple ViT layers** (not just the last):

```
ViT patch embeddings:
Layer 3  →─────────────────────────────────────┐
Layer 6  →──────────────────────────┐          │
Layer 9  →──────────────┐          │          │
Layer 12 →──┐          │          │          │
            ↓          ↓          ↓          ↓
          Reassemble  Reassemble  Reassemble  Reassemble
                 └────────┬────────┘
                     Fusion block
                          ↓
                    Dense depth map
```

This multi-scale fusion is why DPT produces sharp depth boundaries.

### Generation 4: Self-supervised + Foundation Models (2022–2024)

**MiDaS v3.1** trained on 12 diverse datasets mixing stereo, LiDAR, and web images,
achieving cross-domain generalisation.

**Depth Anything V2** used 62,000 labelled images + 320,000 synthetic images with
precise labels to train a ViT-L DPT that generalises zero-shot to any image.

**Marigold** used Stable Diffusion as a backbone — leveraging the rich visual priors
learned by LDMs to produce extremely fine-grained depth maps.

### Generation 5: Metric + Camera-Agnostic (2023–present)

**UniDepth** and **Metric3D v2** explicitly model camera intrinsics, allowing them to
produce metric depth across cameras without recalibration.

---

## 7. Loss Functions Used in Training

### 7.1 Scale-Invariant Log Loss (SILog) — standard for relative depth

Proposed by Eigen et al. (2014). Handles scale ambiguity by measuring error relative to
the scene's overall scale:

```
SILog = (1/n) Σ(d_i)² - (λ/n²)(Σ d_i)²

where d_i = log(predicted_i) - log(ground_truth_i)
      λ ≈ 0.85 (scale-invariance weight)
      n = number of valid pixels
```

### 7.2 Affine-Invariant Loss — for relative-only models

For relative depth models, both scale and shift are unknown. The loss aligns predictions
by least-squares scale+shift before computing error:

```
(s*, t*) = argmin_s,t Σ (s * pred_i + t - gt_i)²
loss = Σ (s* * pred_i + t* - gt_i)²
```

### 7.3 BerHu Loss (reverse Huber)

Used by many modern models. Behaves like L1 for small errors (sparse depth from LiDAR
is noisy) and L2 for large errors (penalise big mistakes):

```
BerHu(x) = |x|             if |x| ≤ c
           (x² + c²) / 2c  if |x| > c
where c = 0.2 * max(|x|)
```

### 7.4 Edge-aware smoothness loss

Prevents blurry depth boundaries. Penalises depth gradients that are not aligned with
image gradients (edges should coincide):

```
L_smooth = |∂d/∂x| * exp(-|∂I/∂x|) + |∂d/∂y| * exp(-|∂I/∂y|)
```

### 7.5 Combined training objective

Production models combine multiple losses:

```
L_total = α * L_SILog + β * L_smooth + γ * L_feature_matching
```

---

## 8. Training Datasets

| Dataset | Images | Depth Source | Scene Type |
|---|---|---|---|
| **NYU Depth v2** | 1,449 labelled | Kinect RGB-D | Indoor |
| **KITTI** | 93k frames | LiDAR Velodyne | Outdoor driving |
| **SUN RGB-D** | 10,335 | Kinect / RealSense | Indoor |
| **MegaDepth** | 1M | SfM from web photos | Landmarks/outdoor |
| **DIML** | 2.3M | RGB-D sensors | Mixed |
| **Hypersim** | 77k | Synthetic (ray-traced) | Indoor synthetic |
| **Virtual KITTI 2** | 21k | Synthetic | Driving synthetic |
| **BlendedMVS** | 17k | MVS reconstruction | Outdoor |
| **Depth Anything V2 synthetic** | 320k | Perfect synthetic labels | Mixed synthetic |

---

## 9. Open Source Models — In-Depth Guide

### 9.1 MiDaS (Intel — 2019, updated 2022)

**What it is:** The first widely-adopted generalisation model. Trained on 12 heterogeneous
datasets using a scale-and-shift invariant loss, enabling cross-domain deployment.

**Architecture:** DPT with ViT-L21 backbone (MiDaS v3.1). Smaller variants use EfficientNet.

**Output:** Relative depth (inverse depth, no units).

**Strength:** Runs in real-time on CPU with small model, very stable.

```python
# Quick start
import torch
model_type = "DPT_Large"  # or "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform
input_batch = transform(img)  # img is numpy HxWx3 uint8
with torch.no_grad():
    prediction = midas(input_batch)
```

**HuggingFace:** `Intel/dpt-large`, `Intel/dpt-hybrid-midas`
**Paper:** https://arxiv.org/abs/1907.01341

---

### 9.2 Depth Anything V2 (2024) ⭐ Recommended for relative

**What it is:** State-of-the-art zero-shot relative depth. Trains a ViT-L DPT backbone on
a massive mix of real and synthetic data. The V2 improvement: replacing pseudo-labels from
V1 (which were sometimes inaccurate) with labels from synthetic data with perfect ground truth.

**Architecture:**
```
Input RGB (H×W×3)
    ↓
ViT-L backbone (DINOv2 pretrained)
    ↓
DPT decoder (multi-scale feature fusion)
    ↓
Dense depth map (H×W×1), normalised
```

**Output:** Relative depth, normalised. Outputs are robust to any scene type.

**Variants:**
| Variant | Params | Speed | Use |
|---|---|---|---|
| Small | 24M | Fastest | Mobile / edge |
| Base | 97M | Balanced | Default |
| Large | 335M | Best quality | Production |

```python
from transformers import pipeline
pipe = pipeline("depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device=0)
result = pipe(image)
depth_map = result["predicted_depth"]   # torch.Tensor (H, W)
```

**HuggingFace:** `depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf`
**Paper:** https://arxiv.org/abs/2406.09414
**GitHub:** https://github.com/DepthAnything/Depth-Anything-V2

---

### 9.3 ZoeDepth (Microsoft — 2023) ⭐ Recommended for metric

**What it is:** Adds a metric head on top of the MiDaS relative encoder. The BinsFormer
head discretises depth into adaptive bins per image (not fixed global bins), making it
robust across indoor and outdoor scenes.

**Architecture:**
```
Input RGB
    ↓
MiDaS ViT-L encoder  ← frozen/fine-tuned relative depth backbone
    ↓
BinsFormer head:
    - Transformer predicts N adaptive depth bins per image
    - MLP per-pixel predicts bin probabilities
    - Depth = Σ bin_centres × bin_probs  (soft bin assignment)
    ↓
Metric depth map (metres)
```

**Key insight:** The relative encoder gives excellent features. The BinsFormer head only
needs to learn the affine mapping from relative to absolute scale — which requires very
little labelled metric data.

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

depth_metres = processor.post_process_depth_estimation(
    outputs,
    source_sizes=[(image.height, image.width)]
)[0]["predicted_depth"]
```

**HuggingFace:** `Intel/zoedepth-nyu`, `Intel/zoedepth-kitti`, `Intel/zoedepth-nyu-kitti`
**Paper:** https://arxiv.org/abs/2302.12288
**GitHub:** https://github.com/isl-org/ZoeDepth

---

### 9.4 Marigold (ETH Zürich — 2024)

**What it is:** Uses Stable Diffusion v2 as a backbone — the rich visual priors learned
from billions of images make it exceptional at fine-grained detail and sharp object boundaries.

**Architecture (Latent Diffusion for Depth):**
```
Input RGB
    ↓
VAE Encoder → latent z_image
    ↓
Concatenate [z_image, z_noise_t]   (condition on image)
    ↓
UNet denoising steps (T=10 default, fast inference)
    ↓
VAE Decoder → depth map
```

The denoising process iteratively refines the depth prediction, guided by the image.

**Strength:** Best edge preservation, best on images with fine structures (hair, foliage).
**Weakness:** Slower than discriminative models (diffusion requires multiple forward passes).

```python
from diffusers import MarigoldDepthPipeline

pipe = MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0",   # LCM = faster 4-step version
    torch_dtype=torch.float16,
).to("cuda")

depth_output = pipe(image)
depth_map = depth_output.prediction[0, 0]  # numpy (H, W), normalised
```

**HuggingFace:** `prs-eth/marigold-depth-v1-0`, `prs-eth/marigold-depth-lcm-v1-0`
**Paper:** https://arxiv.org/abs/2312.02145
**GitHub:** https://github.com/prs-eth/Marigold

---

### 9.5 UniDepth (2024)

**What it is:** Jointly estimates depth AND camera intrinsics (focal length, principal point)
from a single image. This makes it truly camera-agnostic — no calibration file needed.

**Architecture:**
```
Input RGB
    ↓
ViT encoder (DINOv2 pretrained)
    ↓
Camera decoder   → predicts focal length, principal point
Depth decoder    → conditioned on predicted camera params → metric depth
```

**Strength:** Works across phones, DSLRs, drones, dashcams without recalibration.

**HuggingFace:** `lpiccinelli-eth/unidepth-v2-vitl14`
**Paper:** https://arxiv.org/abs/2403.18913
**GitHub:** https://github.com/lpiccinelli-eth/UniDepth

---

### 9.6 Metric3D v2 (Tencent — 2024)

**What it is:** A ViT-Giant model trained on 16 million images with GT depth across 8
different camera types. Achieves SOTA on zero-shot metric depth benchmarks.

**Key innovation:** Camera model canonicalisation — all training images are reprocessed
into a canonical camera space so the model sees consistent geometry regardless of source camera.

**HuggingFace:** `JUGGHM/Metric3D`
**Paper:** https://arxiv.org/abs/2404.15506
**GitHub:** https://github.com/YvanYin/Metric3D

---

## 10. Closed Source / Commercial Models

### 10.1 Apple Depth Pro (2024)

Apple's model focuses on **absolute depth without known camera intrinsics** and produces
**4-megapixel resolution depth maps** — 4× the resolution of most competing models.

Architecture highlights:
- Multi-scale ViT with image patches at multiple resolutions
- Separate focal length estimation network
- Two-stage: coarse global depth → fine-grained detail refinement

Performance: Near real-time on Apple Silicon (Neural Engine).
GitHub (inference only): https://github.com/apple/ml-depth-pro

---

### 10.2 Google Cloud Vision / Vertex AI

Google's cloud APIs include scene understanding features with partial depth information,
primarily for AR and robotics within Google's ecosystem.

---

### 10.3 AWS Rekognition + custom SageMaker

AWS does not expose depth estimation directly, but teams run custom ZoeDepth / Depth
Anything deployments on SageMaker GPU instances for enterprise use.

---

### 10.4 Azure AI Vision (Spatial Analysis)

Microsoft Azure offers spatial analysis for people counting and distance estimation in
surveillance, using depth-aware models internally.

---

## 11. Model Comparison Table

| Model | Year | Type | Arch | SOTA? | Params | Speed (A100) | Units | Zero-shot |
|---|---|---|---|---|---|---|---|---|
| MiDaS v3.1 | 2022 | Relative | ViT-L | No | 307M | ~50ms | None | ✅ |
| Depth Anything V2 Small | 2024 | Relative | ViT-S DPT | ✅ | 24M | ~10ms | None | ✅ |
| Depth Anything V2 Large | 2024 | Relative | ViT-L DPT | ✅ | 335M | ~80ms | None | ✅ |
| Marigold LCM | 2024 | Relative | SD UNet | ✅ edges | 865M | ~500ms | None | ✅ |
| ZoeDepth NYU | 2023 | Metric | MiDaS+BinsF | ✅ indoor | 345M | ~100ms | metres | Partial |
| ZoeDepth KITTI | 2023 | Metric | MiDaS+BinsF | ✅ outdoor | 345M | ~100ms | metres | Partial |
| UniDepth v2 | 2024 | Metric | ViT-L | ✅ | 335M | ~90ms | metres | ✅ |
| Metric3D v2 | 2024 | Metric | ViT-Giant | ✅ SOTA | 2.4B | ~300ms | metres | ✅ |
| Depth Pro | 2024 | Metric | Multi-scale ViT | ✅ | ~300M | ~600ms | metres | ✅ |

---

## 12. Building a Model from Scratch — Architecture Walkthrough

Here is a complete minimal depth estimation model in PyTorch, demonstrating all
the core components from patch embeddings through to the depth head:

```python
"""
minimal_depth_model.py
Simplified DPT-style depth estimation model in pure PyTorch.
Demonstrates: patch embedding → transformer → reassemble → depth head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Patch Embedding (ViT-style input)
# ─────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """
    Splits an image into non-overlapping patches and projects each to D dims.
    A 384×384 image with patch_size=16 → (384/16)² = 576 tokens.
    """
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size  = patch_size
        # Single conv layer does the splitting + linear projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)           # (B, D, H/P, W/P)
        x = x.flatten(2)           # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


# ─────────────────────────────────────────────
# 2. Multi-Head Self-Attention
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x + residual


# ─────────────────────────────────────────────
# 3. MLP Feed-Forward Block
# ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# ─────────────────────────────────────────────
# 4. Transformer Block = MHSA + MLP
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp  = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


# ─────────────────────────────────────────────
# 5. Reassemble Block (DPT idea)
# Converts ViT token sequence back to spatial feature map
# ─────────────────────────────────────────────
class Reassemble(nn.Module):
    """
    DPT Reassemble: takes (B, N+1, D) ViT output → (B, C, H_out, W_out)
    by:
      1. Removing CLS token
      2. Reshaping N tokens to spatial grid
      3. 1×1 conv to reduce channels
      4. Upsample or keep stride
    """
    def __init__(self, embed_dim=768, out_channels=256,
                 patch_size=16, img_size=384, upsample_factor=1):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.up   = nn.Upsample(
            scale_factor=upsample_factor, mode="bilinear", align_corners=True
        ) if upsample_factor > 1 else nn.Identity()

    def forward(self, x):
        # x: (B, N+1, D) — +1 for CLS token
        x = x[:, 1:, :]                                    # drop CLS: (B, N, D)
        B, N, D = x.shape
        G = self.grid_size
        x = x.transpose(1, 2).view(B, D, G, G)            # (B, D, G, G)
        x = self.proj(x)                                   # (B, C, G, G)
        x = self.up(x)                                     # (B, C, G*s, G*s)
        return x


# ─────────────────────────────────────────────
# 6. Fusion Block (DPT decoder)
# ─────────────────────────────────────────────
class FusionBlock(nn.Module):
    """Merges two feature maps (residual fusion from DPT)."""
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act   = nn.ReLU(inplace=True)
        self.up    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + skip                    # residual from earlier layer
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.up(x)                      # upsample × 2
        return x


# ─────────────────────────────────────────────
# 7. Depth Head (final prediction)
# ─────────────────────────────────────────────
class DepthHead(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),               # depth ≥ 0
        )

    def forward(self, x):
        return self.net(x)           # (B, 1, H, W)


# ─────────────────────────────────────────────
# 8. Full Minimal DPT Depth Model
# ─────────────────────────────────────────────
class MinimalDPTDepth(nn.Module):
    """
    Stripped-down DPT depth model showing all components.
    For production, use pretrained Depth Anything V2 instead.

    Input:  (B, 3, 384, 384)
    Output: (B, 1, 384, 384)  — depth map
    """
    def __init__(self, img_size=384, patch_size=16, embed_dim=768,
                 num_heads=12, depth=12, out_channels=256):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)

        # Position embedding (learnable, for N patches + CLS token)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 12 transformer blocks; we tap layers 3, 6, 9, 12 for DPT
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # 4 reassemble blocks at different scales
        self.reassemble = nn.ModuleList([
            Reassemble(embed_dim, out_channels, patch_size, img_size, upsample_factor=4),
            Reassemble(embed_dim, out_channels, patch_size, img_size, upsample_factor=2),
            Reassemble(embed_dim, out_channels, patch_size, img_size, upsample_factor=1),
            Reassemble(embed_dim, out_channels, patch_size, img_size, upsample_factor=1),
        ])

        # 4 fusion blocks (decoder)
        self.fusion = nn.ModuleList([
            FusionBlock(out_channels) for _ in range(4)
        ])

        self.depth_head = DepthHead(out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embed(x)   # (B, N, D)

        # 2. Prepend CLS token + add position embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        # 3. Run transformer blocks, collect features at layers 3,6,9,12
        hook_layers = {2, 5, 8, 11}
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in hook_layers:
                features.append(x)

        # 4. Reassemble all 4 feature sets back to spatial maps
        spatial = [
            self.reassemble[j](features[j])
            for j in range(4)
        ]

        # 5. DPT decoder: fuse from deep → shallow
        fused = self.fusion[3](spatial[3])
        fused = self.fusion[2](fused, spatial[2])
        fused = self.fusion[1](fused, spatial[1])
        fused = self.fusion[0](fused, spatial[0])

        # 6. Predict depth
        depth = self.depth_head(fused)  # (B, 1, H, W)
        return depth


# Quick test
if __name__ == "__main__":
    model = MinimalDPTDepth()
    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # Expected: (2, 1, 384, 384)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

---

## 13. Practical Code — Basic to Advanced

### Level 1 — Basic: Depth Anything V2 in 10 lines

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open("photo.jpg")
result = pipe(image)
result["depth"].save("depth_map.png")   # saves colourised depth
```

---

### Level 2 — Intermediate: Manual inference + statistics

```python
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(device)
model.eval()

image = Image.open("room.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs.pixel_values.to(device)

with torch.no_grad():
    outputs = model(pixel_values)

depth = processor.post_process_depth_estimation(
    outputs, source_sizes=[(image.height, image.width)]
)[0]["predicted_depth"].numpy()

print(f"Depth range: {depth.min():.2f} – {depth.max():.2f} m")
print(f"Mean depth:  {depth.mean():.2f} m")

# Find closest and farthest pixels
y_near, x_near = np.unravel_index(depth.argmin(), depth.shape)
y_far,  x_far  = np.unravel_index(depth.argmax(), depth.shape)
print(f"Nearest point: ({x_near}, {y_near}) at {depth.min():.2f} m")
print(f"Farthest point: ({x_far}, {y_far}) at {depth.max():.2f} m")
```

---

### Level 3 — Advanced: Depth-guided background blur (bokeh effect)

```python
"""
Depth-guided portrait bokeh:
  - Estimate depth with Depth Anything V2
  - Create a foreground mask from depth threshold
  - Apply Gaussian blur to background pixels only
"""

import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

def depth_guided_bokeh(image_path: str, blur_strength: int = 51,
                        subject_depth_threshold: float = 0.4) -> np.ndarray:
    """
    Args:
        subject_depth_threshold: pixels with relative depth < this are 'subject' (near).
    """
    pipe = pipeline("depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if torch.cuda.is_available() else -1)

    pil_image = Image.open(image_path).convert("RGB")
    result = pipe(pil_image)

    depth_np = result["predicted_depth"].squeeze().numpy().astype(np.float32)
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    depth_resized = cv2.resize(depth_norm, (pil_image.width, pil_image.height))

    # Foreground mask (subject = close = low depth value in relative mode)
    fg_mask = (depth_resized < subject_depth_threshold).astype(np.float32)

    # Smooth mask edges to avoid hard transitions
    fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)
    fg_mask_3c = np.stack([fg_mask] * 3, axis=2)

    img_np = np.array(pil_image).astype(np.float32)

    # Blurred version (background)
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    blurred = cv2.GaussianBlur(img_np, (k, k), 0)

    # Blend: fg_mask=1 → sharp, fg_mask=0 → blurred
    composite = fg_mask_3c * img_np + (1 - fg_mask_3c) * blurred
    return composite.astype(np.uint8)

result_img = depth_guided_bokeh("portrait.jpg")
Image.fromarray(result_img).save("bokeh_output.jpg")
```

---

### Level 4 — Advanced: 3D point cloud from metric depth

```python
"""
Reconstruct a 3D point cloud from metric depth + camera intrinsics.
Requires ZoeDepth (metric) and known or estimated focal length.
"""

import numpy as np
import open3d as o3d
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def depth_to_pointcloud(rgb_image: Image.Image, depth_metres: np.ndarray,
                          fx: float, fy: float, cx: float, cy: float) -> o3d.geometry.PointCloud:
    """
    Args:
        fx, fy: focal lengths in pixels
        cx, cy: principal point in pixels
    """
    H, W = depth_metres.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Back-project pixel (u,v,d) → 3D (X,Y,Z)
    Z = depth_metres
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Stack into N×3 array, remove invalid depths
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colours = np.array(rgb_image).reshape(-1, 3) / 255.0

    valid = Z.reshape(-1) > 0
    points  = points[valid]
    colours = colours[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    return pcd


# Run ZoeDepth for metric depth
processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
model.eval()

img = Image.open("scene.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
depth = processor.post_process_depth_estimation(
    outputs, source_sizes=[(img.height, img.width)]
)[0]["predicted_depth"].numpy()

# Typical iPhone 13 main camera intrinsics at 1920×1440
fx = fy = 1540.0
cx, cy = img.width / 2, img.height / 2

pcd = depth_to_pointcloud(img, depth, fx, fy, cx, cy)
o3d.io.write_point_cloud("scene.ply", pcd)
o3d.visualization.draw_geometries([pcd])
```

---

### Level 5 — Expert: Fine-tuning Depth Anything V2 on custom data

```python
"""
Fine-tune Depth Anything V2 on a custom indoor dataset with SILog loss.
Assumes your dataset returns (image_tensor, depth_tensor) pairs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForDepthEstimation, AutoImageProcessor


# ── SILog Loss ─────────────────────────────────────────────────
class SILogLoss(nn.Module):
    def __init__(self, lambda_=0.85, eps=1e-6):
        super().__init__()
        self.lambda_ = lambda_
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = target > self.eps
        d = torch.log(pred.clamp(min=self.eps)) - torch.log(target.clamp(min=self.eps))
        d = d[valid_mask]
        n = d.numel()
        if n == 0:
            return torch.tensor(0.0, requires_grad=True)
        loss = (d ** 2).mean() - self.lambda_ * (d.sum() ** 2) / (n ** 2)
        return loss


# ── Training loop ──────────────────────────────────────────────
def fine_tune_depth_model(
    train_loader: DataLoader,
    checkpoint: str = "depth-anything/Depth-Anything-V2-Small-hf",
    num_epochs: int = 10,
    lr: float = 1e-5,
    device: str = "cuda",
):
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)
    processor = AutoImageProcessor.from_pretrained(checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = SILogLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, depth_gt) in enumerate(train_loader):
            # images: (B, 3, H, W), depth_gt: (B, H, W) in metres
            images   = images.to(device)
            depth_gt = depth_gt.to(device)

            # Forward pass
            outputs = model(pixel_values=images)
            pred_depth = outputs.predicted_depth   # (B, H, W)

            # Resize prediction to match GT if needed
            if pred_depth.shape != depth_gt.shape:
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth.unsqueeze(1),
                    size=depth_gt.shape[-2:],
                    mode="bilinear", align_corners=False
                ).squeeze(1)

            loss = criterion(pred_depth, depth_gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save fine-tuned model
    model.save_pretrained("./fine_tuned_depth_model")
    processor.save_pretrained("./fine_tuned_depth_model")
    print("Model saved to ./fine_tuned_depth_model")
```

---

## 14. The Production API — This Project

### Project structure

```
depth_estimation_project/
├── app/
│   ├── main.py              ← FastAPI app factory + lifespan
│   ├── config.py            ← Pydantic settings (env-driven)
│   ├── schemas.py           ← Request/response models
│   ├── models/
│   │   └── depth_model.py   ← DepthBackend protocol + registry
│   ├── routes/
│   │   ├── depth.py         ← /relative, /metric endpoints
│   │   └── health.py        ← /health endpoint
│   └── utils/
│       └── image.py         ← decode, resize, encode, stats
├── scripts/
│   └── batch_infer.py       ← CLI batch processing tool
├── notebooks/
│   └── standalone_demo.py   ← Zero-setup local demo
├── tests/
│   └── test_api.py          ← pytest suite (unit + integration)
├── configs/
│   └── nginx.conf           ← Reverse proxy config
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

### Quick start

```bash
# 1. Clone and install
git clone <repo>
cd depth_estimation_project
pip install -r requirements.txt

# 2. Start the API
uvicorn app.main:app --reload --port 8000

# 3. Test with curl
curl -X POST http://localhost:8000/api/v1/depth/relative \
     -F "file=@photo.jpg" \
     --output depth.png

# 4. Get JSON response with stats
curl -X POST "http://localhost:8000/api/v1/depth/metric?response_format=json" \
     -F "file=@room.jpg"

# 5. Run tests
pytest tests/ -v

# 6. Docker deployment
docker compose up -d
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/depth/relative` | Relative depth (Depth Anything V2) |
| `POST` | `/api/v1/depth/metric` | Metric depth in metres (ZoeDepth) |
| `GET` | `/api/v1/depth/models` | List loaded models + metadata |
| `GET` | `/api/v1/health` | Liveness check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

### Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `RELATIVE_MODEL_CHECKPOINT` | `Depth-Anything-V2-Small-hf` | HF model ID |
| `METRIC_MODEL_CHECKPOINT` | `Intel/zoedepth-nyu-kitti` | HF model ID |
| `MAX_IMAGE_SIZE` | `1920` | Longest edge before downscaling |
| `MAX_UPLOAD_MB` | `20` | Max upload size |
| `COLORMAP` | `inferno` | Matplotlib colormap for PNG output |
| `INFERENCE_TIMEOUT_SECONDS` | `60` | Per-request timeout |

---

## 15. Industrial Use Cases

### Autonomous Driving
LiDAR is expensive ($10k+). Camera-based depth reduces cost by 90%.
Models like Metric3D run on dashcam feeds to detect vehicles, pedestrians,
and road surfaces at real-world distances. Companies: Waymo, Tesla, Mobileye.

### Augmented Reality (AR)
Depth estimation enables realistic object placement — a virtual chair appears
behind a real table rather than floating over it. Apple ARKit uses depth from
LiDAR on newer iPhones, but camera-only depth powers older devices and Android.
Companies: Meta Quest, Snap, Niantic.

### Robotics & Manipulation
Robots need to know how far objects are to grasp them. Depth Anything V2 on
a Jetson Orin provides real-time depth for pick-and-place tasks.
Companies: Boston Dynamics, Agility Robotics, Amazon Robotics.

### Medical Imaging (Endoscopy)
Monocular depth from endoscope cameras helps surgeons measure tissue distances
without calibration rigs. ZoeDepth-style models fine-tuned on surgical data.
Research groups: Imperial College London, Stanford AI in Medicine.

### E-commerce — "Try Before You Buy"
AR furniture placement (IKEA Place), virtual try-on (Warby Parker glasses) all
need depth to blend virtual objects into the real world realistically.

### Film VFX
Depth maps from production cameras are used to add atmospheric fog, separate
foreground actors for compositing, and guide AI-based background generation.
Studios: ILM, Weta Digital.

### Agriculture & Drones
Crop height estimation, disease mapping, and yield prediction use depth from
drone cameras flying at known altitudes (helps calibrate scale).

### Security & Surveillance
People counting, fall detection, and intrusion detection systems use depth to
identify when a person is too close or has fallen, even in poor lighting.

---

## 16. Evaluation Metrics

When benchmarking depth models, these standard metrics are used:

### AbsRel — Absolute Relative Error
```
AbsRel = (1/N) Σ |pred_i - gt_i| / gt_i
```
Lower is better. Measures average relative error per pixel.

### SqRel — Squared Relative Error
```
SqRel = (1/N) Σ |pred_i - gt_i|² / gt_i
```
Penalises large errors more heavily.

### RMSE — Root Mean Square Error
```
RMSE = sqrt((1/N) Σ (pred_i - gt_i)²)
```
In the same units as depth (e.g. metres for metric models).

### δ < 1.25 (threshold accuracy)
```
δ₁ = % of pixels where max(pred/gt, gt/pred) < 1.25
```
The fraction of pixels where prediction is within 25% of ground truth.
Higher is better. Also reported at 1.25² and 1.25³.

### log10 Error
```
log10 = (1/N) Σ |log10(pred_i) - log10(gt_i)|
```
Scale-independent error measure.

**KITTI benchmark (outdoor, driving) — state of the art:**
| Model | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Metric3D v2 | 0.052 | 2.03m | 97.4% |
| UniDepth v2 | 0.058 | 2.19m | 96.8% |
| ZoeDepth | 0.071 | 2.67m | 95.5% |

---

## 17. Current Research Frontiers

### Video depth estimation
Temporal consistency — depth should not flicker between frames.
Methods: ChronoDepth, Consistent Video Depth (CVD), NVDS.

### Foundation models for depth
DepthFM (flow-matching), Depth Pro (multi-scale ViT), and future GPT-4V-style
models that understand 3D geometry from language + images.

### Sparse → Dense depth completion
Given 5% of pixels from LiDAR, fill in the rest using the RGB image.
Models: NLSPN, CompletionFormer.

### Uncertainty estimation
Safety-critical applications need to know *when* the model is uncertain.
Bayesian approaches, ensembles, and evidential deep learning.

### Self-supervised depth + pose (no GT labels)
Monodepth2 and its successors train only on video by minimising photometric
reconstruction loss between adjacent frames.

### Depth in the wild (domain generalisation)
Models trained on driving fail on endoscopy. New work on domain-adaptive
and test-time adaptation for depth.

---

## 18. All Relevant Links

### Papers
| Paper | Link |
|---|---|
| Eigen et al. (2014) — First CNN depth | https://arxiv.org/abs/1406.2283 |
| MiDaS — Cross-dataset training | https://arxiv.org/abs/1907.01341 |
| DPT — Vision Transformers for Dense Prediction | https://arxiv.org/abs/2103.13413 |
| ZoeDepth — Metric depth via BinsFormer | https://arxiv.org/abs/2302.12288 |
| Depth Anything V1 | https://arxiv.org/abs/2401.10891 |
| Depth Anything V2 | https://arxiv.org/abs/2406.09414 |
| Marigold — Diffusion for depth | https://arxiv.org/abs/2312.02145 |
| UniDepth | https://arxiv.org/abs/2403.18913 |
| Metric3D v2 | https://arxiv.org/abs/2404.15506 |
| Depth Pro (Apple) | https://arxiv.org/abs/2410.02073 |
| Monodepth2 — Self-supervised | https://arxiv.org/abs/1806.01260 |
| BinsFormer | https://arxiv.org/abs/2204.00987 |

### GitHub Repositories
| Repo | Link |
|---|---|
| Depth Anything V2 | https://github.com/DepthAnything/Depth-Anything-V2 |
| ZoeDepth | https://github.com/isl-org/ZoeDepth |
| MiDaS | https://github.com/isl-org/MiDaS |
| Marigold | https://github.com/prs-eth/Marigold |
| UniDepth | https://github.com/lpiccinelli-eth/UniDepth |
| Metric3D | https://github.com/YvanYin/Metric3D |
| Depth Pro (Apple) | https://github.com/apple/ml-depth-pro |
| Monodepth2 | https://github.com/nianticlabs/monodepth2 |
| COLMAP (SfM) | https://github.com/colmap/colmap |

### HuggingFace Models
| Model | HF Link |
|---|---|
| Depth Anything V2 Small | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf |
| Depth Anything V2 Large | https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf |
| ZoeDepth NYU+KITTI | https://huggingface.co/Intel/zoedepth-nyu-kitti |
| Marigold | https://huggingface.co/prs-eth/marigold-depth-v1-0 |
| DPT Large | https://huggingface.co/Intel/dpt-large |
| UniDepth v2 | https://huggingface.co/lpiccinelli-eth/unidepth-v2-vitl14 |

### HuggingFace Datasets
| Dataset | Link |
|---|---|
| NYU Depth v2 | https://huggingface.co/datasets/sayakpaul/nyu_depth_v2 |
| KITTI Depth | https://www.cvlibs.net/datasets/kitti/eval_depth.php |
| SUN RGB-D | https://rgbd.cs.princeton.edu |
| Hypersim (synthetic) | https://github.com/apple/ml-hypersim |

### Learning Resources
| Resource | Link |
|---|---|
| HF Depth Estimation Task | https://huggingface.co/tasks/depth-estimation |
| HF Depth Estimation Guide | https://huggingface.co/docs/transformers/tasks/monocular_depth_estimation |
| Paperswithcode Depth | https://paperswithcode.com/task/monocular-depth-estimation |
| KITTI Leaderboard | https://www.cvlibs.net/datasets/kitti/eval_depth.php |
| Papers With Code NYU | https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2 |
| First Principles of CV (YouTube) | https://www.youtube.com/@firstprinciplesofcomputervision |

### Tools & Libraries
| Tool | Link |
|---|---|
| Open3D (point clouds) | http://www.open3d.org |
| PyTorch | https://pytorch.org |
| HuggingFace Transformers | https://github.com/huggingface/transformers |
| Diffusers (Marigold) | https://github.com/huggingface/diffusers |
| FastAPI | https://fastapi.tiangolo.com |
| OpenCV | https://opencv.org |

---

*Built with ❤️ using Depth Anything V2 + ZoeDepth + FastAPI*
