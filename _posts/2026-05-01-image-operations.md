---
title: Image Operations - The Framework Behind Vision Models
description: Core image operations that form the framework between backbone architectures and specialized applications
date: 2024-10-12
categories:
  - Machine Learning
  - Computer Vision
  - Deep Learning
tags:
  - computer-vision
  - image-processing
  - feature-extraction
  - representation-learning
  - neural-networks
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panel9@4x.png
---

# Introduction

Between the raw backbone (CNN, ViT) and the specialized application (YOLO for detection, LoFTR for matching, ArcFace for recognition) lies a framework of image operations that every modern vision model relies on. These operations are neither the architecture itself nor the task-specific head—they are the connective tissue: how images are diagnosed, transformed, scaled, normalized, and combined.

Just as time series analysis has the Dickey-Fuller test for stationarity and ARMA diagnostic statistics, computer vision has its own diagnostic tests, contrast definitions, and quality metrics. Understanding these operations and their statistical foundations matters because they determine what every downstream task can achieve.

---

## 0. Image Representation Foundations

Before any operation, an image must be represented numerically.

### Pixel Space

A digital image is a tensor $I \in \mathbb{R}^{H \times W \times C}$:

- $H$: height (rows)
- $W$: width (columns)
- $C$: channels (typically 3 for RGB, 1 for grayscale)

Each pixel value is typically in $[0, 255]$ (uint8) or $[0, 1]$ (float32 after normalization).

### Color Spaces

| Color Space | Channels               | Properties                         | Use Case                     |
| ----------- | ---------------------- | ---------------------------------- | ---------------------------- |
| RGB         | Red, Green, Blue       | Additive, perceptually non-uniform | Default for neural networks  |
| HSV         | Hue, Saturation, Value | Decouples color from brightness    | Color-based filtering        |
| LAB         | Lightness, A, B        | Perceptually uniform               | Color matching, segmentation |
| YUV         | Luminance, Chrominance | Compression-friendly               | Video processing             |
| Grayscale   | Single intensity       | No color information               | Edge detection, OCR          |

> **Practical Tip:** Train on RGB unless task is color-invariant. For augmentation diversity, convert to HSV temporarily, modify hue/saturation, convert back.
> {: .prompt-tip }

### Input Normalization

Raw pixel values $[0, 255]$ are unsuitable for gradient descent. Standard normalization:

$$\hat{x} = \frac{x - \mu}{\sigma}$$

where $\mu, \sigma$ are dataset statistics (e.g., ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 1. Image Statistics and Diagnostic Tests

Just as time series has Dickey-Fuller for stationarity, computer vision has diagnostic tests for image quality, distribution, and properties. These tests determine whether an image is suitable for a task before any model is trained.

### 1.1 Contrast: Mathematical Definitions

Contrast measures how distinguishable elements are from their surroundings. Multiple formal definitions exist, each suited to different scenarios.

**Michelson Contrast** (for periodic patterns, e.g., gratings):

$$C_M = \frac{L_{max} - L_{min}}{L_{max} + L_{min}}$$

Range: $[0, 1]$. Used for sinusoidal patterns and visual perception studies.

**Weber Contrast** (for small features on uniform background):

$$C_W = \frac{L - L_b}{L_b}$$

where $L$ is feature luminance, $L_b$ is background luminance. Range: $[-1, \infty)$.

**RMS Contrast** (for natural images):

$$C_{RMS} = \sqrt{\frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W (I[i,j] - \bar{I})^2}$$

This is the standard deviation of pixel intensities. Most useful for general image analysis.

**Threshold Table for RMS Contrast (8-bit grayscale):**

| RMS Contrast | Interpretation | Visual Quality           | Recommended Action                   |
| ------------ | -------------- | ------------------------ | ------------------------------------ |
| < 10         | Very low       | Washed out, near uniform | Histogram stretching or equalization |
| 10 – 30      | Low            | Faded, low detail        | Contrast enhancement (CLAHE)         |
| 30 – 60      | Normal         | Good visibility          | Proceed with standard processing     |
| 60 – 90      | High           | Sharp, detailed          | Possibly over-saturated              |
| > 90         | Very high      | Harsh transitions        | Tone-mapping may be needed           |

### 1.2 Histogram-Based Statistics

The histogram $h(k)$ counts pixel occurrences at intensity level $k$. From this:

**Mean** (average brightness):
$$\mu = \frac{1}{HW}\sum_{i,j} I[i,j]$$

**Variance** (spread of intensities):
$$\sigma^2 = \frac{1}{HW}\sum_{i,j} (I[i,j] - \mu)^2$$

**Skewness** (asymmetry of distribution):
$$\gamma_1 = \frac{1}{HW \sigma^3}\sum_{i,j} (I[i,j] - \mu)^3$$

**Kurtosis** (peakedness):
$$\gamma_2 = \frac{1}{HW \sigma^4}\sum_{i,j} (I[i,j] - \mu)^4 - 3$$

**Entropy** (information content):
$$H = -\sum_{k=0}^{255} p(k) \log_2 p(k)$$

where $p(k) = h(k)/(HW)$ is the probability of intensity $k$.

**Diagnostic Threshold Table:**

| Metric                | Range              | Interpretation           | Action                             |
| --------------------- | ------------------ | ------------------------ | ---------------------------------- |
| Mean ($\mu$)          | < 50               | Underexposed             | Increase brightness or gamma > 1   |
| Mean ($\mu$)          | 100 – 150          | Well-exposed             | Standard processing                |
| Mean ($\mu$)          | > 200              | Overexposed              | Decrease exposure or gamma < 1     |
| Variance ($\sigma^2$) | < 500              | Low contrast             | Apply contrast enhancement         |
| Variance ($\sigma^2$) | 500 – 5000         | Normal                   | Standard pipeline                  |
| Skewness ($\gamma_1$) | $\|\gamma_1\| > 1$ | Asymmetric               | Consider gamma correction          |
| Kurtosis ($\gamma_2$) | > 3                | Heavy tails (saturation) | Tone-mapping needed                |
| Entropy ($H$)         | < 5 bits           | Low information          | Image may be mostly uniform        |
| Entropy ($H$)         | 6 – 7 bits         | Typical natural image    | Proceed normally                   |
| Entropy ($H$)         | > 7.5 bits         | High information / noise | Check for noise; denoise if needed |

### 1.3 Image Quality Metrics

When comparing two images (original vs. processed, or model output vs. ground truth):

**PSNR (Peak Signal-to-Noise Ratio):**

$$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$

where $\text{MAX}_I = 255$ for 8-bit, and $\text{MSE} = \frac{1}{HW}\sum (I_1 - I_2)^2$.

**SSIM (Structural Similarity Index):**

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

Range: $[-1, 1]$, where 1 = identical.

**LPIPS (Learned Perceptual Image Patch Similarity):**

$$\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l}\sum_{h,w} \|w_l \odot (\phi_l(x)_{hw} - \phi_l(y)_{hw})\|_2^2$$

where $\phi_l$ are features from a pretrained network (e.g., VGG).

**Quality Threshold Table:**

| Metric    | Excellent | Good        | Acceptable  | Poor        | Failure |
| --------- | --------- | ----------- | ----------- | ----------- | ------- |
| PSNR (dB) | > 40      | 30 – 40     | 25 – 30     | 20 – 25     | < 20    |
| SSIM      | > 0.95    | 0.85 – 0.95 | 0.70 – 0.85 | 0.50 – 0.70 | < 0.50  |
| LPIPS     | < 0.05    | 0.05 – 0.15 | 0.15 – 0.30 | 0.30 – 0.50 | > 0.50  |

### 1.4 Distribution Comparison Tests

To test if two image sets come from the same distribution (training vs. test, real vs. synthetic):

**Chi-Square Test on Histograms:**

$$\chi^2 = \sum_{k=0}^{255} \frac{(h_1(k) - h_2(k))^2}{h_1(k) + h_2(k) + \epsilon}$$

**Bhattacharyya Distance:**

$$D_B = -\ln\left(\sum_{k=0}^{255} \sqrt{p_1(k) p_2(k)}\right)$$

**Kolmogorov-Smirnov Test:**

$$D_{KS} = \max_k |F_1(k) - F_2(k)|$$

where $F$ is the cumulative distribution.

**Distribution Test Threshold Table:**

| Test                  | Same Distribution | Slight Shift | Major Shift |
| --------------------- | ----------------- | ------------ | ----------- |
| $\chi^2$ (normalized) | < 0.1             | 0.1 – 0.5    | > 0.5       |
| Bhattacharyya         | < 0.1             | 0.1 – 0.4    | > 0.4       |
| KS-statistic          | < 0.05            | 0.05 – 0.20  | > 0.20      |

> **Practical Tip:** Run distribution tests between training and validation sets before training. A KS statistic > 0.2 indicates dataset shift—your model will likely overfit to training distribution. Either rebalance data or use domain adaptation.
> {: .prompt-tip }

### 1.5 Spatial Autocorrelation (Vision Analog of Time-Series Autocorrelation)

Just as time series has $R_{xx}[n]$, images have spatial autocorrelation measuring self-similarity at different offsets.

**2D Autocorrelation:**

$$R[u, v] = \sum_{i,j} I[i,j] \cdot I[i+u, j+v]$$

**Moran's I (global spatial autocorrelation):**

$$I_M = \frac{N}{\sum_{i,j} w_{ij}} \cdot \frac{\sum_{i,j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$

Range: $[-1, 1]$ where:

- $I_M > 0$: Clustered (similar values nearby)
- $I_M = 0$: Random
- $I_M < 0$: Dispersed (alternating pattern)

**Moran's I Threshold Table:**

| Moran's I  | Interpretation       | Image Type                           |
| ---------- | -------------------- | ------------------------------------ |
| > 0.7      | Highly clustered     | Smooth gradients, large objects      |
| 0.3 – 0.7  | Moderately clustered | Natural images, textures             |
| -0.1 – 0.3 | Weak structure       | Noisy or random                      |
| < -0.1     | Anti-correlated      | Checkerboard patterns, fine textures |

---

## 2. Preprocessing and Postprocessing Transformations

Preprocessing transforms images before they enter the model. Postprocessing transforms model outputs. Both rely on transformations whose effects can be analyzed mathematically and visualized.

### 2.1 Histogram Equalization

Maps pixel intensities to spread the histogram uniformly across the range.

**Mathematical Definition:**

$$T(k) = \text{floor}\left((L-1) \cdot \text{CDF}(k)\right)$$

where $L = 256$ for 8-bit and $\text{CDF}(k) = \sum_{i=0}^{k} p(i)$.

**Effect on histogram:**

```
Before:  ▁▁▁▆█▇▅▂▁▁▁▁▁▁    Concentrated mid-range
         0    128         255

After:   ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃    Uniform across full range
         0    128         255
```

**Use case:** Low-contrast images (e.g., medical imaging, dark scenes).

**Limitation:** Globally enhances contrast even where it shouldn't (e.g., already-saturated regions become noise).

**CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applies equalization in tiles with a contrast limit. Better for images with both bright and dark regions.

### 2.2 Gamma Correction

Non-linear pixel transformation:

$$I'[i,j] = 255 \cdot \left(\frac{I[i,j]}{255}\right)^\gamma$$

**Transformation Curve:**

```
Output (I')
   255 ┤      ┌──── γ < 1 (brightens shadows)
       │    ╱
   192 ┤  ╱      γ = 1 (identity)
       │╱     ╱
   128 ┤    ╱       γ > 1 (compresses highlights)
       │  ╱     ╱
    64 ┤╱     ╱
       │   ╱
     0 ┴─────────────
       0   128    255  Input (I)
```

**Gamma Selection Threshold Table:**

| Gamma ($\gamma$) | Effect             | Use Case                     |
| ---------------- | ------------------ | ---------------------------- |
| 0.4 – 0.6        | Strong brightening | Recover dark images          |
| 0.7 – 0.9        | Mild brightening   | Slight under-exposure        |
| 1.0              | Identity           | No change                    |
| 1.1 – 1.5        | Mild darkening     | Slight over-exposure         |
| 1.5 – 2.5        | Strong darkening   | Recover bright washed images |
| 2.2              | sRGB → Linear      | Color-accurate processing    |

### 2.3 Color Channel Analysis

Each channel can be analyzed independently. Plot channel histograms to detect issues.

**Color Cast Detection:**

If $\bar{R} \neq \bar{G} \neq \bar{B}$ significantly, the image has a color cast.

$$\text{Cast Score} = \frac{\max(\bar{R}, \bar{G}, \bar{B}) - \min(\bar{R}, \bar{G}, \bar{B})}{\bar{R} + \bar{G} + \bar{B}}$$

**Color Cast Threshold Table:**

| Cast Score  | Interpretation                        | Action                   |
| ----------- | ------------------------------------- | ------------------------ |
| < 0.05      | Neutral / balanced                    | None                     |
| 0.05 – 0.15 | Mild cast (tungsten, daylight)        | Optional white balance   |
| 0.15 – 0.30 | Strong cast (incandescent)            | Apply white balance      |
| > 0.30      | Severe cast (color filter, broken WB) | Strong correction needed |

**White Balance Correction (Gray World Assumption):**

$$I'_c[i,j] = I_c[i,j] \cdot \frac{\bar{G}}{\bar{c}}, \quad c \in \{R, G, B\}$$

The assumption: the average of all pixels should be gray (equal RGB values).

### 2.4 Color Distribution Visualization

A 2D color distribution plot reveals image characteristics.

**Hue Histogram (HSV space):**

$$h_H(k) = \sum_{i,j} \mathbb{1}[H[i,j] = k]$$

Reveals dominant colors. Useful for:

- Detecting color-themed images (sunset = red/orange dominant)
- Quality control (expected color distribution)

**Saturation-Value Joint Distribution:**

A 2D plot of $(S, V)$ values reveals image character:

```
     V (Value/Brightness)
   1 ┤  ▒▒░░       ████  ← Vivid, healthy image
     │  ▒▒░░     ████
   0.5┤ ░░░░    ▓▓▓▓     ← Faded or grayscale
     │ ░░░░  ▓▓▓▓
     │ ░░░░▓▓▓▓          ← Dark but saturated
   0 ┴────────────────
     0    0.5    1
        S (Saturation)
```

**Interpretation:**

- High S, high V: Vivid, healthy image
- Low S, any V: Faded or grayscale-like
- High S, low V: Dark, saturated (often noisy)

### 2.5 Postprocessing: Output Refinement

Model outputs often need refinement before deployment.

**Non-Maximum Suppression (NMS) for Detection:**

For overlapping bounding boxes, keep highest-confidence box:

$$\text{IoU}(B_1, B_2) = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|}$$

If $\text{IoU} > \tau$ (typically 0.5), suppress lower-confidence box.

**Soft-NMS** decays scores instead of suppressing:

$$s_i = s_i \cdot e^{-\text{IoU}(M, B_i)^2 / \sigma}$$

**NMS Configuration Threshold Table:**

| IoU Threshold ($\tau$) | Effect                 | Use Case                            |
| ---------------------- | ---------------------- | ----------------------------------- |
| 0.3                    | Aggressive suppression | Sparse objects, no overlap expected |
| 0.5                    | Standard               | General object detection            |
| 0.7                    | Loose suppression      | Crowded scenes, partial occlusions  |

**Probability Calibration (Temperature Scaling):**

Model probabilities are often miscalibrated. Apply:

$$p_{calibrated}(c) = \frac{e^{z_c / T}}{\sum_{c'} e^{z_{c'} / T}}$$

where $T$ is learned on validation set. $T > 1$ softens probabilities; $T < 1$ sharpens them.

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

where $B_m$ are confidence bins.

| ECE         | Interpretation         | Action                    |
| ----------- | ---------------------- | ------------------------- |
| < 0.02      | Well calibrated        | Deploy as-is              |
| 0.02 – 0.05 | Acceptable             | Monitor in production     |
| 0.05 – 0.10 | Miscalibrated          | Apply temperature scaling |
| > 0.10      | Severely miscalibrated | Recalibrate or retrain    |

> **Practical Tip:** After training, always check ECE on validation set. If ECE > 0.05, your probabilities don't reflect true accuracy—apply temperature scaling before deployment.
> {: .prompt-tip }

---

## 3. Feature Extraction Operations

Feature extraction transforms pixels into learned representations. The framework consists of three fundamental operation types.

### 3.1 Local Operations (Convolution)

Convolution applies a learned filter to local neighborhoods:

$$y[i,j] = \sum_{a,b} K[a,b] \cdot x[i+a, j+b]$$

**Properties:**

- Translation equivariant (object moves → feature moves)
- Parameter sharing (same filter across image)
- Local receptive field (sees neighborhood only)

**Convolution Variants:**

| Operation            | Receptive Field          | Parameters           | Use Case                   |
| -------------------- | ------------------------ | -------------------- | -------------------------- |
| Standard Conv        | $k \times k$             | $k^2 C_{in} C_{out}$ | General feature extraction |
| Depthwise Conv       | $k \times k$ per channel | $k^2 C$              | Mobile architectures       |
| Pointwise Conv (1×1) | $1 \times 1$             | $C_{in} C_{out}$     | Channel mixing             |
| Dilated Conv         | $k \times k$ with gaps   | $k^2 C_{in} C_{out}$ | Larger RF, same params     |
| Transposed Conv      | Inverse mapping          | $k^2 C_{in} C_{out}$ | Upsampling                 |

### 3.2 Global Operations (Self-Attention)

Self-attention computes relationships between all spatial positions:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Trade-off vs Convolution:**

| Aspect                  | Convolution             | Self-Attention                 |
| ----------------------- | ----------------------- | ------------------------------ |
| **Receptive field**     | Local, grows with depth | Global from layer 1            |
| **Complexity**          | $O(HW \cdot k^2 C)$     | $O((HW)^2 C)$                  |
| **Inductive bias**      | Strong (locality)       | Weak (learns relationships)    |
| **Data efficiency**     | High                    | Low (needs large pre-training) |
| **Hardware efficiency** | Optimized (cuDNN)       | Less optimized                 |

### 3.3 Hybrid Operations

Modern architectures combine both:

- **Early layers**: Convolution (local features, hardware efficient)
- **Late layers**: Attention (global reasoning)
- **Examples**: ConvNeXt, CoAtNet, MobileViT

---

## 4. Spatial Operations

Spatial operations control how features are scaled and arranged in space.

### 4.1 Downsampling

Reduce spatial resolution while increasing semantic depth.

| Method              | Function             | Information Loss           | Learnable |
| ------------------- | -------------------- | -------------------------- | --------- |
| Max pooling         | $\max$               | Keeps strongest activation | No        |
| Average pooling     | $\text{mean}$        | Smooths features           | No        |
| Strided convolution | Learned filter       | Minimal (learned)          | Yes       |
| Blur pooling        | Gaussian + subsample | Anti-aliased               | No        |

### 4.2 Upsampling

Increase spatial resolution—essential for segmentation, generation.

| Method                 | How                            | Quality                       | Cost      |
| ---------------------- | ------------------------------ | ----------------------------- | --------- |
| Nearest neighbor       | Repeat pixels                  | Blocky                        | Free      |
| Bilinear               | Linear interpolation           | Smooth                        | Cheap     |
| Bicubic                | Cubic interpolation            | Smoother                      | Moderate  |
| Transposed convolution | Learned filter                 | Best (with checkerboard risk) | Expensive |
| Pixel shuffle          | Channel-to-space rearrangement | Best (no checkerboard)        | Cheap     |

> **Practical Tip:** Use pixel shuffle (sub-pixel convolution) over transposed convolution. It avoids checkerboard artifacts and is more parameter-efficient.
> {: .prompt-tip }

### 4.3 Multi-Scale Processing (Feature Pyramid Network)

Real-world objects appear at different scales. Multi-scale operations handle this.

$$P_l = \text{Conv}(C_l + \text{Upsample}(P_{l+1}))$$

This produces features at multiple resolutions, each combining:

- **Bottom-up path**: Low-resolution, semantically strong
- **Top-down path**: High-resolution, spatially precise

**Used by:** Object detection (small + large objects), segmentation (fine + coarse boundaries).

---

## 5. Channel Operations

Channels carry semantic information. Operations on the channel dimension control what features are emphasized.

### 5.1 Channel Mixing (1×1 Convolution)

A 1×1 convolution mixes channels at each spatial position:

$$y_c[i,j] = \sum_{c'} W[c, c'] \cdot x_{c'}[i,j]$$

**Use cases:**

- **Bottleneck**: Reduce channels before expensive operation
- **Projection**: Match channel dimensions for residual connections

### 5.2 Channel Attention (Squeeze-and-Excitation)

Compute per-channel importance weights:

$$s_c = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GlobalAvgPool}(x_c)))$$

$$y_c = s_c \cdot x_c$$

**Cost:** Few parameters, ~1% computation overhead, often 1-2% accuracy gain.

### 5.3 Spatial Attention

Complement to channel attention—learn which spatial positions matter:

$$s[i,j] = \sigma(\text{Conv}([\text{MaxPool}_{ch}(x), \text{AvgPool}_{ch}(x)]))$$

Together, channel + spatial attention forms **CBAM**.

---

## 6. Normalization Operations

Normalization stabilizes activations and gradients during training.

### 6.1 The Normalization Family

Different operations normalize over different dimensions of the activation tensor $x \in \mathbb{R}^{N \times C \times H \times W}$:

| Operation    | Normalizes Over                 | Dependence  | Use Case                   |
| ------------ | ------------------------------- | ----------- | -------------------------- |
| BatchNorm    | $(N, H, W)$ per channel         | Batch size  | Standard CNNs, large batch |
| LayerNorm    | $(C, H, W)$ per sample          | Independent | Transformers               |
| InstanceNorm | $(H, W)$ per channel per sample | Independent | Style transfer             |
| GroupNorm    | $(H, W, G)$ per group           | Independent | Small batch, detection     |

### 6.2 Batch Size Threshold Table

BatchNorm degrades with small batches because batch statistics become noisy:

| Batch Size | Recommended            | Why                           |
| ---------- | ---------------------- | ----------------------------- |
| 1-4        | LayerNorm or GroupNorm | Batch stats unreliable        |
| 8-16       | GroupNorm              | Compromise stability          |
| 32-128     | BatchNorm              | Standard, well-tuned          |
| 128+       | BatchNorm              | Optimal, large-scale training |

> **Practical Tip:** When fine-tuning with small batches on a model trained with BatchNorm, freeze the batch statistics rather than recomputing them.
> {: .prompt-tip }

---

## 7. Augmentation Operations

Augmentation expands the training distribution by transforming inputs.

### 7.1 Geometric Augmentations

Modify spatial structure: random crop, flip, rotation, scale, affine.

### 7.2 Photometric Augmentations

Modify pixel values: brightness, contrast, saturation, hue shift, noise injection.

### 7.3 Mixing Augmentations

**MixUp:** Linear blend
$$x' = \lambda x_i + (1-\lambda) x_j$$

**CutMix:** Spatial paste of one image into another.

**CutOut:** Random masking with zeros.

**Augmentation Strength Table:**

| Method                    | Diversity | Realism  | When to Use                  |
| ------------------------- | --------- | -------- | ---------------------------- |
| Geometric only            | Low       | High     | Small dataset, simple task   |
| Geometric + Photometric   | Medium    | High     | Standard training            |
| + MixUp/CutMix            | High      | Medium   | Strong regularization needed |
| AutoAugment / RandAugment | Highest   | Variable | Large compute budget         |

---

## 8. Loss Design Operations

Loss functions translate task requirements into gradients.

### 8.1 Pixel-Level Losses (Regression)

| Loss        | Formula                             | Property                      |
| ----------- | ----------------------------------- | ----------------------------- |
| L1          | $\|y - \hat{y}\|$                   | Robust to outliers, sharp     |
| L2 (MSE)    | $(y - \hat{y})^2$                   | Smooth, sensitive to outliers |
| Smooth L1   | Hybrid                              | Robust + smooth               |
| Charbonnier | $\sqrt{(y-\hat{y})^2 + \epsilon^2}$ | Smooth approximation of L1    |

### 8.2 Classification Losses

| Loss               | Use Case                |
| ------------------ | ----------------------- |
| Cross-Entropy      | Standard classification |
| Focal Loss         | Class imbalance         |
| Label Smoothing CE | Reduce overconfidence   |

### 8.3 Metric Learning Losses

| Loss              | Property                       |
| ----------------- | ------------------------------ |
| Contrastive       | Pull positives, push negatives |
| Triplet           | Anchor + positive + negative   |
| ArcFace / CosFace | Angular margin on hypersphere  |
| InfoNCE           | Self-supervised contrastive    |

### 8.4 Multi-Task Losses

Real systems combine losses:

$$L = \lambda_1 L_{cls} + \lambda_2 L_{box} + \lambda_3 L_{seg} + ...$$

The weights $\lambda_i$ matter as much as the losses themselves.

---

## Why This Framework Matters

The operations above are not optional details—they determine what's achievable:

> - **Image diagnostic tests** (contrast, distribution shift) reveal problems before training, saving compute
> - **Feature extraction quality** sets a ceiling for every downstream task
> - **Spatial operations** determine handling of scale, resolution, and locality
> - **Channel operations** decide what semantic information is emphasized
> - **Normalization** controls whether training is stable or diverges
> - **Augmentation** sets the effective data distribution the model learns from
> - **Loss design** translates task goals into gradients—wrong loss, wrong learning
> - **Postprocessing** (NMS, calibration) determines deployment quality
>   {: .prompt-warning }

Specialized models (YOLO, LoFTR, ArcFace) are combinations of these operations tuned for specific tasks. Understanding the framework lets you read any vision paper, modify any architecture, and design new systems by composing operations rather than copying recipes.

---

**References:**

- [Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf) (SSIM)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (ResNet)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (Channel attention)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) (Multi-scale)
- [Group Normalization](https://arxiv.org/abs/1803.08494) (Batch-independent norm)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Temperature scaling)
