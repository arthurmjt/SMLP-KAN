# Approximate or Perish: Spectral MLP-KAN Diffusion with Attentive Function Learning for Unsupervised Hyperspectral Image Restoration

<div align="center">

[![CVPR](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-Google%20Drive-red.svg)](https://drive.google.com/file/d/1zgrublrd0Sbcc5g2rbzxJaCKad2QaXY1/view?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hongcheng Jiang<sup>1\*</sup>, Jingtang Ma<sup>2†</sup>, Gaoyuan Du<sup>3</sup>, Jingchen Sun<sup>4</sup>, Gengyuan Zhang<sup>5</sup>, Zejun Zhang<sup>6</sup>, Kai Luo<sup>7‡</sup>**

<sup>1</sup>Liaoning Finance & Trade College &nbsp;|&nbsp;
<sup>2</sup>Amazon Web Services &nbsp;|&nbsp;
<sup>3</sup>University of Tennessee, Knoxville

<sup>4</sup>University at Buffalo, SUNY &nbsp;|&nbsp;
<sup>5</sup>Ludwig Maximilian University of Munich &nbsp;|&nbsp;
<sup>6</sup>University of Southern California &nbsp;|&nbsp;
<sup>7</sup>University of Virginia

<sup>\*</sup>University of Missouri–Kansas City (former PhD student) &nbsp;&nbsp;
<sup>†</sup>Project leader &nbsp;&nbsp;
<sup>‡</sup>Corresponding author

[hjq44@mail.umkc.edu](mailto:hjq44@mail.umkc.edu) &nbsp;|&nbsp;
[majingta@amazon.com](mailto:majingta@amazon.com) &nbsp;|&nbsp;
[kl3pq@virgina.com](mailto:kl3pq@virgina.com)

[[Paper]](https://drive.google.com/file/d/1zgrublrd0Sbcc5g2rbzxJaCKad2QaXY1/view?usp=sharing) &nbsp;|&nbsp;
[[Code]](https://github.com/kailuo93/SMLP-KAN)

</div>

---

## 🔥 News

- **2026-04** — Code released at [SMLP-KAN](https://github.com/kailuo93/SMLP-KAN).
- **2026-03** — SMLP-KAN accepted at **CVPRW (PBVS) 2026**.

---

## Motivation

<div align="center">
<img src="fig/teaser.png" width="900"/>
<br>
<em>
Conventional KAN extracts feature maps uniformly across the image, producing redundant representations.
Our proposed Attentive KAN (RHAG) adaptively selects informative feature maps via dynamic weighting,
reducing redundancy and enhancing spatial relevance.
</em>
</div>

<br>

<div align="center">
<img src="fig/spectral_profile.png" width="900"/>
<br>
<em>
Spectral profile visualization of a clean HSI (CL-HSI) and its degraded counterpart (DG-HSI)
from the WDC dataset. The 1-D spectral distributions of DG-HSIs exhibit statistical similarity
to those of CL-HSIs, making diffusion-based prior learning tractable.
</em>
</div>

---

## Abstract

Hyperspectral imaging suffers from degradation due to atmospheric disturbances, diffusion effects, and surface reflectance variations. To address this, we propose **SMLP-KAN** (Spectral MLP-KAN Diffusion Prior), an unsupervised framework that combines spectral diffusion priors with function learning and attention-driven feature refinement. It consists of three components: an **MLP-KAN Feature Extractor** for hierarchical abstraction and explicit function modeling, a **Spectral Diffusion Prior Module** to encode the stochastic evolution from degraded to clean spectra, and an **Attentive Function Learning Mechanism** to refine high-confidence features and suppress redundancies. Without labeled data, SMLP-KAN fuses degraded hyperspectral signals with spatial priors, enhancing spectral fidelity and spatial resolution. Experiments on real-world datasets show it outperforms state-of-the-art methods in **sharpening**, **denoising**, and **inpainting**.

---

## Architecture

<div align="center">
<img src="fig/architecture.png" width="900"/>
<br>
<em>
Overall architecture of SMLP-KAN: shallow feature extraction (MLPB),
deep feature extraction via RHAG (PSAB + CAAB, repeated K times),
and feature reconstruction (FRB).
</em>
</div>

<br>

**Three-stage per-spectral-vector pipeline:**

```
DG-HSI spectral vector  x_s  ∈ R^S
         │
         ▼
 ┌──────────────────────────────────┐
 │  Stage 1 — MLP Block (MLPB)     │
 │  F_s = Linear(SiLU(Linear(x_s)))│
 └──────────────┬───────────────────┘
                │
                ▼
 ┌──────────────────────────────────┐
 │  Stage 2 — RHAG  (× K layers)   │
 │  F_n = LayerNorm(F_s)            │
 │  ┌──────────┐   ┌─────────────┐  │
 │  │  PSAB    │   │    CAAB     │  │
 │  │ (spline) │   │  (k-means)  │  │
 │  └────┬─────┘   └──────┬──────┘  │
 │       └── F_d = F_p + F_c ───────┘
 └──────────────┬───────────────────┘
                │
                ▼
 ┌──────────────────────────────────┐
 │  Stage 3 — FRB                  │
 │  ε_θ = Dropout(LayerNorm(F_d))  │
 └──────────────────────────────────┘
                │
                ▼
         CL-HSI output  x_t  ∈ R^S
```

| Component | Role | Key Formula |
|---|---|---|
| **MLPB** | Shallow spectral feature extraction | `F_s = Linear(SiLU(Linear(x_s)))` |
| **PSAB** | Spline-based spectral smoothing + attention | `φ(F_n) = 1−tanh²((F_n−g)/d)` → `F_p = softmax(Linear(F_y))·F_y` |
| **CAAB** | Cluster-based dominant feature selection | `F_c = k-means(LayerNorm(F_n))`, fixed k=2 |
| **FRB** | Reconstruction and regularisation | `ε_θ = Dropout(LayerNorm(F_d))` |

### Attentive KAN (RHAG)

<div align="center">
<img src="fig/rhag.png" width="900"/>
<br>
<em>
Left: standard KAN using hierarchical 1-D functions for approximation.
Right: our Attentive KAN (RHAG), which selectively refines feature interactions
with dynamic weighting. Bottom insets highlight the structured basis-function refinement
in RHAG compared to a vanilla KAN.
</em>
</div>

### Spectral Diffusion Process

<div align="center">
<img src="fig/diffusion_process.png" width="900"/>
<br>
<em>
Spectral diffusion process for three representative bands in SMLP-KAN.
The forward process (left to right) adds Gaussian noise;
the reverse process denoises guided by the learned spectral prior
and the spatial fidelity term (HR-PCI).
</em>
</div>

---

## Why MLP-KAN?

<div align="center">
<img src="fig/psnr_bar.png" width="900"/>
<br>
<em>
PSNR comparison of pure MLP, pure KAN, and MLP-KAN (PSAB + CAAB) on four benchmark datasets
for ×4 hyperspectral sharpening. MLP focuses on representation learning;
KAN emphasises function approximation. MLP-KAN unifies both,
achieving consistently higher PSNR across all datasets.
</em>
</div>

---

## Results

### HSI Sharpening (×2 scale, PSNR dB ↑)

| Method | Botswana | Chikusei | PaviaC | PaviaU | WDC |
|---|:-:|:-:|:-:|:-:|:-:|
| DHP-DARN | 27.64 | 27.89 | 31.60 | 35.79 | 24.12 |
| DIP-HyperKite | 28.69 | 27.52 | 34.33 | 35.55 | 27.22 |
| HyperPNN | 29.78 | 27.09 | 33.03 | 33.65 | 25.67 |
| PLRDiff | 15.27 | 32.78 | 33.45 | 35.33 | 11.53 |
| PSDip | 29.20 | 28.54 | 27.75 | 31.16 | 27.08 |
| **SMLP-KAN** | **34.74** | **36.18** | **34.78** | **35.98** | **31.30** |

### HSI Denoising (σ = 0.1)

<div align="center">
<img src="fig/denoising_visual.png" width="900"/>
<br>
<em>
Visual comparison for HSI denoising (σ = 0.1) on PaviaC dataset.
SMLP-KAN preserves fine spatial structure and spectral fidelity better than all baselines.
</em>
</div>

<br>

<div align="center">
<img src="fig/bandwise_psnr.png" width="650"/>
<br>
<em>
Band-wise PSNR for HSI denoising (σ = 0.1) on PaviaC (102 bands).
SMLP-KAN maintains consistently high PSNR across all spectral bands.
</em>
</div>

<br>

| Method | Botswana | Chikusei | PaviaC | PaviaU | WDC |
|---|:-:|:-:|:-:|:-:|:-:|
| GPPNN | 28.75 | 24.46 | 28.33 | 31.45 | 25.24 |
| PLRDiff | 29.47 | 28.95 | 31.11 | 32.41 | 23.27 |
| **SMLP-KAN** | **31.68** | **30.01** | **31.30** | **32.61** | **26.50** |

### HSI Inpainting (masking rate 0.05, PSNR dB ↑)

| Method | Botswana | Chikusei | PaviaC | PaviaU | WDC |
|---|:-:|:-:|:-:|:-:|:-:|
| DMLD-Net | 30.33 | 28.59 | 29.50 | 30.43 | 22.44 |
| PLRDiff | 27.11 | 29.82 | 30.77 | 31.93 | 12.73 |
| **SMLP-KAN** | **30.97** | **30.41** | **30.80** | **31.55** | **28.13** |

### Real-world Evaluation

<div align="center">
<img src="fig/realworld.png" width="900"/>
<br>
<em>
Results on the Liao Ning-01 satellite dataset (ZY-1 02D, Dalian, Liaoning).
Input: LR-HSI (100×100×166) + HR-PCI (300×300×1).
SMLP-KAN recovers fine spatial details while preserving consistent spectral
relationships across all 166 bands.
</em>
</div>

### Parameter Efficiency

| Method | Parameters | FLOPs / step |
|---|:-:|:-:|
| HIR-Diff | 391.05 M | — |
| PLRDiff | 391.95 M | — |
| **SMLP-KAN** | **9.5 M** | see paper |

SMLP-KAN achieves state-of-the-art performance with **~41× fewer parameters** than competing diffusion-based methods.

---

## Ablation Study

### Effect of RHAG Components (HSI Inpainting, masking rate 0.05)

| PSAB | CAAB | Botswana | Chikusei | PaviaU |
|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | 26.02 | 29.35 | 25.37 |
| ✓ | ✗ | 28.51 | 29.69 | 29.98 |
| ✗ | ✓ | 28.41 | 29.65 | 29.57 |
| ✓ | ✓ | **30.97** | **30.41** | **31.55** |

Both PSAB and CAAB provide complementary gains; their combination consistently yields the best results.

### Regularisation Parameters λ and γ

<div align="center">
<img src="fig/lambda_gamma.png" width="600"/>
<br>
<em>
PSNR heatmap under varying λ and γ for HSI inpainting (masking rate 0.05) on Botswana.
Peak performance concentrates in a specific (λ, γ) region.
Excessively large or small values lead to consistent performance degradation.
</em>
</div>

---

## Requirements

```
Python    >= 3.8
PyTorch   >= 1.11
torchvision
scipy
numpy
matplotlib
fvcore      # optional — FLOPs estimation only
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=1.11.0
torchvision>=0.12.0
scipy>=1.7.0
numpy>=1.21.0
matplotlib>=3.4.0
fvcore
```

---

## Data Preparation

Download the five benchmark datasets and place them under `data/`:

| Dataset | Spatial | Bands | Source |
|---|:-:|:-:|---|
| Botswana | 1476×256 | 145 | [EO-1 Hyperion / USGS](https://www.usgs.gov/) |
| Chikusei | 2517×2335 | 128 | [Space Appl. Lab., Univ. Tokyo](https://naotoyokoya.com/Download.html) |
| PaviaC | 1096×715 | 102 | [Univ. Pavia](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| PaviaU | 610×340 | 102 | [Univ. Pavia](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| WDC Mall | 1208×307 | 191 | [AVIRIS / JPL](https://aviris.jpl.nasa.gov/) |

Each dataset is cropped to a central **256×256** patch. DG-HSIs and HR-PCIs are synthesised following [Wald's protocol](https://doi.org/10.14358/PERS.63.6.691), with HR-PCI obtained by averaging the visible bands of the CL-HSI.

Expected `.mat` keys:

| Key | Shape | Description |
|---|---|---|
| `LRMS` | (h, w, L) | Low-resolution HSI |
| `PAN` | (H, W) | High-resolution panchromatic image |
| `HRMS` | (H, W, L) | Ground-truth clean HSI |
| `K` *(optional)* | (k, k) | Point spread function |
| `R` *(optional)* | (l, L) | Spectral response function |

Expected directory layout:

```
data/
├── PaviaC_256_2_4.mat
├── PaviaC_256_4_4.mat
├── Botswana_256_2_4.mat
└── ...
```

---

## Usage

```bash
python SMLP_KAN.py
```

Or configure programmatically:

```python
ndata, nratio, nsnr = 0, 8, 0    # dataset / ratio / noise index

# Stage 1: learn spectral diffusion prior
spec_net = SDM(ndata=ndata, nratio=nratio, nsnr=nsnr)
spec_net.train()

# Blind PSF / SRF estimation
blind = Blind(ndata=ndata, nratio=nratio, nsnr=nsnr, blind=True, kernel=8)
blind.train()
blind.get_save_result(is_save=True)

# Stage 2: score-guided reconstruction
gams = [1e-3, 1e-3, 1e-3, 1e-1]
net = SMLPKAN(ndata=ndata, nratio=nratio, nsnr=nsnr, psf=blind.psf, srf=blind.srf)
net.train(gam=gams[ndata])
```

---

## Project Structure

```
SMLP-KAN/
├── fig/
│   ├── teaser.png               # Fig 1  — KAN vs Attentive KAN feature maps
│   ├── spectral_profile.png     # Fig 2  — CL-HSI vs DG-HSI spectral profiles
│   ├── psnr_bar.png             # Fig 3  — MLP / KAN / MLP-KAN PSNR bar chart
│   ├── diffusion_process.png    # Fig 4  — Spectral diffusion process (3 bands)
│   ├── architecture.png         # Fig 5  — Full SMLP-KAN architecture
│   ├── rhag.png                 # Fig 6  — KAN vs RHAG comparison
│   ├── denoising_visual.png     # Fig 7  — HSI denoising visual results
│   ├── bandwise_psnr.png        # Fig 8  — Band-wise PSNR on PaviaC
│   ├── realworld.png            # Fig 9  — Liao Ning-01 real-world results
│   └── lambda_gamma.png         # Fig 10 — λ/γ regularisation heatmap
├── data/
│   ├── data_info.py             # DataInfo base class (I/O and preprocessing)
│   └── psy.py
├── model/
│   ├── smlp_kan.py              # SMLP-KAN backbone (MLPB, PSAB, CAAB, RHAG, FRB)
│   └── gaussian_diffusion.py    # DDPM forward/reverse process
├── utils/
│   ├── blur_down.py             # Gaussian blur + downsampling + AWGN
│   ├── toolkits.py
│   ├── torchkits.py
│   └── ema.py
├── blind.py                     # Blind PSF/SRF estimation
├── metrics.py                   # PSNR, SSIM, SAM, ERGAS, CC
├── SMLP_KAN.py                  # Main entry point
├── requirements.txt
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{jiang2026smlpkan,
  title     = {Approximate or Perish: Spectral MLP-KAN Diffusion with Attentive
               Function Learning for Unsupervised Hyperspectral Image Restoration},
  author    = {Jiang, Hongcheng and Ma, Jingtang and Du, Gaoyuan and Sun, Jingchen
               and Zhang, Gengyuan and Zhang, Zejun and Luo, Kai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

Related works from our group that this code builds upon:

```bibtex
@article{jiang2025transformer,
  author  = {Jiang, Hongcheng and Chen, ZhiQiang},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  title   = {Transformer-based Diffusion and Spectral Priors Model for
             Hyperspectral Pansharpening},
  year    = {2025},
  pages   = {1--17},
  doi     = {10.1109/JSTARS.2025.3590685}
}

@inproceedings{jiang2025hyperspectral,
  title     = {Hyperspectral Pansharpening with Transformer-Based Spectral Diffusion Priors},
  author    = {Jiang, Hongcheng and Chen, ZhiQiang},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {581--590},
  year      = {2025}
}

@article{liu2024spectral,
  title   = {A Spectral Diffusion Prior for Unsupervised Hyperspectral Image Super-Resolution},
  author  = {Liu, Jianjun and Wu, Zebin and Xiao, Liang},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2024}
}
```

---

## Acknowledgements

We thank the authors of
[HIR-Diff](https://github.com/LiPang/HIRDiff),
[PLRDiff](https://github.com/xyrui/PLRDiff), and
[SDP](https://github.com/liuofficial/SDP)
for their open-source implementations, which served as references for this work.

---

## Contact

For questions or issues, please open a [GitHub Issue](https://github.com/jianghongcheng/SMLP-KAN/issues)
or contact the corresponding author at [kl3pq@virgina.com](mailto:kl3pq@virgina.com).