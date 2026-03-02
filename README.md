# Physics-Guided Automated Precipitate Detection in 4D-STEM

## Overview

This project develops and validates a physics-guided computational pipeline for automated second-phase (precipitate) detection in 4D-STEM diffraction datasets.

The core idea is:

> Use physically meaningful diffraction features (radial intensity fingerprints), combined with unsupervised learning (PCA + clustering), to identify regions with distinct scattering behavior — without manual labeling.

The project consists of three parts:

1. Experimental application to Ni–W 4D-STEM data  
2. Simulation-based validation with ground truth precipitates  
3. Robustness and detection-limit analysis  

Together, these form a complete computational microscopy framework.

---

# Background: What is 4D-STEM?

4D-STEM (Four-Dimensional Scanning Transmission Electron Microscopy) records:

- A 2D scan of probe positions over the sample (real space: x, y)
- A full 2D diffraction pattern at each scan position (reciprocal space: kx, ky)

This produces a 4D dataset:

Each scan location contains structural information encoded in the diffraction pattern, including:

- Crystal orientation
- Strain
- Thickness variations
- Presence of second phases or precipitates

The challenge is extracting this information automatically and reproducibly.

---
## Pipeline Overview (Physics-Guided + Unsupervised)

This project segments 4D-STEM scans into regions with distinct diffraction behavior using a physics-guided feature + unsupervised learning pipeline.

### Input Data Format
The pipeline expects a 4D-STEM array:

- `data4d.shape = (ny, nx, ky, kx)`
  - `(ny, nx)` = real-space scan grid
  - `(ky, kx)` = diffraction pattern at each scan position

### Step 1 — Radial Fingerprint Feature Extraction
Function: `radial_fingerprint_features(data4d, nbins=20, center=None)`

For every diffraction pattern:

1. Define the diffraction center:
   - default center = `(ky//2, kx//2)` if not provided
2. Compute radius `R` for every diffraction pixel relative to the center.
3. Split reciprocal space into `nbins` concentric radial bins (rings).
4. For each bin, sum intensity inside the ring.

This yields a radial intensity profile per scan position (a "fingerprint"):

- `rad_norm.shape = (ny, nx, nbins)`

To make features robust to overall intensity scaling (e.g., thickness, dose), the radial profile is normalized by the total diffraction intensity:

- `rad_norm = radial_sums / total_intensity`

The function returns:
- `Xflat`: flattened features `(ny*nx, nbins)`
- `Ximg`: image-form features `(ny, nx, nbins)`

### Step 2 — Optional Low-q Exclusion
In `run_pipeline`, you can remove the first few bins near the central beam:

- `exclude_low_q > 0` removes the first `exclude_low_q` bins.

This is useful because very low-q intensity can dominate and may reflect thickness/background more than structural differences.

### Step 3 — Standardization
Before PCA and clustering, features are standardized:

- `StandardScaler()` makes each feature dimension have mean 0 and variance 1.

This prevents any one radial bin from dominating due to scale differences.

### Step 4 — PCA (Dimensionality Reduction)
The standardized features are projected to a lower-dimensional space using PCA:

- `pca = PCA(n_components=n_pca)`
- `Z = pca.fit_transform(Xs)`

Then only the first `pca_use` principal components are used for clustering:

- `Z_use = Z[:, :pca_use]`

The pipeline also returns `explained_variance_ratio` to show how much diffraction variance PCA captures.

### Step 5 — Unsupervised Clustering
Two clustering options are supported:

- **KMeans** (`method="kmeans"`)
- **Gaussian Mixture Model** (`method="gmm"`)

Both cluster scan positions based on `Z_use`.

Output labels are reshaped back into scan space:

- `labels.shape = (ny, nx)`

This label map is the final segmentation result.

### Outputs Returned by `run_pipeline`
`run_pipeline()` returns a dictionary containing:

- `labels`: segmentation map `(ny, nx)`
- `labels_flat`: labels `(ny*nx,)`
- `Xfeat_flat`: radial features `(ny*nx, nbins - exclude_low_q)`
- `Xfeat_img`: radial features `(ny, nx, nbins - exclude_low_q)`
- `Z`: PCA embedding `(ny*nx, n_pca)`
- `explained_variance_ratio`: PCA variance fractions
- fitted objects: `model`, `pca`, `scaler`

### Physical Interpretation (Materials Science Meaning)
- The radial fingerprint captures how scattering intensity is distributed across scattering vectors (q).
- Differences in fingerprints can reflect:
  - different phases / second phases (different scattering factors or reflections)
  - orientation domains (diffraction intensity redistribution)
  - strain/thickness variations (changes in low-q vs high-q intensity)
- Clustering groups scan positions with similar scattering behavior, producing a physics-guided segmentation map.

# Part 1 — 01_NiW_experimental

## Objective

Apply a physics-guided unsupervised pipeline to experimental Ni–W 4D-STEM data to segment regions with distinct diffraction behavior.

This part demonstrates real-data application.

---

## Method

### 1. Radial Diffraction Fingerprints

For each diffraction pattern:

- Reciprocal space is divided into radial bins
- Intensity is averaged within each bin
- The profile is normalized by total intensity

This produces a compact feature vector representing:

Intensity vs scattering vector magnitude

This is a physics-based descriptor of scattering behavior.

---

### 2. Standardization

Features are standardized to zero mean and unit variance to remove scale bias.

---

### 3. Dimensionality Reduction (PCA)

Principal Component Analysis (PCA) is applied to:

- Reduce noise
- Capture dominant diffraction variance
- Compress feature space

---

### 4. Unsupervised Clustering

Clustering is performed using:

- K-means or
- Gaussian Mixture Models (GMM)

Cluster labels are mapped back to scan coordinates to produce spatial segmentation maps.

---

## Outputs

- Mean diffraction pattern
- Segmentation map
- Cluster radial fingerprints

---

## Interpretation

The segmentation map identifies spatially coherent regions with distinct diffraction signatures.

Important:

This notebook does not claim confirmed precipitate detection in Ni–W.

Instead, it demonstrates:

> The method can identify diffraction regimes in experimental data without labels.

Validation of precipitate detection capability is performed in Part 2.

---

# Part 2 — 02_Synthetic_BD_validation

## Objective

Validate that the pipeline can detect precipitates when they are known to exist.

Because experimental data lacks ground truth labels, synthetic 4D-STEM datasets are generated.

This implements:

- (B) Precipitate detection  
- (D) Simulation-validated framework  

---

## Synthetic Dataset Design

Each synthetic dataset includes:

- Matrix diffraction pattern
- Orientation/thickness variation across scan
- Circular precipitate regions in real space
- Modified diffraction patterns inside precipitates
- Poisson shot noise controlled by electron dose

Three physical cases are simulated:

---

### Case 1 — Matrix-only (Baseline)

- No precipitates exist
- Used to test false positive behavior

Result:
- IoU ≈ 0
- F1 ≈ 0

This confirms the method does not hallucinate precipitates.

---

### Case 2 — Coherent Precipitates

- Subtle diffraction modification
- Harder detection scenario

Result (current setting):
- IoU ≈ 1
- F1 ≈ 1

Detection succeeds due to sufficient contrast.

---

### Case 3 — Incoherent Precipitates

- Strong additional scattering
- Clear diffraction contrast

Result:
- IoU ≈ 1
- F1 ≈ 1

Detection is robust and accurate.

---

## Evaluation Metrics

Detection quality is quantified using:

- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Precision
- Recall
- F1 Score
- Intersection over Union (IoU)

This transforms the project from visual clustering to quantitative validation.

---

## Key Scientific Result

The pipeline:

- Produces zero false positives in matrix-only case
- Correctly identifies precipitate regions
- Separates diffraction regimes using purely physics-guided features

This establishes true precipitate detection capability.

---

# Part 3 — 03_Robustness_ablation

## Objective

Evaluate robustness and detection limits under varying conditions.

A scientifically credible detection method must:

- Avoid false positives
- Remain stable under noise
- Avoid overfitting to hyperparameters

---

## Noise Sweep (Dose Study)

Poisson noise is controlled via electron dose.

Lower dose → higher noise.

Detection performance (IoU, F1) is evaluated across multiple dose levels.

Result summary:

- Matrix-only case → IoU ≈ 0 at all noise levels
- Incoherent case → IoU ≈ 1 across noise levels
- Coherent case → IoU ≈ 1 under current contrast

---

## Hyperparameter Sweep

The following parameters are varied:

- PCA components
- Number of clusters (k)

Two evaluation modes are used:

1. Fixed hyperparameters
2. Best-over-hyperparameters

This ensures performance is not due to parameter tuning alone.

---

## Scientific Contribution

This notebook elevates the project from:

"Clustering demonstration"

to:

"Quantitative detection-limit study of physics-guided diffraction segmentation."

---

Together, these components form a complete computational microscopy framework for automated second-phase mapping using 4D-STEM.

---

# Summary

This project demonstrates:

- Physics-guided feature engineering (radial diffraction fingerprints)
- Unsupervised learning for diffraction segmentation
- Simulation-based validation with ground truth
- Quantitative robustness analysis
- Application to real experimental data

The framework provides a reproducible, label-free method for automated second-phase detection in 4D-STEM datasets.

#Updated 03/01/2026

#  PipelineFinalProject  
**Physics-Guided Unsupervised Phase Detection in 4D-STEM**

---

##  Overview

`pipelinefinalproject.py` is an updated and expanded version of my original pipeline (`pipeline.py`) for unsupervised phase detection in 4D-STEM datasets.

### Version History

- **pipeline.py (Previous Version)**  
  - Used **radial fingerprint features only**
  - Demonstrated proof-of-concept unsupervised phase separation
  - Limited robustness to real experimental nuisance variation

- **pipelinefinalproject.py (Current Version)**  
  - Multi-feature, physics-guided architecture  
  - Robust preprocessing and detrending  
  - Modular feature extraction  
  - PCA stabilization  
  - Improved cluster mapping heuristic  
  - Synthetic validation framework (SIMDataTest)

This updated version is designed to handle both:
- Synthetic benchmarking data
- Real experimental datasets (DM4, MIB/HDR)

---

#  Scientific Motivation

In real 4D-STEM experiments, phase contrast competes with:

- Thickness gradients
- Illumination variation
- Scan drift
- Detector non-uniformity
- Central disk dominance
- Shot noise

Unsupervised clustering will always separate the strongest variance direction.

The purpose of `pipelinefinalproject` is to:

> Suppress nuisance variation while preserving physically meaningful diffraction signatures of second phases.

---

#  Pipeline Architecture
#  PipelineFinalProject  
**Physics-Guided Unsupervised Phase Detection in 4D-STEM**

---

##  Overview

`pipelinefinalproject.py` is an updated and expanded version of my original pipeline (`pipeline.py`) for unsupervised phase detection in 4D-STEM datasets.

### Version History

- **pipeline.py (Previous Version)**  
  - Used **radial fingerprint features only**
  - Demonstrated proof-of-concept unsupervised phase separation
  - Limited robustness to real experimental nuisance variation

- **pipelinefinalproject.py (Current Version)**  
  - Multi-feature, physics-guided architecture  
  - Robust preprocessing and detrending  
  - Modular feature extraction  
  - PCA stabilization  
  - Improved cluster mapping heuristic  
  - Synthetic validation framework (SIMDataTest)

This updated version is designed to handle both:
- Synthetic benchmarking data
- Real experimental datasets (DM4, MIB/HDR)

---

#  Scientific Motivation

In real 4D-STEM experiments, phase contrast competes with:

- Thickness gradients
- Illumination variation
- Scan drift
- Detector non-uniformity
- Central disk dominance
- Shot noise

Unsupervised clustering will always separate the strongest variance direction.

The purpose of `pipelinefinalproject` is to:

> Suppress nuisance variation while preserving physically meaningful diffraction signatures of second phases.

---

#  Pipeline Architecture
Raw 4D Data
↓
Preprocessing
↓
Physics-Based Feature Extraction
↓
Robust Scaling + PCA
↓
Unsupervised Clustering (GMM / KMeans)
↓
Precipitate Mapping Heuristic
↓
Optional Spatial Refinement


---

# ⚙ Core Components

## 1️ Preprocessing

- Gaussian smoothing (noise suppression)
- Log compression (dynamic range stabilization)
- Winsorization (robust clipping)
- Central disk masking
- Intensity normalization
- Spatial detrending (removes thickness gradients)

This prevents clustering from learning brightness instead of phase.

---

## 2️ Feature Groups (Modular)

Unlike the original `pipeline.py` (radial only), the final version supports:

### `"radial"`
Radial intensity fingerprints  
Captures:
- Ring shifts
- Superlattice reflections
- Lattice parameter changes

---

### `"detectors"`
Virtual BF/DF/ADF integration  
Captures:
- Scattering redistribution
- Diffuse intensity changes

---

### `"angular"`
Angular variance + entropy  
Captures:
- Ordering anisotropy
- Symmetry breaking

---

### `"bragginess"`
Local maxima density / peak intensity  
Captures:
- Bragg disk strength
- Superlattice spot formation

---

### `"com"`
Center-of-mass shifts  
Captures:
- Strain
- Lattice distortions

---

## 3️ PCA Stabilization

Dimensionality reduction with variance target (default 95–98%).

Purpose:
- Remove correlated nuisance features
- Improve clustering stability
- Reduce runtime

---

## 4️ Clustering

Supported methods:
- `kmeans` (fast)
- `gmm` (covariance-aware, recommended)

---

## 5️ Precipitate Mapping Heuristic

Cluster labels are arbitrary.

The pipeline assigns precipitate phase based on:
- Cluster size prior (precipitates typically rare)
- Feature statistics

---

## 6️ Optional Spatial Refinement

Reduces salt-and-pepper noise.

---

#  SIMDataTest (Synthetic Validation Framework)

File: `SIMDataTest.py`

---

## Purpose

To benchmark pipeline robustness under controlled difficulty levels.

Three synthetic regimes are generated:

| Case   | Contrast | Noise | Difficulty |
|--------|----------|-------|------------|
| Easy   | High     | Low   | Clear separation |
| Medium | Moderate | Moderate | Partial feature overlap |
| Hard   | Low      | High  | Strong feature overlap |

---

## Why SIMDataTest Is Important

Unsupervised segmentation performance depends on:

- Signal-to-noise ratio
- Feature separability
- Class imbalance

SIMDataTest allows:

- Controlled validation
- Performance degradation analysis
- Failure regime identification
- Parameter tuning verification

---

## Metrics Reported

- Precision
- Recall
- F1 Score
- IoU
- Accuracy
- TP / FP / FN

Visual outputs:
- Ground truth mask
- Predicted mask
- XOR error map
- Mean diffraction patterns
- Radial fingerprint comparison

---

#  Expected Behavior

### Easy
- Near-perfect F1
- Clean phase separation

### Medium
- High precision
- Reduced recall
- Conservative detection

### Hard
- Overlapping feature space
- Increased FP or FN
- Demonstrates physical detection limit

Degradation is expected and physically meaningful.

---

# Supported Data Formats

## DM4
Loaded via HyperSpy or py4DSTEM import.

## MIB / HDR (SPED)
Loaded via HyperSpy.

All data must be converted to: 
---

# ⚙ Core Components

## 1️ Preprocessing

- Gaussian smoothing (noise suppression)
- Log compression (dynamic range stabilization)
- Winsorization (robust clipping)
- Central disk masking
- Intensity normalization
- Spatial detrending (removes thickness gradients)

This prevents clustering from learning brightness instead of phase.

---

## 2️ Feature Groups (Modular)

Unlike the original `pipeline.py` (radial only), the final version supports:

### `"radial"`
Radial intensity fingerprints  
Captures:
- Ring shifts
- Superlattice reflections
- Lattice parameter changes

---

### `"detectors"`
Virtual BF/DF/ADF integration  
Captures:
- Scattering redistribution
- Diffuse intensity changes

---

### `"angular"`
Angular variance + entropy  
Captures:
- Ordering anisotropy
- Symmetry breaking

---

### `"bragginess"`
Local maxima density / peak intensity  
Captures:
- Bragg disk strength
- Superlattice spot formation

---

### `"com"`
Center-of-mass shifts  
Captures:
- Strain
- Lattice distortions

---

## 3️ PCA Stabilization

Dimensionality reduction with variance target (default 95–98%).

Purpose:
- Remove correlated nuisance features
- Improve clustering stability
- Reduce runtime

---

## 4️ Clustering

Supported methods:
- `kmeans` (fast)
- `gmm` (covariance-aware, recommended)

---

## 5️ Precipitate Mapping Heuristic

Cluster labels are arbitrary.

The pipeline assigns precipitate phase based on:
- Cluster size prior (precipitates typically rare)
- Feature statistics

---

## 6️ Optional Spatial Refinement

Reduces salt-and-pepper noise.

---

#  SIMDataTest (Synthetic Validation Framework)

File: `SIMDataTest.py`

---

## Purpose

To benchmark pipeline robustness under controlled difficulty levels.

Three synthetic regimes are generated:

| Case   | Contrast | Noise | Difficulty |
|--------|----------|-------|------------|
| Easy   | High     | Low   | Clear separation |
| Medium | Moderate | Moderate | Partial feature overlap |
| Hard   | Low      | High  | Strong feature overlap |

---

## Why SIMDataTest Is Important

Unsupervised segmentation performance depends on:

- Signal-to-noise ratio
- Feature separability
- Class imbalance

SIMDataTest allows:

- Controlled validation
- Performance degradation analysis
- Failure regime identification
- Parameter tuning verification

---

## Metrics Reported

- Precision
- Recall
- F1 Score
- IoU
- Accuracy
- TP / FP / FN

Visual outputs:
- Ground truth mask
- Predicted mask
- XOR error map
- Mean diffraction patterns
- Radial fingerprint comparison

---

#  Expected Behavior

### Easy
- Near-perfect F1
- Clean phase separation

### Medium
- High precision
- Reduced recall
- Conservative detection

### Hard
- Overlapping feature space
- Increased FP or FN
- Demonstrates physical detection limit

Degradation is expected and physically meaningful.

---

#  Supported Data Formats

## DM4
Loaded via HyperSpy or py4DSTEM import.

## MIB / HDR (SPED)
Loaded via HyperSpy.

All data must be converted to: (Ny, Nx, Ky, Kx)


---

#  Example Usage

```python
import pipelinefinalproject as pf

res = pf.detect_phases_multi(
    data4d,
    n_clusters=2,
    method="gmm",
    feature_groups=["radial", "angular", "bragginess"],
    detrend=True,
    spatial_refine=True,
    verbose=True
)

labels = res["labels_map"]


