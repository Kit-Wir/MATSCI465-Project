# Physics-Guided Automated Second Phase/Precipitate Detection in 4D-STEM

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

#  PipelineFinalProject  
**Physics-Guided Unsupervised Phase Detection in 4D-STEM**

---

##  Overview

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
Raw 4D Data --> Preprocessing --> Physics-Based Feature Extraction --> Robust Scaling + PCA --> Unsupervised Clustering (GMM / KMeans) --> Precipitate Mapping Heuristic --> Optional Spatial Refinement 

---

#  Core Components

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


