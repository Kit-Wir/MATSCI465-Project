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

# Pipeline_Robustness_Test

Robustness benchmarking notebook for the **physics-guided automated second-phase / precipitate detection pipeline** developed for the MATSCI 465 final project. This notebook evaluates how well the pipeline performs under progressively more difficult simulated 4D-STEM conditions by testing **easy**, **medium**, and **hard** cases with known ground-truth masks.

The notebook is designed to answer a simple question:

> **How robust is the detection pipeline when precipitates become rarer, lower-contrast, noisier, and harder to separate from the matrix?**

This benchmarking approach is consistent with the project goal of developing a reproducible diffraction-based framework for automated second-phase mapping in metallic alloys using 4D-STEM, and with the proposal’s plan to test robustness under synthetic perturbations. :contentReference[oaicite:0]{index=0} It also matches the course emphasis on reproducible computational workflows, quantitative evaluation, and 4D-STEM analysis pipelines. :contentReference[oaicite:1]{index=1}

---

## Purpose

`Pipeline_Robustness_Test.ipynb` is a testing notebook for the final detection pipeline in `pipelinefinalproject.py`. It does **not** develop the pipeline itself; instead, it provides a controlled framework to:

- generate simulated 4D-STEM datasets with known precipitate masks,
- run the final unsupervised detection pipeline,
- map cluster labels to a precipitate class,
- compute quantitative segmentation metrics,
- compare performance across multiple difficulty levels,
- visualize prediction quality and failure modes.

This notebook is useful for:

- validating whether the pipeline works under controlled conditions,
- demonstrating robustness in the final report or presentation,
- identifying where the pipeline begins to fail,
- motivating future improvements for real experimental datasets.

---

## What the notebook does

The notebook performs the following steps:

1. **Loads the final pipeline module**
   - Imports `pipelinefinalproject as pf`
   - Confirms which pipeline file is being used

2. **Defines evaluation metrics**
   - Precision
   - Recall
   - F1-score
   - IoU
   - Accuracy
   - TP / FP / FN counts

3. **Defines a label-mapping helper**
   - Because clustering labels are arbitrary, the notebook tests both binary mappings:
     - precipitate = label 0
     - precipitate = label 1
   - It then keeps the mapping that gives the better agreement with ground truth

4. **Defines a plotting helper**
   For each simulation case, the notebook can display:
   - ground-truth precipitate mask
   - predicted precipitate mask
   - XOR error map
   - example matrix diffraction pattern
   - example precipitate diffraction pattern
   - radial fingerprints for matrix vs precipitate

5. **Uses fixed pipeline settings**
   The pipeline is run with a consistent set of parameters so that performance differences come mainly from the **simulation difficulty**, not from changing the model.

6. **Runs three difficulty levels**
   - **Easy**
   - **Medium**
   - **Hard**

7. **Prints a summary table**
   Final performance is reported side-by-side across all cases.

---

## Difficulty levels

The notebook defines three synthetic scenarios:

### Easy
A relatively favorable case with:
- higher precipitate fraction,
- stronger diffraction contrast,
- lower noise,
- no drift,
- minimal beamstop effects,
- no orientation variation.

This case tests whether the pipeline can recover precipitates when the signal is clear.

### Medium
A more realistic intermediate case with:
- lower precipitate fraction,
- moderate contrast,
- higher noise,
- nonzero drift,
- thickness gradient,
- beamstop effects,
- orientation variation.

This case tests whether the pipeline remains stable when multiple nuisance factors are present.

### Hard
A challenging case with:
- rare precipitates,
- weak contrast,
- strong noise,
- larger drift,
- thickness gradient,
- beamstop effects,
- orientation variation.

This case tests the limits of the pipeline and highlights likely failure modes in more difficult experimental conditions.

---

## Pipeline settings used

The notebook uses a fixed configuration for `detect_phases_multi()`:

- `n_clusters=2`
- `method="gmm"`
- `radial_bins=96`
- `feature_groups=["radial", "detectors", "angular", "bragginess"]`
- preprocessing with:
  - Gaussian blur
  - log compression
  - winsorization
  - center masking
  - max normalization
- PCA enabled with `pca_var=0.98`
- no detrending
- precipitate mapping enabled
- rare-phase prior via `size_prior=0.2`
- no spatial refinement

These settings were chosen to provide a stable baseline for comparing simulation conditions rather than for exhaustive hyperparameter optimization.

---

## Simulation options

The notebook supports two ways to generate test data.

### Option A: Custom notebook simulator
Uses your own simulation function:

```python
simulate_metal_4dstem_dataset(...)
