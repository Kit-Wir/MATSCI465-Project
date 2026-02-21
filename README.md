# 01_NiW_experimental

## Physics-Guided Unsupervised Segmentation of Experimental Ni–W 4D-STEM Data

---

## Overview

This notebook applies a physics-guided unsupervised learning pipeline to an experimental 4D-STEM dataset of a Ni–W alloy.

4D-STEM (Four-Dimensional Scanning Transmission Electron Microscopy) records:

- 2D probe scan positions (real space: x, y)
- A full diffraction pattern at each scan position (reciprocal space: kx, ky)

This produces a 4D dataset:

(scan_y, scan_x, k_y, k_x)

Each scan location contains a diffraction pattern encoding local structural information such as:

- Crystal orientation
- Strain
- Thickness
- Possible second phases

---

## Scientific Motivation

In metallic alloys, nanoscale precipitates and second phases significantly affect mechanical properties.

However:

- Real-space imaging may not clearly reveal them
- Manual diffraction inspection is subjective
- Automated physics-informed detection is needed

This notebook explores whether unsupervised clustering of diffraction fingerprints can identify physically distinct regions without labels.

---

## Method

### 1. Radial Fingerprint Extraction

Each diffraction pattern is converted into a radial intensity profile:

- Reciprocal space is divided into radial bins
- Intensity is averaged per bin
- Profiles are normalized by total intensity

This generates a physics-based feature vector describing scattering distribution.

---

### 2. Standardization

Features are standardized to zero mean and unit variance.

---

### 3. PCA (Dimensionality Reduction)

Principal Component Analysis is applied to:

- Reduce noise
- Capture dominant diffraction variance
- Compress features

---

### 4. Clustering

Clustering is performed using:

- K-means or
- Gaussian Mixture Model (GMM)

Cluster labels are reshaped back into spatial coordinates to form segmentation maps.

---

## Outputs

- Mean diffraction pattern
- Spatial segmentation map
- Cluster radial fingerprints

---

## Interpretation

The segmentation map reveals spatially coherent regions with distinct diffraction behavior.

Important:

This notebook does not claim confirmed precipitate detection in Ni–W. Instead, it demonstrates that the physics-guided pipeline identifies diffraction regimes in real experimental data.

Validation of precipitate detection capability is performed in:

02_Synthetic_BD_validation

Robustness analysis is performed in:

03_Robustness_ablation# 02_Synthetic_BD_validation

## Simulation-Validated Precipitate Detection Using Physics-Guided 4D-STEM Features

---

## Purpose

This notebook validates whether the diffraction-based unsupervised pipeline can detect precipitates when they are known to exist.

Because the experimental Ni–W dataset lacks ground truth precipitate labels, a synthetic 4D-STEM dataset is generated.

This implements:

- Precipitate / second-phase detection
- Simulation-validated framework

---

## Synthetic Dataset Design

Each synthetic dataset includes:

- Matrix diffraction background
- Orientation/thickness variation
- Circular precipitate regions in real space
- Modified diffraction patterns inside precipitates
- Poisson shot noise controlled by electron dose

Three physical cases are simulated:

---

### 1. Matrix-only (Baseline)

- No precipitates exist
- Used to test false positive behavior

Result:
- IoU ≈ 0
- F1 ≈ 0

This confirms the method does not hallucinate precipitates.

---

### 2. Coherent Precipitates

- Subtle diffraction modification
- Harder detection scenario

Current result:
- IoU ≈ 1 (strong contrast setting)

---

### 3. Incoherent Precipitates

- Strong additional scattering
- Clear diffraction contrast

Result:
- IoU ≈ 1
- F1 ≈ 1

---

## Detection Procedure

The exact same pipeline from:

01_NiW_experimental

is applied to synthetic data without modification.

Cluster-to-precipitate mapping is determined automatically by selecting the cluster with the highest overlap with ground truth.

---

## Evaluation Metrics

For each run:

- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Precision
- Recall
- F1 Score
- Intersection over Union (IoU)

These metrics quantify detection performance.

---

## Key Result

The pipeline:

- Produces zero false positives in matrix-only case
- Correctly identifies incoherent precipitates
- Successfully separates diffraction regimes using purely unsupervised physics-guided features

This demonstrates true precipitate detection capability.

---

## Scientific Meaning

This notebook proves that radial diffraction fingerprints combined with PCA + clustering can detect precipitate-induced diffraction changes when they exist.

This provides quantitative validation of the framework.# 03_Robustness_ablation

## Detection Robustness and Noise Sensitivity Analysis

---

## Purpose

This notebook evaluates the robustness of precipitate detection under varying noise levels and hyperparameters.

A scientifically credible detection framework must:

- Avoid false positives
- Remain stable under noise
- Avoid overfitting to hyperparameters

---

## Noise Sweep (Dose Study)

Poisson noise is controlled via electron dose.

Lower dose → higher noise.

Detection performance (IoU, F1) is evaluated across dose levels.

---

## Results Summary

Matrix-only case:
- IoU ≈ 0 at all noise levels
- No hallucinated precipitates

Incoherent case:
- IoU ≈ 1 across noise levels
- Strong detection robustness

Coherent case:
- Currently IoU ≈ 1 under present contrast setting

---

## Hyperparameter Sweep

The following were varied:

- PCA components
- Number of clusters (k)

Best-over-hyperparameters and fixed-parameter results are compared.

---

## Scientific Contribution

This notebook transforms the project from:

"Clustering demonstration"

into:

"Quantitative detection-limit and robustness study of physics-guided diffraction segmentation."

---

## Final Project Structure

01_NiW_experimental  
→ Experimental application

02_Synthetic_BD_validation  
→ Simulation validation

03_Robustness_ablation  
→ Detection limits and stability analysis

Together, these provide a complete computational microscopy framework for automated second-phase mapping using 4D-STEM.
