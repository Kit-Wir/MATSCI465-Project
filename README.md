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

# Complete Project Structure
