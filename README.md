# Physics-Guided Automated Second-Phase Mapping in Metallic Alloys Using 4D-STEM

### MATSCI465-Project

A Python pipeline for **automated detection of second phases and precipitates in metallic materials using 4D-STEM diffraction data**.

This repository was developed for **MATSCI 465 – Advanced Electron Microscopy & Diffraction (Northwestern University)** and explores how **physics-informed feature engineering combined with unsupervised machine learning** can be used to identify microstructural phases directly from diffraction signatures.

---

# Background

Understanding the **distribution of second phases and nanoscale precipitates** is critical for controlling the mechanical and functional properties of structural alloys.

Examples include:

- strengthening precipitates in **Al–Cu alloys**
- **γ′ precipitates in Ni-based superalloys**
- **GPI zones in age-hardenable aluminum alloys**
- nanoscale phases in **additively manufactured metals**

Traditionally, precipitates are identified using:

- high-resolution imaging  
- diffraction pattern indexing  
- strain mapping  
- manual interpretation by experts  

However, modern **4D-STEM experiments generate extremely large datasets**, where a diffraction pattern is recorded at every probe position.

A typical dataset has the form:

(scan_y, scan_x, qy, qx)

This results in **thousands to millions of diffraction patterns**, making manual analysis impractical.

Recent advances in computational microscopy suggest that **automated analysis pipelines** can extract meaningful microstructural information directly from diffraction signals.

This project explores a **physics-guided approach to automated phase detection** in 4D-STEM data.

---

# Project Goal

The goal of this project is to develop a **generalizable computational pipeline** that can:

1. Process diffraction patterns from a 4D-STEM datacube  
2. Extract physically meaningful diffraction features  
3. Reduce feature dimensionality  
4. Identify distinct diffraction signatures using clustering  
5. Map those clusters back to spatial phase distributions  

The pipeline is designed to work **without prior labeling**, making it useful for exploratory analysis of new materials systems.

---

# Core Idea

Different phases in a material produce **distinct diffraction signatures** due to differences in:

- lattice spacing  
- crystal structure  
- orientation  
- strain fields  
- scattering intensity distribution  

Instead of relying on a single descriptor, the pipeline extracts **multiple complementary physics-based features** from each diffraction pattern.

These features are combined and clustered to identify regions with similar diffraction behavior.

---

# Pipeline Overview

The workflow of the detection pipeline is:

4D-STEM datacube  
→ Diffraction preprocessing  
→ Physics-guided feature extraction  
→ Feature normalization  
→ Dimensionality reduction (PCA)  
→ Unsupervised clustering  
→ Phase map reconstruction

---

# Feature Engineering

Each diffraction pattern is converted into a **feature vector** describing its scattering behavior.

The pipeline extracts several categories of physics-informed descriptors.

---

## Radial Fingerprints

Radial intensity profiles capture the **distribution of scattering intensity as a function of momentum transfer (q)**.

These features encode:

- lattice spacing differences  
- ring positions  
- peak sharpness  

They are particularly useful for distinguishing phases with **different lattice parameters or diffraction peak structures**.

---

## Virtual Detector Features

The pipeline constructs **virtual detector signals** similar to those used in STEM imaging:

- Bright Field (BF)  
- Dark Field (DF)  
- Annular Dark Field (ADF)

These detectors summarize scattering intensity in different angular regions of reciprocal space.

---

## Angular Anisotropy

Angular features capture directional variations in diffraction intensity.

These descriptors can indicate:

- crystal orientation changes  
- strain-induced anisotropy  
- textured precipitates  

The pipeline computes statistics such as:

- angular sector variance  
- entropy of angular intensity distribution  

---

## Bragg-like Scattering Features

Crystalline precipitates often produce stronger localized peaks.

The pipeline measures:

- number of local maxima  
- peak intensity statistics  
- overall "Bragg-like" scattering strength  

These features help distinguish **crystalline precipitates from diffuse matrix scattering**.

---

# Dimensionality Reduction

The combined feature vector can contain **hundreds of dimensions**.

To improve clustering stability and reduce noise:

1. **Robust scaling** is applied  
2. **Principal Component Analysis (PCA)** is used  

PCA retains the principal components explaining approximately **98% of feature variance**.

---

# Unsupervised Clustering

After dimensionality reduction, scan positions are clustered using:

- **Gaussian Mixture Models (GMM)**  
- **K-Means**

For this project, the pipeline typically separates the data into two clusters:

cluster 0 → matrix  
cluster 1 → precipitate

Because clustering labels are arbitrary, the pipeline determines the **most physically plausible mapping** based on cluster size and feature characteristics.

---

# Repository Structure

MATSCI465-Project

pipelinefinalproject.py  
SIMDataTest.ipynb  
Pipeline_Robustness_Test.ipynb  
README.md

---

# Main Pipeline

## pipelinefinalproject.py

This is the **final version of the precipitate detection pipeline**.

Key capabilities include:

- diffraction preprocessing  
- diffraction center estimation  
- radial fingerprint extraction  
- virtual detector feature extraction  
- angular anisotropy analysis  
- Bragg-like scattering statistics  
- PCA dimensionality reduction  
- unsupervised clustering  
- spatial phase map reconstruction  

---

# Synthetic Data Testing

Because real 4D-STEM datasets often lack ground-truth phase labels, synthetic data is used to validate the pipeline.

---

## SIMDataTest.ipynb

This notebook generates simulated 4D-STEM datasets representing:

Matrix patterns

- broad diffuse diffraction rings  
- lower anisotropy  

Precipitate patterns

- sharper diffraction peaks  
- stronger anisotropy  
- higher scattering contrast  

The notebook tests whether the pipeline can correctly separate the two phases.

---

# Robustness Testing

## Pipeline_Robustness_Test.ipynb

This notebook evaluates the pipeline under **increasingly challenging simulation conditions**.

Three simulation regimes are tested.

### Easy

- strong precipitate signal  
- low noise  
- minimal experimental artifacts  

Expected result: near-perfect phase separation.

### Medium

- moderate contrast  
- beam drift  
- noise  
- orientation variation  

Expected result: good but imperfect phase separation.

### Hard

- weak precipitate contrast  
- rare precipitates  
- strong noise  
- thickness gradients  
- beam drift  

Expected result: degraded but still informative clustering.

---

# Evaluation Metrics

Because synthetic datasets include ground truth phase masks, the following segmentation metrics are computed:

- Precision  
- Recall  
- F1 score  
- Intersection over Union (IoU)  
- Accuracy  
- True / False positive counts  

These metrics provide a quantitative measure of pipeline performance.

---

# Example Usage

```python
import pipelinefinalproject as pf

results = pf.detect_phases_multi(data4d)

phase_map = results["labels_map"]
```

Where the dataset has the form:

data4d shape = (ny, nx, pattern_y, pattern_x)

---

# Dependencies

Core Python libraries:

numpy  
scipy  
matplotlib  
scikit-learn  

Optional libraries for real 4D-STEM workflows:

py4DSTEM  
hyperspy  
abTEM  

---

# Limitations

The current implementation has several limitations:

- clustering assumes two phases  
- no explicit crystallographic indexing  
- strain information not yet integrated  
- spatial regularization not included  

Future versions could integrate:

- strain-based phase detection  
- Bragg peak indexing  
- graph-based clustering  
- deep learning feature extraction  

---

# Future Work

Potential research directions include:

- applying the pipeline to **real metallic 4D-STEM datasets**  
- incorporating **strain mapping features**  
- detecting **coherent nanoscale precipitates**  
- extending the method to **multi-phase materials**

---

# Author

Kittichat Wiratkapun  
Kyle Xu

M.S. Student  
Materials Science and Engineering  
Northwestern University

---

# Acknowledgments

Developed as part of:

MATSCI 465 – Advanced Electron Microscopy & Diffraction  
Northwestern University
