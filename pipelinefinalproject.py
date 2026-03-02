"""
4D-STEM Second‑Phase Detection Pipeline
======================================

This module implements a simple analysis pipeline for classifying pixels in a
4D‑STEM dataset into matrix and second‑phase (precipitate) categories.  The
approach is intentionally modular: it computes physically‑motivated features
from individual diffraction patterns and uses unsupervised clustering to
differentiate patterns based on those features.  While simple, the code is
structured to be a starting point for more sophisticated workflows (e.g. using
supervised learning, template matching or deep neural networks).  Key steps in
the pipeline are:

1. **Preprocessing**: Each diffraction pattern is background subtracted and
   normalized.  A local background estimate is obtained by smoothing and
   subtracting a blurred version of the pattern.

2. **Center determination**: The beam centre is estimated for each pattern
   using the intensity‑weighted centre of mass (COM).  The COM method is
   widely used in 4D‑STEM to measure the net momentum transfer of the
   diffraction pattern【368022248550690†L163-L169】.

3. **Radial profile calculation**:  For each pattern we compute a 1D radial
   fingerprint by azimuthally averaging intensities in concentric annuli
   around the estimated centre.  Radial profile analysis has been shown to be
   sensitive to differences in local structure.  For example, amorphous and
   crystalline regions produce distinctly different radial profiles: amorphous
   patterns show broad diffuse rings while crystalline precipitates exhibit
   sharp peaks at characteristic scattering vectors【752631729019755†L60-L71】.

4. **Feature extraction**:  Additional descriptors such as the total
   scattering intensity, the radius of peak maxima, and the variance of the
   radial profile are computed.  These features capture key differences
   between matrix and precipitate patterns; for instance, crystalline
   precipitates often have higher intensity at specific radii due to Bragg
   reflections, whereas the matrix may be dominated by diffuse scattering.

5. **Clustering**:  The radial fingerprints and auxiliary features are
   concatenated into a feature matrix for unsupervised clustering.  Here we
   provide two simple algorithms: k‑means and Gaussian mixture models.  The
   number of clusters can be specified by the user (default `n_clusters=2`,
   corresponding to matrix and second phase).  For k‑means, the cluster with
   the largest average total intensity (or variance) is assumed to correspond
   to the precipitates.

The functions in this module operate on 4D datasets represented as
NumPy arrays with shape `(Ny, Nx, P, P)`, where `(Ny, Nx)` is the scanning
grid and `(P, P)` are the pixel dimensions of the diffraction pattern.  The
pipeline is intentionally lightweight and relies only on NumPy and SciPy.  It
should run on modest hardware, but beware that very large datasets (e.g.
`>1000×1000` scan positions) may require more memory than is available.

Example usage (simulation and clustering):

```python
import numpy as np
from fourdstem_pipeline import simulate_4dstem_dataset, detect_phases

# Simulate a 4D dataset with precipitates covering ~20% of the field
data, mask_true = simulate_4dstem_dataset(ny=64, nx=64, pattern_size=128,
                                          precipitate_fraction=0.2,
                                          seed=42)

# Run the detection pipeline
result = detect_phases(data, n_clusters=2, method='kmeans',
                       radial_bins=100, verbose=True)

predicted_mask = result['labels'].reshape(data.shape[0], data.shape[1])

# Compute detection accuracy against the ground truth mask (for simulated data)
accuracy = (predicted_mask == mask_true).mean()
print(f"Detection accuracy: {accuracy:.2%}")

```

This simulation function creates synthetic crystalline and amorphous
diffraction patterns with noise, so it provides a basic sanity check for the
algorithm.  When applying the pipeline to real data you should replace the
simulation step with loading your 4D dataset (e.g. using h5py or py4DSTEM
readers).  Then call `detect_phases` on the 4D array.

All main functions are documented below.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
try:
    # Gaussian mixture models moved to sklearn.mixture in recent versions
    from sklearn.mixture import GaussianMixture
except ImportError:
    # Fall back if unavailable
    GaussianMixture = None

# Import trigonometric functions for anisotropy and centre‑of‑mass metrics
from math import pi, sin, cos, atan2

# Publicly exposed functions.  Additional helper functions for virtual
# bright‑/dark‑field imaging, anisotropy and centre‑of‑mass shift metrics
# are also exported for advanced workflows.
__all__ = [
    "background_subtract",
    "estimate_center_of_mass",
    "radial_profile",
    "extract_features",
    "detect_phases",
    "detect_phases_multi",
    "virtual_bright_dark_field",
    "anisotropy_features",
    "com_shift_features",
    "simulate_4dstem_dataset",
]


def background_subtract(pattern: np.ndarray, blur_sigma: float = 5.0) -> np.ndarray:
    """Estimate and subtract a smooth background from a diffraction pattern.

    Parameters
    ----------
    pattern : np.ndarray
        2D array representing a single diffraction pattern.
    blur_sigma : float, optional
        Standard deviation for the Gaussian blur used to estimate the
        background.  Larger values remove lower‑frequency variations.

    Returns
    -------
    np.ndarray
        Background‑subtracted pattern with negative values set to zero.

    Notes
    -----
    Background subtraction is essential in 4D‑STEM analysis because
    inelastically scattered electrons add a slowly varying intensity to the
    diffraction pattern【916875021399175†L610-L729】.  By convolving the pattern with
    a Gaussian kernel we obtain a smoothed estimate of this background and
    subtract it from the raw data.  Negative values are clipped to zero.
    """
    # Estimate smooth background via Gaussian filtering
    background = ndi.gaussian_filter(pattern.astype(float), sigma=blur_sigma)
    corrected = pattern.astype(float) - background
    corrected[corrected < 0] = 0.0
    return corrected


def estimate_center_of_mass(pattern: np.ndarray) -> tuple[float, float]:
    """Estimate the beam centre by computing the intensity‑weighted centre of mass.

    Parameters
    ----------
    pattern : np.ndarray
        2D array representing a diffraction pattern (background subtracted).

    Returns
    -------
    (float, float)
        Coordinates (row, column) of the centre of mass.

    Notes
    -----
    The centre of mass (COM) of a diffraction pattern provides a measure of
    net momentum transfer.  When the sample exerts a force on the beam, the
    COM shifts accordingly, and mapping this shift across the scan reveals
    local electric and magnetic fields【368022248550690†L163-L169】.  In this context we
    use the COM purely to align the radial integration.
    """
    total_intensity = pattern.sum()
    if total_intensity == 0:
        # Avoid division by zero; return the central pixel
        return (pattern.shape[0] / 2.0, pattern.shape[1] / 2.0)
    indices = np.indices(pattern.shape)
    y = np.sum(indices[0] * pattern) / total_intensity
    x = np.sum(indices[1] * pattern) / total_intensity
    return float(y), float(x)


def radial_profile(pattern: np.ndarray, center: tuple[float, float] | None = None,
                   radial_bins: int | None = None) -> np.ndarray:
    """Compute the azimuthally averaged radial profile of a diffraction pattern.

    Parameters
    ----------
    pattern : np.ndarray
        2D diffraction pattern (background subtracted).
    center : tuple of float, optional
        (row, column) coordinates of the beam centre.  If None, the
        centre of mass is estimated automatically.
    radial_bins : int, optional
        Number of bins to use for the radial histogram.  If None, the
        number of bins is chosen such that each pixel row contributes one
        radial bin.

    Returns
    -------
    np.ndarray
        1D array containing the radial profile (mean intensity as a function
        of radius).

    Notes
    -----
    The radial profile is computed by assigning each pixel to a radial bin
    based on its distance from the centre and averaging the intensities in
    each bin.  Azimuthal averaging enhances the signal‑to‑noise ratio and
    collapses rotationally symmetric information【752631729019755†L60-L71】.  Sharp peaks
    in the radial profile correspond to strong Bragg reflections from
    crystalline domains, whereas diffuse rings or flat profiles correspond
    to amorphous material.
    """
    ny, nx = pattern.shape
    if center is None:
        centre = estimate_center_of_mass(pattern)
    else:
        centre = center
    y0, x0 = centre
    y_indices, x_indices = np.indices(pattern.shape)
    # Compute radial distance from the centre for each pixel
    r = np.sqrt((y_indices - y0)**2 + (x_indices - x0)**2)
    # Determine number of bins
    max_r = r.max()
    if radial_bins is None:
        radial_bins = int(max(ny, nx) / 2)
    bin_edges = np.linspace(0, max_r, radial_bins + 1)
    # Bin the radial distances
    radial_sum = np.zeros(radial_bins)
    counts = np.zeros(radial_bins)
    # Flatten arrays for efficient computation
    r_flat = r.flatten()
    intensity_flat = pattern.flatten()
    # Digitize radii into bins; bin indices run from 1 to radial_bins inclusive
    bin_indices = np.digitize(r_flat, bin_edges) - 1
    # Accumulate intensities per bin
    for idx, intensity in zip(bin_indices, intensity_flat):
        if 0 <= idx < radial_bins:
            radial_sum[idx] += intensity
            counts[idx] += 1
    # Avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = radial_sum / counts
    # Replace NaNs with zeros where no pixels fall into the bin
    profile[np.isnan(profile)] = 0.0
    return profile


def extract_features(radial: np.ndarray) -> dict[str, float]:
    """Extract scalar descriptors from a radial profile.

    Parameters
    ----------
    radial : np.ndarray
        The radial intensity profile.

    Returns
    -------
    dict
        Dictionary containing simple scalar features: total intensity,
        maximum intensity, position of the maximum, number of significant peaks,
        and variance of the radial profile.

    Notes
    -----
    The scalar features provide a compact representation of the radial profile
    that captures salient differences between phases.  Crystalline
    precipitates often produce high maximum intensities and multiple
    well‑separated peaks, whereas amorphous or disordered phases have lower
    maxima and smoother profiles.  These descriptors can be combined with
    the full radial fingerprint for clustering.
    """
    total_intensity = np.sum(radial)
    max_intensity = float(np.max(radial))
    position_of_max = float(np.argmax(radial))
    # Count significant peaks using SciPy's find_peaks
    # A peak is considered significant if its height exceeds 10% of the max
    peaks, _ = find_peaks(radial, height=max_intensity * 0.1)
    num_peaks = len(peaks)
    variance = float(np.var(radial))
    return {
        "total": float(total_intensity),
        "max": max_intensity,
        "argmax": position_of_max,
        "n_peaks": float(num_peaks),
        "variance": variance,
    }


def detect_phases(data: np.ndarray, n_clusters: int = 2, method: str = 'kmeans',
                  radial_bins: int | None = 150, verbose: bool = False) -> dict[str, np.ndarray]:
    """Detect second‑phase regions in a 4D‑STEM dataset using unsupervised clustering.

    Parameters
    ----------
    data : np.ndarray
        4D array of shape (Ny, Nx, P, P) containing diffraction patterns for
        each scan position.
    n_clusters : int, optional
        Number of clusters (phases) to identify.  Default is 2 (matrix and
        precipitate).
    method : {'kmeans', 'gmm'}, optional
        Clustering algorithm to use.  'kmeans' employs k‑means clustering,
        while 'gmm' uses a Gaussian mixture model.
    radial_bins : int or None, optional
        Number of radial bins for the fingerprint.  If None, a sensible
        default based on the pattern size is used.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    dict
        Dictionary containing the cluster labels (flattened), the feature
        matrix used for clustering, and the radial fingerprints.  Keys:

        - 'labels': array of length Ny*Nx with cluster assignments for each
          scan position.
        - 'features': 2D array of shape (Ny*Nx, M) containing feature vectors.
        - 'radial_profiles': 2D array of shape (Ny*Nx, radial_bins) containing
          radial fingerprints.

    Notes
    -----
    The algorithm processes each diffraction pattern independently, computes
    its radial profile and auxiliary scalar features, concatenates them into a
    feature vector, and applies clustering.  For k‑means, the cluster with
    higher average total intensity and variance is heuristically designated
    as the precipitate phase.  For GMM, the cluster with the larger mean of
    the total intensity dimension is taken as the precipitate.

    The output labels are arbitrary integer identifiers.  In simulations, one
    may compare these labels to a ground truth mask.  For real datasets,
    further analysis (e.g. mapping cluster labels back to real space) is
    required to interpret the results.
    """
    ny, nx, py, px = data.shape
    n_positions = ny * nx
    # Preallocate arrays
    profiles = []
    scalar_features = []
    # Flatten scanning grid for easier iteration
    for j in range(ny):
        for i in range(nx):
            pattern = data[j, i]
            # Background subtraction
            corrected = background_subtract(pattern)
            # Estimate centre and compute radial profile
            com = estimate_center_of_mass(corrected)
            radial = radial_profile(corrected, center=com, radial_bins=radial_bins)
            profiles.append(radial)
            # Extract scalar descriptors
            feats = extract_features(radial)
            scalar_features.append([feats['total'], feats['max'], feats['argmax'],
                                    feats['n_peaks'], feats['variance']])
    profiles = np.array(profiles)
    scalar_features = np.array(scalar_features)
    # Combine radial fingerprints and scalar descriptors into a feature matrix
    feature_matrix = np.hstack([profiles, scalar_features])
    # Normalize features to unit variance to avoid scale dominance
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-8
    feature_norm = (feature_matrix - feature_mean) / feature_std
    if verbose:
        print(f"Clustering {n_positions} patterns into {n_clusters} clusters using {method}")
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(feature_norm)
        # Heuristically designate precipitate cluster based on intensity & variance
        # Compute average total intensity and variance per cluster
        cluster_stats = []
        for c in range(n_clusters):
            mask = labels == c
            mean_total = scalar_features[mask, 0].mean() if np.any(mask) else 0
            mean_var = scalar_features[mask, 4].mean() if np.any(mask) else 0
            cluster_stats.append((mean_total, mean_var))
        # Sort clusters by descending intensity and variance; designate label '1' as precipitate
        sorted_indices = np.argsort([-stat[0] for stat in cluster_stats])
        # Map labels to 0/1 for matrix/precipitate if n_clusters==2
        if n_clusters == 2:
            primary, secondary = sorted_indices[0], sorted_indices[1]
            new_labels = np.zeros_like(labels)
            new_labels[labels == primary] = 1  # label 1 = precipitate
            labels = new_labels
    elif method == 'gmm':
        if GaussianMixture is None:
            raise ImportError(
                "GaussianMixture is not available in this version of scikit‑learn. "
                "Use method='kmeans' or install a version of scikit‑learn that provides GaussianMixture."
            )
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
        labels = gmm.fit_predict(feature_norm)
        if n_clusters == 2:
            # Heuristic as above using mean of the total intensity dimension
            cluster_means = []
            for c in range(n_clusters):
                mask = labels == c
                mean_total = scalar_features[mask, 0].mean() if np.any(mask) else 0
                cluster_means.append(mean_total)
            # Precipitate cluster has higher mean total intensity
            precip_label = int(np.argmax(cluster_means))
            labels = (labels == precip_label).astype(int)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    return {
        'labels': labels,
        'features': feature_norm,
        'radial_profiles': profiles,
    }


def simulate_diffraction_pattern(pattern_size: int = 128, phase: str = 'matrix',
                                peaks: list[float] | None = None,
                                amplitude: float = 1.0, noise_level: float = 0.05,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a synthetic diffraction pattern.

    Parameters
    ----------
    pattern_size : int
        Size (pixels) of the square diffraction pattern.
    phase : {'matrix', 'precipitate'}
        Type of phase.  The matrix pattern is amorphous and has a broad
        diffuse ring; the precipitate pattern exhibits discrete Bragg peaks at
        radii specified in `peaks`.
    peaks : list of float or None
        Radii (in pixels) at which to place Bragg peaks for the precipitate
        pattern.  Values should be between 0 and (pattern_size/2).  If None,
        default peaks at 1/6, 1/4 and 1/3 of the maximal radius are used.
    amplitude : float
        Peak amplitude relative to diffuse background.
    noise_level : float
        Standard deviation of additive Gaussian noise relative to the peak
        amplitude.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated diffraction pattern.

    Notes
    -----
    This helper function creates simple synthetic diffraction patterns.  It is
    not a physical simulation but captures basic features: diffuse scattering
    for the matrix, and sharp Bragg peaks for the precipitate.  The peaks are
    generated as concentric rings whose intensity falls off as a Gaussian
    function of radial distance from the peak centre.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Create coordinate grid centered at the middle of the pattern
    coords = np.indices((pattern_size, pattern_size))
    y = coords[0] - pattern_size / 2
    x = coords[1] - pattern_size / 2
    r = np.sqrt(x**2 + y**2)
    max_r = pattern_size / 2
    pattern = np.zeros((pattern_size, pattern_size), dtype=float)
    if phase == 'matrix':
        # Diffuse amorphous pattern: broad Gaussian ring around a characteristic
        # radius (e.g. 1/3 of max radius)
        ring_radius = max_r * 0.35
        ring_width = max_r * 0.05
        pattern += np.exp(-((r - ring_radius)**2) / (2 * ring_width**2))
    elif phase == 'precipitate':
        if peaks is None:
            peaks = [max_r * 0.2, max_r * 0.35, max_r * 0.5]
        for peak_radius in peaks:
            width = max_r * 0.02
            pattern += amplitude * np.exp(-((r - peak_radius)**2) / (2 * width**2))
    else:
        raise ValueError("phase must be 'matrix' or 'precipitate'")
    # Add random noise
    pattern += noise_level * rng.standard_normal(pattern.shape)
    # Ensure non-negative intensities
    pattern[pattern < 0] = 0.0
    return pattern


def simulate_4dstem_dataset(ny: int = 32, nx: int = 32, pattern_size: int = 128,
                            precipitate_fraction: float = 0.1, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a 4D‑STEM dataset containing matrix and precipitate regions.

    Parameters
    ----------
    ny, nx : int
        Dimensions of the scanning grid (number of probe positions along y and x).
    pattern_size : int
        Size of each diffraction pattern (pixels).
    precipitate_fraction : float
        Fraction of scan positions occupied by precipitates (between 0 and 1).
    seed : int or None
        Seed for the random number generator.

    Returns
    -------
    (data, mask)
        A tuple where `data` is a 4D array of shape (ny, nx, pattern_size, pattern_size)
        containing simulated diffraction patterns, and `mask` is a 2D boolean
        array of shape (ny, nx) indicating the ground truth precipitate positions
        (True for precipitates, False for matrix).

    Notes
    -----
    The simulated dataset populates the scanning grid randomly with precipitate
    patterns according to `precipitate_fraction`.  Each pattern is generated
    using `simulate_diffraction_pattern`.  The resulting dataset can be used
    to evaluate the clustering performance of the detection pipeline.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((ny, nx, pattern_size, pattern_size), dtype=float)
    mask = rng.random((ny, nx)) < precipitate_fraction
    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                data[j, i] = simulate_diffraction_pattern(pattern_size, phase='precipitate', rng=rng)
            else:
                data[j, i] = simulate_diffraction_pattern(pattern_size, phase='matrix', rng=rng)
    return data, mask


# -----------------------------------------------------------------------------
# Advanced feature calculations for phase discrimination
# -----------------------------------------------------------------------------
def virtual_bright_dark_field(pattern: np.ndarray,
                              center: tuple[float, float] | None = None,
                              bright_outer: float = 0.1,
                              df_inner: float = 0.3,
                              df_outer: float = 0.6) -> dict[str, float]:
    """Compute virtual bright‑field and dark‑field intensities.

    A 4D‑STEM dataset allows arbitrary virtual apertures to be applied
    post‑acquisition【603835949797498†L58-L61】.  Integrating the scattering
    signal within a central disk yields a virtual bright‑field (BF) image
    while integrating over an annular region at higher scattering angles
    yields a virtual dark‑field (DF) image【603835949797498†L58-L61】.  The
    radii specified here are fractions of the maximum radial distance in
    the diffraction pattern.

    Parameters
    ----------
    pattern : np.ndarray
        2D diffraction pattern (background subtracted).
    center : tuple of float, optional
        Beam centre; if None, the centre of mass is used.
    bright_outer : float
        Fraction of the maximum radius defining the outer radius of the
        bright‑field aperture.  Pixels with radial distance less than
        `bright_outer * max_r` contribute to the BF intensity.
    df_inner : float
        Inner radius of the dark‑field annulus as a fraction of the
        maximum radius.  Pixels with radial distance greater than
        `df_inner * max_r` and less than `df_outer * max_r` contribute
        to the DF intensity.
    df_outer : float
        Outer radius of the dark‑field annulus as a fraction of the
        maximum radius.

    Returns
    -------
    dict
        Keys include 'bf' (sum of intensities within the bright‑field
        aperture), 'df' (sum within the dark‑field annulus), and
        'bf_df_ratio' (ratio of DF to BF intensities).

    Notes
    -----
    The integration ranges chosen here are reasonable defaults for
    discriminating diffuse matrix scattering from the sharp Bragg peaks of
    precipitates.  Users may tune these radii for their particular
    dataset.  Intensities are summed (not averaged) to preserve the
    relative weight of scattering contributions.
    """
    ny, nx = pattern.shape
    # Determine centre for radial coordinates
    if center is None:
        y0, x0 = estimate_center_of_mass(pattern)
    else:
        y0, x0 = center
    # Compute radial distances
    y_indices, x_indices = np.indices(pattern.shape)
    r = np.sqrt((y_indices - y0)**2 + (x_indices - x0)**2)
    max_r = r.max()
    # Masks for BF and DF regions
    bf_mask = r <= (bright_outer * max_r)
    df_mask = (r >= (df_inner * max_r)) & (r <= (df_outer * max_r))
    # Integrated intensities
    bf_intensity = float(np.sum(pattern[bf_mask]))
    df_intensity = float(np.sum(pattern[df_mask]))
    ratio = df_intensity / (bf_intensity + 1e-12)
    return {
        'bf': bf_intensity,
        'df': df_intensity,
        'bf_df_ratio': ratio,
    }


def anisotropy_features(pattern: np.ndarray,
                        center: tuple[float, float] | None = None,
                        n_sectors: int = 8,
                        radial_cutoff: float = 0.8) -> dict[str, float]:
    """Compute simple metrics of angular anisotropy in a diffraction pattern.

    Diffraction patterns from crystalline precipitates often exhibit
    anisotropic intensity distributions because of their orientation and
    discrete Bragg reflections.  By partitioning the pattern into angular
    sectors and comparing the mean intensity in each sector, one can
    quantify the degree of anisotropy.  Higher anisotropy values may
    indicate ordered phases, whereas amorphous matrix patterns should be
    relatively isotropic.

    Parameters
    ----------
    pattern : np.ndarray
        2D diffraction pattern (background subtracted).
    center : tuple of float, optional
        Beam centre; if None, the centre of mass is used.
    n_sectors : int
        Number of angular sectors to divide the pattern into (default 8).
    radial_cutoff : float
        Fraction of the maximum radius beyond which pixels are ignored.  A
        cutoff less than 1.0 excludes the noisier high‑angle tail.

    Returns
    -------
    dict
        Contains 'anisotropy_std' (standard deviation of sector means
        relative to their mean) and 'anisotropy_ratio' (difference between
        maximum and minimum sector means normalised by the mean).

    Notes
    -----
    The metric is dimensionless.  If all sectors have identical mean
    intensity the metrics will be zero.  Patterns with strong directional
    scattering will give larger values.  Because this calculation uses
    azimuthal segmentation, it is sensitive to experimental alignment and
    sampling noise; it should be used in combination with other features
    when clustering.
    """
    ny, nx = pattern.shape
    if center is None:
        y0, x0 = estimate_center_of_mass(pattern)
    else:
        y0, x0 = center
    # Compute relative coordinates
    y_indices, x_indices = np.indices(pattern.shape)
    dy = y_indices - y0
    dx = x_indices - x0
    r = np.sqrt(dx**2 + dy**2)
    max_r = r.max()
    # Mask for radial cutoff
    radial_mask = r <= (radial_cutoff * max_r)
    # Angles in the range [0, 2*pi)
    angles = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    sector_means = []
    # Compute mean intensity in each sector
    for i in range(n_sectors):
        angle_min = (2 * np.pi / n_sectors) * i
        angle_max = angle_min + (2 * np.pi / n_sectors)
        sector_mask = radial_mask & (angles >= angle_min) & (angles < angle_max)
        # Avoid empty sectors
        if np.any(sector_mask):
            sector_sum = np.sum(pattern[sector_mask])
            sector_mean = sector_sum / float(np.sum(sector_mask))
        else:
            sector_mean = 0.0
        sector_means.append(sector_mean)
    sector_means = np.array(sector_means)
    mean_intensity = np.mean(sector_means) + 1e-12
    anisotropy_std = float(np.std(sector_means) / mean_intensity)
    anisotropy_ratio = float((np.max(sector_means) - np.min(sector_means)) / mean_intensity)
    return {
        'anisotropy_std': anisotropy_std,
        'anisotropy_ratio': anisotropy_ratio,
    }


def com_shift_features(pattern: np.ndarray,
                       center: tuple[float, float] | None = None) -> dict[str, float]:
    """Compute centre‑of‑mass (CoM) shift metrics.

    The centre of mass of a diffraction pattern reflects the net momentum
    transfer during electron scattering【603835949797498†L68-L70】.  In differential
    phase contrast imaging (DPC), CoM maps are related to the gradient
    of the electrostatic potential.  Here we include a simple measure of
    the CoM shift magnitude and its direction encoded as sine and cosine.

    Parameters
    ----------
    pattern : np.ndarray
        2D diffraction pattern (background subtracted).
    center : tuple of float, optional
        Nominal beam centre; if None, the geometric centre of the pattern
        (midpoint of its dimensions) is used.

    Returns
    -------
    dict
        Contains 'com_shift' (magnitude of the CoM vector divided by the
        pattern radius) and 'com_sin'/'com_cos' (sine and cosine of the
        CoM vector angle).

    Notes
    -----
    A non‑zero CoM shift indicates momentum transfer due to local
    electromagnetic fields or sample tilt.  While this feature alone
    does not discriminate phases, combining it with other descriptors can
    improve clustering in some datasets.
    """
    ny, nx = pattern.shape
    # Compute CoM of the pattern
    com_y, com_x = estimate_center_of_mass(pattern)
    if center is None:
        # Use geometric centre as reference
        ref_y, ref_x = ny / 2.0, nx / 2.0
    else:
        ref_y, ref_x = center
    # Compute vector from reference centre to CoM
    dy = com_y - ref_y
    dx = com_x - ref_x
    # Magnitude normalised by maximum possible radius
    max_r = np.sqrt((ny / 2.0)**2 + (nx / 2.0)**2)
    shift_mag = np.sqrt(dx**2 + dy**2) / (max_r + 1e-12)
    # Encode direction using sine and cosine to avoid discontinuities
    angle = atan2(dy, dx)
    return {
        'com_shift': float(shift_mag),
        'com_sin': float(sin(angle)),
        'com_cos': float(cos(angle)),
    }


def detect_phases_multi(
    data: np.ndarray,
    n_clusters: int = 2,
    method: str = 'kmeans',
    radial_bins: int | None = 150,
    feature_set: str | list[str] = 'all',
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Detect second‑phase regions using multiple physics‑based feature sets.

    This function extends :func:`detect_phases` by allowing the user to
    choose which features to include in the clustering.  In addition to
    radial fingerprints and their scalar descriptors, one can include
    virtual bright‑/dark‑field intensities, anisotropy metrics, and
    centre‑of‑mass shift features.  The resulting feature matrix is
    normalised and clustered using k‑means or a Gaussian mixture model.

    Parameters
    ----------
    data : np.ndarray
        4D array of shape (Ny, Nx, P, P) containing diffraction patterns.
    n_clusters : int, optional
        Number of clusters (phases) to identify.  Default is 2 (matrix
        and precipitate).
    method : {'kmeans', 'gmm'}, optional
        Clustering algorithm to use.  'kmeans' employs k‑means
        clustering, while 'gmm' uses a Gaussian mixture model.  See
        :func:`detect_phases` for details.
    radial_bins : int or None, optional
        Number of radial bins for the fingerprint.  Used only if
        radial features are included.  If None, a sensible default
        based on the pattern size is used.
    feature_set : str or list of str, optional
        Which groups of features to use.  Acceptable values are:

          - 'radial': radial fingerprints and scalar descriptors.
          - 'bf_df': virtual bright‑field and dark‑field intensities.
          - 'anisotropy': angular anisotropy metrics.
          - 'com': centre‑of‑mass shift metrics.
          - 'all': include all of the above.

        A list of strings may be provided to include multiple groups.  If
        a string other than these is supplied, it will be ignored.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    dict
        Dictionary containing the cluster labels (flattened), the
        normalised feature matrix, the list of feature names, and a
        mapping from index in the feature matrix to feature name.

    Notes
    -----
    The clustering step uses the same heuristics as :func:`detect_phases`
    to assign the cluster corresponding to precipitates.  If radial
    features are present, the total intensity is used to decide which
    cluster corresponds to the precipitate.  Otherwise the dark‑field
    intensity or anisotropy ratio is used.  When none of these
    descriptors are available, cluster labels are returned as is.
    """
    # Determine which feature groups to include
    if isinstance(feature_set, str):
        if feature_set == 'all':
            features_requested = ['radial', 'bf_df', 'anisotropy', 'com']
        else:
            features_requested = [feature_set]
    else:
        features_requested = list(feature_set)

    ny, nx, py, px = data.shape
    n_positions = ny * nx
    feature_vectors = []
    feature_names = []  # names for each column in the feature matrix
    # Precompute radial bin count if not provided
    if radial_bins is None:
        radial_bins = int(max(py, px) / 2)
    # Iterate through scanning positions
    for j in range(ny):
        for i in range(nx):
            pattern = data[j, i]
            # Background subtraction
            corrected = background_subtract(pattern)
            # Determine centre of mass once to reuse across features
            com = estimate_center_of_mass(corrected)
            row_values: list[float] = []
            row_names: list[str] = []
            # Radial features: fingerprint + scalar descriptors
            if 'radial' in features_requested:
                radial = radial_profile(corrected, center=com, radial_bins=radial_bins)
                feats = extract_features(radial)
                # Append radial fingerprint values
                row_values.extend(radial.tolist())
                row_names.extend([f'radial_{k}' for k in range(len(radial))])
                # Append scalar descriptors
                for key in ['total', 'max', 'argmax', 'n_peaks', 'variance']:
                    row_values.append(float(feats[key]))
                    row_names.append(key)
            # Bright/dark field features
            if 'bf_df' in features_requested:
                bf_feats = virtual_bright_dark_field(corrected, center=com)
                for key in ['bf', 'df', 'bf_df_ratio']:
                    row_values.append(float(bf_feats[key]))
                    row_names.append(key)
            # Anisotropy features
            if 'anisotropy' in features_requested:
                ani_feats = anisotropy_features(corrected, center=com)
                for key in ['anisotropy_std', 'anisotropy_ratio']:
                    row_values.append(float(ani_feats[key]))
                    row_names.append(key)
            # Centre of mass shift features
            if 'com' in features_requested:
                com_feats = com_shift_features(corrected, center=None)
                for key in ['com_shift', 'com_sin', 'com_cos']:
                    row_values.append(float(com_feats[key]))
                    row_names.append(key)
            feature_vectors.append(row_values)
            # Save names only on first iteration
            if not feature_names:
                feature_names = row_names
    # Convert to array
    feature_matrix = np.array(feature_vectors)
    # Normalise features to zero mean and unit variance
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-8
    feature_norm = (feature_matrix - feature_mean) / feature_std
    if verbose:
        print(f"Clustering {n_positions} patterns into {n_clusters} clusters using {method} with features {features_requested}")
    # Perform clustering
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(feature_norm)
    elif method == 'gmm':
        if GaussianMixture is None:
            raise ImportError(
                "GaussianMixture is not available in this version of scikit‑learn. "
                "Use method='kmeans' or install a version of scikit‑learn that provides GaussianMixture."
            )
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
        labels = gmm.fit_predict(feature_norm)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    # Heuristically assign the precipitate cluster
    # Determine candidate descriptor for cluster assignment
    descriptor_candidates = []
    for desc in ['total', 'df', 'bf_df_ratio', 'anisotropy_ratio']:
        if desc in feature_names:
            descriptor_candidates.append(desc)
    precip_label = None
    if descriptor_candidates:
        # Use the first available descriptor
        desc = descriptor_candidates[0]
        idx = feature_names.index(desc)
        cluster_means = []
        for c in range(n_clusters):
            mask = labels == c
            mean_val = feature_matrix[mask, idx].mean() if np.any(mask) else -np.inf
            cluster_means.append(mean_val)
        # Cluster with highest mean is precipitate
        precip_label = int(np.argmax(cluster_means))
        # Remap to 0/1 if two clusters
        if n_clusters == 2:
            labels = (labels == precip_label).astype(int)
    # Return results
    return {
        'labels': labels,
        'features': feature_norm,
        'feature_names': feature_names,
    }