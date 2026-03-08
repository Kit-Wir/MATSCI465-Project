# fp_suppression.py
# False-positive suppression wrapper for pipelinefinalproject.detect_phases_multi
#
# Key idea:
# - Run k=2 clustering once to get features + labels
# - Decide "no precip" if (a) clusters are too balanced OR (b) k=1 fits better (GMM BIC) OR (c) separation is weak
# - If precip exists, choose precip cluster as the *smaller* cluster (default) and optionally apply spatial cleanup

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

import pipelinefinalproject as pf

try:
    from sklearn.mixture import GaussianMixture
except Exception:
    GaussianMixture = None

try:
    from sklearn.metrics import silhouette_score
except Exception:
    silhouette_score = None

try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None


@dataclass
class FPSuppressConfig:
    # k-selection / suppression rules
    max_minor_frac: float = 0.20       # if min(cluster_frac) > this => too balanced => no precip
    use_bic: bool = True               # only applies when method="gmm"
    bic_delta: float = 0.0             # require bic1 <= bic2 - bic_delta to choose k=1
    sil_min: float = 0.05              # for method="kmeans": if silhouette < sil_min => no precip

    # precip cluster selection when k=2 accepted
    precip_rule: str = "smaller"       # "smaller" or "label1"

    # optional spatial cleanup
    median_size: int = 0               # 0 disables; otherwise e.g. 3
    min_component_size: int = 0        # 0 disables; otherwise remove tiny blobs


def _bic_choose_k(feature_norm: np.ndarray, random_state: int = 0) -> Tuple[int, float, float]:
    """Return (k_selected, bic_k1, bic_k2) using GMM BIC on feature_norm."""
    if GaussianMixture is None:
        return 2, float("nan"), float("nan")

    g1 = GaussianMixture(n_components=1, covariance_type="full", random_state=random_state)
    g2 = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)

    g1.fit(feature_norm)
    g2.fit(feature_norm)

    bic1 = float(g1.bic(feature_norm))
    bic2 = float(g2.bic(feature_norm))

    ksel = 1 if bic1 <= bic2 else 2
    return ksel, bic1, bic2


def _silhouette(feature_norm: np.ndarray, labels01: np.ndarray) -> float:
    """Silhouette score for 2 clusters; returns nan if unavailable/degenerate."""
    if silhouette_score is None:
        return float("nan")
    labs = labels01.reshape(-1)
    if len(np.unique(labs)) < 2:
        return float("nan")
    # silhouette requires >= 2 samples per cluster
    if np.min(np.bincount(labs.astype(int))) < 2:
        return float("nan")
    return float(silhouette_score(feature_norm, labs))


def _choose_precip_label(labels01: np.ndarray, rule: str) -> int:
    """Choose which label (0 or 1) corresponds to precipitate."""
    if rule == "label1":
        return 1
    # default: "smaller"
    frac1 = float(labels01.mean())
    frac0 = 1.0 - frac1
    return 1 if frac1 <= frac0 else 0


def _spatial_cleanup(mask01: np.ndarray, median_size: int, min_component_size: int) -> np.ndarray:
    """Optional median filter + remove small connected components."""
    if ndi is None:
        return mask01

    out = mask01.astype(np.uint8)

    if median_size and median_size > 1:
        out = ndi.median_filter(out, size=median_size).astype(np.uint8)

    if min_component_size and min_component_size > 0:
        lab, nlab = ndi.label(out)
        if nlab > 0:
            counts = np.bincount(lab.ravel())
            # counts[0] is background
            keep = np.zeros_like(counts, dtype=bool)
            keep[1:] = counts[1:] >= min_component_size
            out = keep[lab].astype(np.uint8)

    return out


def detect_fp_suppressed(
    data4d: np.ndarray,
    *,
    method: str = "gmm",
    radial_bins: int = 40,
    feature_set: str | list[str] = "all",
    cfg: Optional[FPSuppressConfig] = None,
    random_state: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Wrapper around pf.detect_phases_multi that suppresses false positives.

    Returns dict with:
      - pred_mask: (ny,nx) uint8, final precip mask (0/1)
      - labels_raw: (ny,nx) uint8, raw 0/1 labels from pf (when n_clusters=2)
      - k_selected: 1 or 2
      - decision: short string explanation
      - diagnostics: bic1/bic2/silhouette/minor_frac/feature_set/method
    """
    if cfg is None:
        cfg = FPSuppressConfig()

    ny, nx = data4d.shape[:2]

    # Run once with k=2 to get features + labels
    res2 = pf.detect_phases_multi(
        data4d,
        n_clusters=2,
        method=method,
        radial_bins=radial_bins,
        feature_set=feature_set,
        verbose=verbose,
    )

    labels01_flat = np.asarray(res2["labels"]).astype(np.uint8).reshape(-1)
    labels01_map = labels01_flat.reshape(ny, nx)

    frac1 = float(labels01_flat.mean())
    minor_frac = float(min(frac1, 1.0 - frac1))

    decision = "k=2"
    ksel = 2

    bic1 = bic2 = float("nan")
    sil = float("nan")

    # Rule 1: too balanced => likely "forced clustering" false positive
    if minor_frac > cfg.max_minor_frac:
        ksel = 1
        decision = f"no-precip (balanced clusters, minor_frac={minor_frac:.3f})"

    # Rule 2: model selection
    if ksel == 2:
        if method == "gmm" and cfg.use_bic:
            k_bic, bic1, bic2 = _bic_choose_k(res2["features"], random_state=random_state)
            # require k=1 to be meaningfully better if bic_delta > 0
            if k_bic == 1 and (bic2 - bic1) >= cfg.bic_delta:
                ksel = 1
                decision = f"no-precip (BIC prefers k=1, bic2-bic1={bic2-bic1:.2f})"
        elif method == "kmeans":
            sil = _silhouette(res2["features"], labels01_flat)
            if np.isfinite(sil) and sil < cfg.sil_min:
                ksel = 1
                decision = f"no-precip (low silhouette={sil:.3f})"

    if ksel == 1:
        pred = np.zeros((ny, nx), dtype=np.uint8)
    else:
        precip_lab = _choose_precip_label(labels01_map, cfg.precip_rule)
        pred = (labels01_map == precip_lab).astype(np.uint8)
        pred = _spatial_cleanup(pred, cfg.median_size, cfg.min_component_size)

    return {
        "pred_mask": pred,
        "labels_raw": labels01_map.astype(np.uint8),
        "k_selected": int(ksel),
        "decision": decision,
        "diagnostics": {
            "method": method,
            "feature_set": feature_set,
            "radial_bins": int(radial_bins),
            "minor_frac": float(minor_frac),
            "frac_label1": float(frac1),
            "bic1": bic1,
            "bic2": bic2,
            "silhouette": sil,
            "cfg": cfg.__dict__,
        },
    }