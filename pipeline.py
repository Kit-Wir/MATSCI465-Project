import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def radial_fingerprint_features(data4d, nbins=20, center=None, eps=1e-8):
    ny, nx, ky, kx = data4d.shape
    
    if center is None:
        cy, cx = ky // 2, kx // 2
    else:
        cy, cx = center

    Y, X = np.ogrid[:ky, :kx]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)

    edges = np.linspace(0, R.max(), nbins + 1)
    masks = [(R >= edges[i]) & (R < edges[i+1]) for i in range(nbins)]

    rad = np.stack([data4d[..., m].sum(axis=-1) for m in masks], axis=-1)  # (ny,nx,nbins)

    total = data4d.sum(axis=(-2, -1)) + eps
    rad_norm = rad / total[..., None]

    return rad_norm.reshape(ny * nx, nbins), rad_norm  # flat + image form


def run_pipeline(
    data4d,
    nbins=20,
    exclude_low_q=0,
    n_pca=10,
    pca_use=5,
    method="kmeans",
    k=2,
    random_state=0,
):
    ny, nx, ky, kx = data4d.shape

    Xflat, Ximg = radial_fingerprint_features(data4d, nbins=nbins)

    if exclude_low_q > 0:
        Xflat = Xflat[:, exclude_low_q:]
        Ximg  = Ximg[..., exclude_low_q:]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xflat)

    pca = PCA(n_components=n_pca, random_state=random_state)
    Z = pca.fit_transform(Xs)
    Z_use = Z[:, :pca_use]

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(Z_use)
    elif method == "gmm":
        model = GaussianMixture(n_components=k, random_state=random_state)
        labels = model.fit_predict(Z_use)
    else:
        raise ValueError("method must be 'kmeans' or 'gmm'")

    return {
        "labels": labels.reshape(ny, nx),
        "labels_flat": labels,
        "Xfeat_flat": Xflat,
        "Xfeat_img": Ximg,
        "Z": Z,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "model": model,
        "pca": pca,
        "scaler": scaler,
    }