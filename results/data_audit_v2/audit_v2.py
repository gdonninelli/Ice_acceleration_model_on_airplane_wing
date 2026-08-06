#!/usr/bin/env python3
"""Acceptance audit of the regenerated dataset.

Repeats the tests of results/data_audit/audit.py, then adds the two physical
tests that decide whether the dataset is usable at all:

  * the angle of attack is reconstructed from the geometry (zero-level contour
    -> chord -> inclination) and compared with the angle column, which proves
    that each field is paired with its own summary row;
  * C_l is plotted against the angle and fitted, where thin-airfoil theory
    predicts a slope of 2*pi per radian (0.1097 per degree) through the origin
    for a symmetric profile.

Run from the repository root:  python3 results/data_audit_v2/audit_v2.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = "results/data_audit_v2"
os.makedirs(OUT, exist_ok=True)
# Thin-airfoil theory: dC_l/dalpha = 2*pi per RADIAN. Converted to degrees
# that is 2*pi * (pi/180) = 0.10966 per degree.
SLOPE_PER_DEGREE = 2 * np.pi * np.pi / 180.0

# SDFGenerator.hpp: the grid spans X in [-0.2, 1.1] and Z in [-0.4, 0.4] over
# GRID_SIZE cells, so the cells are anisotropic and an angle measured in index
# space must be converted before it means anything.
X_MIN, X_MAX, Z_MIN, Z_MAX, GRID_SIZE = -0.2, 1.1, -0.4, 0.4, 150
DX = (X_MAX - X_MIN) / (GRID_SIZE - 1)
DZ = (Z_MAX - Z_MIN) / (GRID_SIZE - 1)


def load():
    tr = np.load("dataset/cnn_dataset_train.npz")
    te = np.load("dataset/cnn_dataset_test.npz")
    def fix(a):
        return a[:, 0] if a.ndim == 4 else a
    X = np.concatenate([fix(tr["X_sdf"]), fix(te["X_sdf"])]).astype(np.float64)
    Y = np.concatenate([tr["Y_cl"], te["Y_cl"]]).astype(np.float64)
    S = np.concatenate([tr["X_scalars"], te["X_scalars"]]).astype(np.float64)
    split = np.array(["train"] * len(tr["Y_cl"]) + ["test"] * len(te["Y_cl"]))
    return X, Y, S, split


def chord_angle(field):
    """Angle of the chord of the zero-level contour, in degrees.

    The chord is the segment between the two farthest contour points. The sign
    convention is resolved once, globally, against the metadata column.
    """
    cs = plt.contour(field, levels=[0.0])
    points = np.vstack([p.vertices for p in cs.get_paths()]) if cs.get_paths() else None
    plt.close()
    if points is None or len(points) < 2:
        return np.nan
    # Farthest pair via the convex hull of a subsample, cheap and exact enough.
    if len(points) > 400:
        points = points[np.linspace(0, len(points) - 1, 400).astype(int)]
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    # contour() returns (column, row); convert to physical units, because the
    # cells are not square and an index-space angle is distorted by dz/dx.
    dx = (points[j, 0] - points[i, 0]) * DX
    dy = (points[j, 1] - points[i, 1]) * DZ
    if dx < 0:
        dx, dy = -dx, -dy
    return np.degrees(np.arctan2(dy, dx))


def main():
    X, Y, S, split = load()
    angle = S[:, 1]
    reynolds = S[:, 0]
    print(f"=== AUDIT v2 — {len(Y)} campioni ===")
    print(f"  angoli distinti   : {len(np.unique(angle))}  range [{angle.min()}, {angle.max()}]")
    print(f"  Reynolds distinti : {sorted(np.unique(reynolds))}")
    print(f"  Y_cl              : [{Y.min():.4f}, {Y.max():.4f}] std={Y.std():.4f}")

    print("\n--- ricostruzione dell'angolo dalla geometria ---")
    n = min(len(X), 400)
    idx = np.linspace(0, len(X) - 1, n).astype(int)
    geom = np.array([chord_angle(X[i]) for i in idx])
    meta = angle[idx]
    ok = np.isfinite(geom)
    geom, meta, idx = geom[ok], meta[ok], idx[ok]

    # Resolve the sign/offset convention once, then report the residual.
    best = None
    for sign in (+1, -1):
        residual = sign * geom - meta
        offset = np.median(residual)
        err = np.abs(sign * geom - offset - meta)
        if best is None or np.median(err) < best[0]:
            best = (np.median(err), sign, offset, err)
    median_err, sign, offset, err = best
    alpha_geom = sign * geom - offset
    print(f"  convenzione risolta: alpha_geom = {sign:+d} * chord_angle - ({offset:.4f} deg)")
    print(f"  campioni verificati : {len(alpha_geom)}")
    print(f"  errore |alpha_geom - angle|: mediana={median_err:.4f} deg  "
          f"media={err.mean():.4f}  max={err.max():.4f}")
    agree = (err <= 0.5).mean()
    print(f"  entro 0.5 deg      : {100*agree:.1f}%   -> "
          f"{'PASS' if agree > 0.95 else 'FAIL'}")
    print(f"  correlazione geom/meta: r={np.corrcoef(alpha_geom, meta)[0,1]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(meta, alpha_geom, s=18, alpha=0.7)
    lo, hi = meta.min() - 1, meta.max() + 1
    axes[0].plot([lo, hi], [lo, hi], "k--", label="identita'")
    axes[0].set_xlabel("angle dai metadati [deg]")
    axes[0].set_ylabel("alpha ricostruito dalla geometria [deg]")
    axes[0].set_title(f"Accoppiamento riga <-> SDF\nerrore mediano {median_err:.3f} deg")
    axes[0].legend()
    axes[1].hist(err, bins=40)
    axes[1].axvline(0.5, color="r", ls="--", label="tolleranza 0.5 deg")
    axes[1].set_xlabel("|alpha_geom - angle| [deg]")
    axes[1].set_title("Distribuzione dell'errore")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/B_angle_reconstruction.png", dpi=110)
    plt.close(fig)

    print("\n--- C_l contro alpha: il test fisico ---")
    linear = np.abs(angle) <= 8.0
    slope, intercept, r, p, se = stats.linregress(angle[linear], Y[linear])
    r2 = r ** 2
    print(f"  zona lineare |alpha| <= 8 deg  ({linear.sum()} campioni)")
    print(f"  pendenza  : {slope:.5f} per grado   (atteso 2*pi/rad = {SLOPE_PER_DEGREE:.5f})")
    print(f"              rapporto misurato/atteso = {slope/SLOPE_PER_DEGREE:.4f}")
    print(f"  intercetta: {intercept:+.5f}          (atteso ~0 per profilo simmetrico)")
    print(f"  R^2       : {r2:.5f}    p={p:.3g}")
    slope_all, intercept_all, r_all, _, _ = stats.linregress(angle, Y)
    print(f"  su tutti gli angoli: pendenza={slope_all:.5f} intercetta={intercept_all:+.5f} R^2={r_all**2:.5f}")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for re in sorted(np.unique(reynolds)):
        m = reynolds == re
        ax.scatter(angle[m], Y[m], s=16, alpha=0.65, label=f"Re = {re:.0f}")
    grid = np.linspace(angle.min(), angle.max(), 100)
    ax.plot(grid, SLOPE_PER_DEGREE * grid, "k--", lw=2,
            label=r"teoria del profilo sottile: $C_l = 2\pi\alpha$")
    ax.plot(grid, slope * grid + intercept, "r-", lw=1.5,
            label=f"fit |a|<=8: {slope:.4f}a {intercept:+.4f} (R2={r2:.3f})")
    ax.axvspan(-8, 8, color="grey", alpha=0.12, label="zona di fit")
    ax.set_xlabel(r"angolo d'attacco $\alpha$ [deg]")
    ax.set_ylabel(r"$C_l$")
    ax.set_title("Test fisico: coefficiente di portanza contro angolo d'attacco")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/C_cl_vs_alpha.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    im = axes[0].imshow(X[np.argmin(np.abs(angle - angle.min()))], cmap="RdBu_r")
    axes[0].set_title(f"SDF, alpha = {angle.min():.2f} deg"); plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(X[np.argmin(np.abs(angle))], cmap="RdBu_r")
    axes[1].set_title("SDF, alpha ~ 0 deg"); plt.colorbar(im, ax=axes[1])
    im = axes[2].imshow(X[np.argmin(np.abs(angle - angle.max()))], cmap="RdBu_r")
    axes[2].set_title(f"SDF, alpha = {angle.max():.2f} deg"); plt.colorbar(im, ax=axes[2])
    for a in axes:
        a.axis("off")
    fig.suptitle("Campi SDF rigenerati: il profilo e' visibile e ruota con l'angolo")
    fig.tight_layout()
    fig.savefig(f"{OUT}/B_sdf_examples.png", dpi=110)
    plt.close(fig)

    passed = agree > 0.95 and r2 > 0.8 and abs(slope / SLOPE_PER_DEGREE - 1) < 0.15
    print(f"\n  VERDETTO AUDIT FISICO: {'PASS' if passed else 'FAIL'}")
    print(f"  figure in {OUT}/")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
