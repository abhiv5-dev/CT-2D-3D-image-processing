#!/usr/bin/env python3
"""
Generates:
1. docs/calibration_roi_overlay.png (across 6 tuning slices: 350, 366, 382, 398, 414, 438)
   - Air ROI [30:130, 30:130] in red
   - Morphological specimen mask M_sample (smoothed, Otsu, largest CC, hole-filled, eroded 15px) outline in green
   - Inner core ROI (radius 100px) in gold
   - Per-ray aligned boundary points and arc in orange
   - Reporting mask quality checks (Area, Solidity, Frac<=3px) and erf edge width in title without overlap.

2. docs/calibration_aligned_profiles.png (across 3 tuning slices: 350, 382, 438)
   - 61 individual per-ray aligned profiles (light blue)
   - Aligned average profile (navy bold)
   - Non-linear error-function (erf) fit (crimson dashed)
   - Stating unfiltered width, plateau delta, standard error, and validity ratio.

Skips cleanly (exit 0) if the real dataset is missing.
"""

import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from scipy.ndimage import gaussian_filter, label, binary_fill_holes, binary_erosion, map_coordinates
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from scipy.optimize import curve_fit
from scipy.special import erf

repo_root = Path(__file__).resolve().parent.parent

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def find_dataset(cli_data_dir=None):
    if cli_data_dir:
        cand = Path(cli_data_dir)
        try:
            if cand.is_dir():
                tifs = sorted(list(cand.glob("*.tif")) + list(cand.glob("*.tiff")), key=natural_sort_key)
                if len(tifs) > 0:
                    return cand, tifs
        except (PermissionError, OSError):
            pass
        return None, []

    candidates = []
    env_dir = os.environ.get("CT_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(repo_root / "data" / "real")

    for cand in candidates:
        try:
            if cand.is_dir():
                tifs = sorted(list(cand.glob("*.tif")) + list(cand.glob("*.tiff")), key=natural_sort_key)
                if len(tifs) > 0:
                    return cand, tifs
        except (PermissionError, OSError):
            continue
    return None, []

def load_slice(fpath):
    try:
        sys.path.insert(0, str(repo_root / "build"))
        import ct
        return ct.load_tiff(str(fpath))
    except Exception:
        pass

    import tifffile
    with tifffile.TiffFile(str(fpath)) as tif:
        raw = tif.pages[0].asarray().astype(np.float32)
        desc = tif.pages[0].tags.get("ImageDescription")
        slope, offset = 1.0, 0.0
        if desc:
            desc_str = str(desc.value)
            m_s = re.search(r"slope\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", desc_str)
            m_o = re.search(r"offset\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", desc_str)
            if m_s:
                slope = float(m_s.group(1))
            if m_o:
                offset = float(m_o.group(1))
        return raw * slope + offset

def erf_model(s, a, b, s0, sigma):
    return a + b * erf((s - s0) / (np.sqrt(2.0) * sigma))

def process_slice_edge(img):
    H, W = img.shape
    # 1. Specimen object mask
    smoothed = gaussian_filter(img, 1.5)
    t_otsu = float(threshold_otsu(smoothed))
    binary = smoothed > t_otsu
    lbl, num = label(binary)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    largest_cc = (lbl == np.argmax(counts))
    filled = binary_fill_holes(largest_cc)
    eroded = binary_erosion(filled, iterations=15)

    area = int(np.sum(filled))
    chull = convex_hull_image(filled)
    solidity = float(area / np.sum(chull))
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_contact = int(np.sum(filled & border))

    ys, xs = np.where(filled)
    xc, yc = float(np.mean(xs)), float(np.mean(ys))

    # 2. Per-ray edge search in right sector: [-30 deg, +30 deg]
    thetas = np.radians(np.linspace(-30, 30, 61))
    radii_coarse = np.linspace(150, 380, 461)
    s_relative = np.linspace(-25, 25, 101)

    detected_edges = []
    detected_xy = []
    aligned_profiles = []
    diffs = []

    for t in thetas:
        xr = xc + radii_coarse * np.cos(t)
        yr = yc + radii_coarse * np.sin(t)
        m_vals = map_coordinates(filled.astype(np.float32), [yr, xr], order=0)
        idx_in = np.where(m_vals > 0.5)[0]
        r_mask = radii_coarse[idx_in[-1]] if len(idx_in) > 0 else 270.0

        # Outermost strong gradient crossing around r_mask
        r_fine = np.linspace(r_mask - 12, r_mask + 12, 49)
        xf = xc + r_fine * np.cos(t)
        yf = yc + r_fine * np.sin(t)
        prof_f = map_coordinates(img, [yf, xf], order=1)
        grad_f = -np.gradient(prof_f, r_fine)
        r_edge = float(r_fine[np.argmax(grad_f)])
        detected_edges.append(r_edge)
        detected_xy.append((xc + r_edge * np.cos(t), yc + r_edge * np.sin(t)))
        diffs.append(abs(r_mask - r_edge))

        # Aligned profile centered at r_edge
        r_align = r_edge + s_relative
        xa = xc + r_align * np.cos(t)
        ya = yc + r_align * np.sin(t)
        aligned_profiles.append(map_coordinates(img, [ya, xa], order=1))

    diffs = np.array(diffs)
    frac_within_3px = float(np.mean(diffs <= 3.0))
    mask_fail = (area < 100000) or (solidity < 0.95) or (border_contact > 0) or (frac_within_3px < 0.70)

    aligned_profiles = np.array(aligned_profiles)
    detected_edges = np.array(detected_edges)
    mean_profile = np.mean(aligned_profiles, axis=0)

    # Step contrast: s in [-25, -15] (inside specimen) vs [15, 25] (outside air)
    m1 = np.mean(mean_profile[:20])
    m2 = np.mean(mean_profile[-20:])
    delta = abs(m1 - m2)

    # Spatial block standard error across 10 blocks of 6 rays
    block_means = [np.mean(aligned_profiles[i:i+6], axis=0) for i in range(0, 60, 6)]
    block_deltas = [abs(np.mean(b[:20]) - np.mean(b[-20:])) for b in block_means]
    se = np.std(block_deltas) / np.sqrt(len(block_deltas))
    val = delta / se if se > 0 else 999.0

    p0 = [0.5*(m1+m2), 0.5*(m2-m1), 0.0, 3.0]
    popt, _ = curve_fit(erf_model, s_relative, mean_profile, p0=p0, maxfev=5000)
    sigma_erf = abs(popt[3])
    w_erf = 2.5631 * sigma_erf
    erf_fit_curve = erf_model(s_relative, *popt)

    R_fit = float(np.mean(detected_edges))

    return {
        "xc": xc, "yc": yc, "filled": filled, "eroded": eroded,
        "area": area, "solidity": solidity, "border_contact": border_contact,
        "frac_within_3px": frac_within_3px, "mask_fail": mask_fail,
        "detected_edges": detected_edges, "detected_xy": detected_xy,
        "R_fit": R_fit, "delta": delta, "se": se, "val": val,
        "w_erf": w_erf, "sigma_erf": sigma_erf,
        "s_relative": s_relative, "aligned_profiles": aligned_profiles,
        "mean_profile": mean_profile, "erf_fit_curve": erf_fit_curve
    }

def main():
    parser = argparse.ArgumentParser(description="Generate calibration ROI overlay and aligned profiles figures.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to real CT dataset")
    args = parser.parse_args()

    data_dir, slice_files = find_dataset(args.data_dir)
    if not data_dir or len(slice_files) == 0:
        print("Notice: Real CT dataset not found. Skipping overlay generation cleanly.")
        sys.exit(0)

    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    overlay_indices = [350, 366, 382, 398, 414, 438]
    overlay_indices = [i for i in overlay_indices if i < len(slice_files)]

    res_all = {}
    for idx in overlay_indices:
        fpath = slice_files[idx]
        img = load_slice(fpath)
        res_all[idx] = (img, fpath.name, process_slice_edge(img))

    # 1. Figure 1: Calibration ROI Overlay Figure (6 slices)
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
    axes1 = axes1.ravel()

    for pos, idx in enumerate(overlay_indices):
        img, fname, r = res_all[idx]
        ax = axes1[pos]
        p1 = float(np.percentile(img, 1))
        p99 = float(np.percentile(img, 99))
        ax.imshow(img, cmap="gray", vmin=p1, vmax=p99)

        # Air ROI [30:130, 30:130] (red)
        rect_air = patches.Rectangle((30, 30), 100, 100, linewidth=1.8, edgecolor="red", facecolor="red", alpha=0.25, label="Air ROI [30:130, 30:130]")
        ax.add_patch(rect_air)

        # Outer specimen mask M_sample (green outline, eroded 15px)
        ax.contour(r["eroded"], levels=[0.5], colors=["limegreen"], linewidths=1.8)

        # Inner texture ROI (radius 100 px, gold)
        circle_inner = patches.Circle((r["xc"], r["yc"]), 100.0, linewidth=1.8, edgecolor="gold", facecolor="gold", alpha=0.22, label="Inner Texture ROI (R=100px)")
        ax.add_patch(circle_inner)

        # Detected boundary points (orange dots) & arc
        x_edge = [pt[0] for pt in r["detected_xy"]]
        y_edge = [pt[1] for pt in r["detected_xy"]]
        ax.plot(x_edge, y_edge, color="orange", lw=2.2, label="Detected Right Boundary Arc")
        ax.scatter(x_edge[::5], y_edge[::5], color="darkorange", s=18, zorder=5)

        flag_str = " [FAIL]" if r["mask_fail"] else " [PASS]"
        title_text = (
            f"Tuning Slice #{pos*2+1} (idx {idx}: {fname})\n"
            f"Area={r['area']}, Sol={r['solidity']:.3f}, Frac<=3px={r['frac_within_3px']:.1%}{flag_str}\n"
            f"Unfiltered Erf Width={r['w_erf']:.2f}px (Contrast={r['delta']:.3f}, Val={r['val']:.0f}x)"
        )
        ax.set_title(title_text, fontsize=9.5, fontweight="bold", pad=8)
        ax.axis("off")
        if pos == 0:
            custom_lines = [
                patches.Patch(facecolor="red", edgecolor="red", alpha=0.4, label="Air ROI [30:130, 30:130]"),
                Line2D([0], [0], color="limegreen", lw=2, label="Specimen Mask M_sample (eroded 15px)"),
                patches.Patch(facecolor="gold", edgecolor="gold", alpha=0.4, label="Inner Texture ROI (R=100px)"),
                Line2D([0], [0], color="orange", lw=2, label="Aligned Right Edge Arc"),
            ]
            ax.legend(handles=custom_lines, loc="lower left", fontsize=7.5, framealpha=0.9)

    fig1.suptitle("Calibration ROI & Edge Geometry Across 6 Tuning Slices\n[Air ROI (Red) | Specimen Mask M_sample (Green Outline) | Inner ROI (Gold) | Aligned Edge (Orange)]", fontsize=12, fontweight="bold")
    overlay_path = docs_dir / "calibration_roi_overlay.png"
    fig1.savefig(overlay_path, dpi=200)
    plt.close(fig1)
    print(f"Saved overlay figure to: {overlay_path}")

    # 2. Figure 2: Aligned Profiles Figure (3 slices)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    plot_indices = [350, 382, 438]
    plot_indices = [i for i in plot_indices if i in res_all]

    for pos, idx in enumerate(plot_indices):
        _, fname, r = res_all[idx]
        ax = axes2[pos]
        s_rel = r["s_relative"]

        # Plot individual rays
        for prof in r["aligned_profiles"]:
            ax.plot(s_rel, prof, color="steelblue", alpha=0.20, lw=0.8)

        # Plot mean profile & fitted erf model
        ax.plot(s_rel, r["mean_profile"], color="navy", lw=2.5, label="Aligned Average Profile")
        ax.plot(s_rel, r["erf_fit_curve"], color="crimson", lw=2.0, linestyle="--", label=f"Erf Fit (W_erf={r['w_erf']:.2f}px)")

        ax.axvline(0, color="gray", linestyle=":", lw=1.2, label="Aligned Boundary (s=0)")
        ax.set_xlabel("Aligned Normal Distance s from Boundary (pixels)", fontsize=10, fontweight="bold")
        if pos == 0:
            ax.set_ylabel("Attenuation Intensity", fontsize=10, fontweight="bold")

        info_box = (
            f"Unfiltered W_erf: {r['w_erf']:.2f} px\n"
            f"Plateau Delta: {r['delta']:.4f}\n"
            f"Std Error SE: {r['se']:.4f}\n"
            f"Validity Ratio: {r['val']:.1f}x\n"
            f"Area: {r['area']} px, Sol: {r['solidity']:.3f}"
        )
        ax.text(0.04, 0.22, info_box, transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9))

        ax.set_title(f"Slice {idx} ({fname})\nUnfiltered Edge Width = {r['w_erf']:.2f} px", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig2.suptitle("Per-Ray Aligned Edge Profiles and Error-Function (Erf) Step Fits Across 3 Tuning Slices\n[Individual Rays (Light Blue) | Aligned Average (Navy) | Fitted Model (Crimson)]", fontsize=12, fontweight="bold")
    profiles_path = docs_dir / "calibration_aligned_profiles.png"
    fig2.savefig(profiles_path, dpi=200)
    plt.close(fig2)
    print(f"Saved aligned profiles figure to: {profiles_path}")

if __name__ == "__main__":
    main()
