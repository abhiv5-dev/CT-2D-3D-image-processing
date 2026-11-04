#!/usr/bin/env python3
"""
Slice Survey Script (scripts/slice_survey.py).
Scans every 5th slice of the real CT dataset stack and computes:
- Index, file name
- 1st and 99th percentiles (P1, P99)
- Robust dynamic range R = P99 - P1
- Robust noise estimate sigma_hat (Immerkaer / Laplacian MAD)
- Standard deviation in top-left air corner ROI [30:130, 30:130]

Outputs:
- docs/slice_survey.csv
- docs/slice_survey.png (R and sigma_hat vs. slice index)
- docs/slice_survey_montage.png (Montage of slices 0, 100, 250, 350, 500, 700 with per-slice P1-P99 scaling)

Exits cleanly (exit code 0) if the real dataset is missing.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import correlate

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Loads a 2D CT slice as float32 attenuation values, applying TIFF slope/offset if present."""
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
            m_slope = re.search(r"slope\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", desc_str)
            m_offset = re.search(r"offset\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", desc_str)
            if m_slope:
                slope = float(m_slope.group(1))
            if m_offset:
                offset = float(m_offset.group(1))
        return raw * slope + offset

# Immerkaer Laplacian-of-Laplacian 3x3 kernel
IMMERKAER_KERNEL = np.array([
    [ 1, -2,  1],
    [-2,  4, -2],
    [ 1, -2,  1]
], dtype=np.float32)

def compute_metrics_for_slice(img):
    """Computes P1, P99, R, robust noise estimate sigma_hat (Laplacian MAD), and air_std."""
    p1 = float(np.percentile(img, 1))
    p99 = float(np.percentile(img, 99))
    R = float(p99 - p1)

    # Immerkaer Laplacian convolution with MAD estimator
    c = correlate(img, IMMERKAER_KERNEL, mode="reflect")
    # For Gaussian noise, MAD of (c) / (0.67448975 * sqrt(36)) = sigma_hat
    sigma_hat = float(np.median(np.abs(c)) / (0.67448975 * 6.0))

    # Top-left corner air ROI: [30:130, 30:130]
    air_crop = img[30:130, 30:130]
    air_std = float(np.std(air_crop))

    return p1, p99, R, sigma_hat, air_std

def main():
    parser = argparse.ArgumentParser(description="Survey CT slices across the volume stack.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to real CT dataset")
    parser.add_argument("--step", type=int, default=5, help="Sampling stride (default: 5)")
    args = parser.parse_args()

    data_dir, slice_files = find_dataset(args.data_dir)
    if not data_dir or len(slice_files) == 0:
        print("Notice: Real CT dataset not found. Skipping slice survey cleanly.")
        sys.exit(0)

    print(f"Discovered real CT dataset at: {data_dir} ({len(slice_files)} slices)")

    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = docs_dir / "slice_survey.csv"

    survey_data = []
    print(f"Sampling every {args.step}th slice...")

    for idx in range(0, len(slice_files), args.step):
        fpath = slice_files[idx]
        img = load_slice(fpath)
        p1, p99, R, sigma_hat, air_std = compute_metrics_for_slice(img)
        survey_data.append({
            "index": idx,
            "file_name": fpath.name,
            "p1": p1,
            "p99": p99,
            "R": R,
            "sigma_hat": sigma_hat,
            "air_std": air_std
        })

    # 1. Write docs/slice_survey.csv
    fieldnames = ["index", "file_name", "p1", "p99", "R", "sigma_hat", "air_std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in survey_data:
            writer.writerow({
                "index": row["index"],
                "file_name": row["file_name"],
                "p1": f"{row['p1']:.6f}",
                "p99": f"{row['p99']:.6f}",
                "R": f"{row['R']:.6f}",
                "sigma_hat": f"{row['sigma_hat']:.6f}",
                "air_std": f"{row['air_std']:.6f}"
            })
    print(f"Saved survey CSV to: {csv_path} ({len(survey_data)} slices evaluated)")

    # Extract arrays for plotting
    indices = [d["index"] for d in survey_data]
    Rs = [d["R"] for d in survey_data]
    sigmas = [d["sigma_hat"] for d in survey_data]
    air_stds = [d["air_std"] for d in survey_data]

    # 2. Plot docs/slice_survey.png (R and sigma_hat vs. index)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(indices, Rs, color="tab:blue", lw=2, label=r"Dynamic Range $R = P_{99} - P_1$")
    ax1.axvspan(0, 225, color="gray", alpha=0.15, label="Empty Air (No Sample)")
    ax1.axvspan(230, len(slice_files)-1, color="tab:green", alpha=0.12, label="Sample Present (Rock Core)")
    ax1.axvline(225, color="tab:red", linestyle="--", alpha=0.7, label="Sample Transition (~idx 225)")
    ax1.set_ylabel(r"Dynamic Range $R$", fontsize=11, fontweight="bold")
    ax1.set_title(r"CT Slice Stack Survey: Dynamic Range $R$ and Robust Noise $\hat{\sigma}$ vs. Slice Index", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.plot(indices, sigmas, color="tab:purple", lw=2, label=r"$\hat{\sigma}$ (Immerkaer / Laplacian MAD)")
    ax2.plot(indices, air_stds, color="tab:orange", lw=1.5, linestyle="--", label=r"Air ROI Std $\sigma_{\text{air}}$ [30:130, 30:130]")
    ax2.axvspan(0, 225, color="gray", alpha=0.15)
    ax2.axvspan(230, len(slice_files)-1, color="tab:green", alpha=0.12)
    ax2.axvline(225, color="tab:red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Slice Index in Stack (0 to 846)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Noise Estimate $\hat{\sigma}$", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plot_path = docs_dir / "slice_survey.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved survey plot to: {plot_path}")

    # 3. Montage of slices 0, 100, 250, 350, 500, 700 with per-slice P1-P99 display range
    montage_indices = [0, 100, 250, 350, 500, 700]
    montage_indices = [i for i in montage_indices if i < len(slice_files)]

    fig_m, axes_m = plt.subplots(2, 3, figsize=(12, 8))
    axes_m = axes_m.ravel()

    for pos, m_idx in enumerate(montage_indices):
        fpath = slice_files[m_idx]
        img = load_slice(fpath)
        p1, p99, R, sigma_hat, air_std = compute_metrics_for_slice(img)
        ax = axes_m[pos]
        im = ax.imshow(img, cmap="gray", vmin=p1, vmax=p99)
        sample_label = "Sample Present" if m_idx >= 230 else "Empty Air (No Sample)"
        ax.set_title(
            f"Index {m_idx}: {fpath.name} ({sample_label})\n"
            f"P1={p1:.3f}, P99={p99:.3f}, R={R:.3f}, $\\hat{{\\sigma}}$={sigma_hat:.4f}",
            fontsize=9
        )
        ax.axis("off")

    fig_m.suptitle(
        r"Slice Survey Montage (Indices 0, 100, 250, 350, 500, 700) with Per-Slice $P_1-P_{99}$ Display Range",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    montage_path = docs_dir / "slice_survey_montage.png"
    plt.savefig(montage_path, dpi=200)
    plt.close(fig_m)
    print(f"Saved survey montage to: {montage_path}")

if __name__ == "__main__":
    main()
