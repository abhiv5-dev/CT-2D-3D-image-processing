#!/usr/bin/env python3
"""
Small Real-Data Demo Script.
Processes 6 evenly spaced slices from the real CT dataset with the current
real-data filter settings, saving one crop figure and one noise CSV.
Skipped cleanly (exit 0) if the real dataset is missing or unavailable.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
import numpy as np

# Set headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local modules and C++ bindings are in sys.path
repo_root = Path(__file__).resolve().parent.parent
python_dir = repo_root / "python"
build_dir = repo_root / "build"
for p in [str(python_dir), str(build_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import tifffile
    import ct
    from scipy.ndimage import correlate
except ImportError as e:
    # If modules aren't available, report and exit gracefully
    print(f"Notice: Required module not found ({e}). Skipping real-data demo.")
    sys.exit(0)

IMMERKAER_KERNEL = np.array([
    [ 1, -2,  1],
    [-2,  4, -2],
    [ 1, -2,  1]
], dtype=np.float32)

def estimate_sigma_hat(img):
    c = correlate(img, IMMERKAER_KERNEL, mode="reflect")
    return float(np.median(np.abs(c)) / (0.67448975 * 6.0))

def load_slice(fpath):
    try:
        return ct.load_tiff(str(fpath))
    except Exception:
        pass
    with tifffile.TiffFile(str(fpath)) as tif:
        raw = tif.pages[0].asarray().astype(np.float32)
        desc = tif.pages[0].tags.get("ImageDescription")
        slope, offset = 1.0, 0.0
        if desc:
            desc_str = str(desc.value)
            m_s = re.search(r"slope\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", desc_str)
            m_o = re.search(r"offset\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", desc_str)
            if m_s: slope = float(m_s.group(1))
            if m_o: offset = float(m_o.group(1))
        return raw * slope + offset

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def find_dataset(cli_data_dir=None):
    candidates = []
    if cli_data_dir:
        candidates.append(Path(cli_data_dir))
    env_dir = os.environ.get("CT_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(repo_root / "data" / "real")

    for cand in candidates:
        try:
            if cand.is_dir():
                # Check for tiff files
                tifs = sorted(list(cand.glob("*.tif")) + list(cand.glob("*.tiff")), key=natural_sort_key)
                if len(tifs) >= 6:
                    return cand, tifs
        except (PermissionError, OSError):
            continue
    return None, []

def main():
    parser = argparse.ArgumentParser(description="Real-data CT filtering demo")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to real data slices")
    args = parser.parse_args()

    data_dir, slice_files = find_dataset(args.data_dir)
    if not data_dir or len(slice_files) < 6:
        print("Notice: Real dataset not found (or fewer than 6 slices found).")
        print("Skipping real-data demo cleanly as dataset is optional.")
        sys.exit(0)

    print(f"Loaded real dataset from: {data_dir} ({len(slice_files)} slices discovered)")

    # Pick 6 evenly spaced slices across the specimen block (450 to 629)
    if len(slice_files) >= 630:
        indices = np.linspace(450, 629, 6, dtype=int)
    else:
        indices = np.linspace(0, len(slice_files) - 1, 6, dtype=int)
    chosen_files = [slice_files[i] for i in indices]
    print(f"Selected 6 slices from specimen block: {[f.name for f in chosen_files]}")

    # Standard fixed ROIs for real CT stack:
    # Air ROI in corner, Specimen matrix ROI in solid specimen core
    air_slice = (slice(30, 130), slice(30, 130))
    mat_slice = (slice(462, 562), slice(462, 562))

    csv_rows = []
    # Dictionary of crops from the middle slice (slice index 2 of the 6)
    mid_crops = {}

    for s_idx, fpath in enumerate(chosen_files):
        orig = load_slice(fpath)
        sigma_hat = estimate_sigma_hat(orig)

        # Calibrated filter defaults from config/default_params.json (Gaussian sigma=1.0 uncalibrated practical default)
        filters = [
            ("Gaussian (sigma=1.0)", lambda x: ct.gaussian_filter_2d(x, sigma=1.0, radius=3)),
            ("Median (r=1)", lambda x: ct.median_filter_2d(x, radius=1)),
            ("Bilateral (m=3.0x)", lambda x, sh=sigma_hat: ct.bilateral_filter_2d(x, sigma_spatial=2.0, sigma_range=3.0 * sh, radius=5)),
            ("Diffusion (m=1.5x, it=15)", lambda x, sh=sigma_hat: ct.anisotropic_diffusion_2d(x, kappa=1.5 * sh, iterations=15, dt=0.20)),
            ("NLM (m=0.75x)", lambda x, sh=sigma_hat: ct.nlm_filter_2d(x, 2, 5, 0.75 * sh)),
            ("Wiener (r=2)", lambda x: ct.wiener_filter_2d(x, radius=2)),
            ("TV_Chambolle (m=1.0x)", lambda x, sh=sigma_hat: ct.tv_chambolle_2d(x, 1.0 * sh, 2e-4, 30)),
            ("Morphology (opening, r=1)", lambda x: ct.grayscale_opening_2d(x, radius=1, element="box")),
        ]

        H, W = orig.shape
        if H >= 950 and W >= 850:
            crop_slice = (slice(450, 550), slice(450, 650))
        else:
            crop_slice = (slice(H // 4, min(H, H // 4 + 100)), slice(W // 4, min(W, W // 4 + 200)))

        air_std_orig = float(np.std(orig[air_slice]))
        mat_std_orig = float(np.std(orig[mat_slice]))

        if s_idx == 2:
            mid_crops["Original"] = orig[crop_slice]

        for fname, fn in filters:
            filtered = fn(orig)
            air_std_filt = float(np.std(filtered[air_slice]))
            mat_std_filt = float(np.std(filtered[mat_slice]))

            air_red = (1.0 - air_std_filt / air_std_orig) * 100.0 if air_std_orig > 0 else 0.0
            mat_red = (1.0 - mat_std_filt / mat_std_orig) * 100.0 if mat_std_orig > 0 else 0.0

            csv_rows.append({
                "filter": fname,
                "slice_file": fpath.name,
                "air_noise_before": f"{air_std_orig:.6f}",
                "air_noise_after": f"{air_std_filt:.6f}",
                "air_noise_red_pct": f"{air_red:.2f}%",
                "texture_std_before": f"{mat_std_orig:.6f}",
                "texture_std_after": f"{mat_std_filt:.6f}",
                "texture_std_red_pct": f"{mat_red:.2f}%"
            })

            if s_idx == 2:
                mid_crops[fname] = filtered[crop_slice]

    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    # 1. Write noise CSV
    csv_path = docs_dir / "real_data_demo_noise.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filter", "slice_file", "air_noise_before", "air_noise_after",
            "air_noise_red_pct", "texture_std_before", "texture_std_after", "texture_std_red_pct"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved real-data demo noise CSV to: {csv_path}")

    # 2. Write one figure with before and after crops
    fig, axes = plt.subplots(1, len(mid_crops), figsize=(2.8 * len(mid_crops), 3.2))
    vmin = float(np.percentile(mid_crops["Original"], 2))
    vmax = float(np.percentile(mid_crops["Original"], 98))

    for ax, (name, crop_img) in zip(axes, mid_crops.items()):
        ax.imshow(crop_img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle("Real CT Data: Before vs. After Filter Crops (Support Band Detail)", fontsize=12)
    plt.tight_layout()
    fig_path = docs_dir / "real_data_demo_crops.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved real-data demo crops figure to: {fig_path}")

if __name__ == "__main__":
    main()
