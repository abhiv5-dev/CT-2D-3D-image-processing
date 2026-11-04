import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import erf

# Ensure build dir in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import ct
from ct_eval.metrics import roi_noise_std, edge_width_10_90, fit_edge_erf

# ==============================================================================
# CONFIGURATION BLOCK FOR REAL DATA (recon0061.tif)
# ==============================================================================
REAL_DATA_CONFIG = {
    "air_roi": {"y0": 50, "y1": 150, "x0": 50, "x1": 150, "label": "Air ROI"},
    "matrix_roi": {"y0": 450, "y1": 550, "x0": 450, "x1": 550, "label": "Matrix ROI"},
    # Validated band edge parameters: 400 px stretch, tilt = -5.48 deg
    "band_edge": {
        "x_start": 550,
        "x_end": 950,
        "n_profiles": 401,
        "slope": -0.09757,
        "intercept": 950.33,
        "tilt_deg": -5.48,
        "s_min": -25.0,
        "s_max": 2.0,
        "n_samples": 28,
        "block_size": 8,
    },
    # Crops for visual comparison
    "crops": {
        "band": {"y0": 830, "y1": 930, "x0": 650, "x1": 850, "title": "Support Band Crop"},
        "rim":  {"y0": 40,  "y1": 140, "x0": 400, "x1": 600, "title": "Cylinder Rim Crop"}
    }
}

def extract_band_profiles(img, config=REAL_DATA_CONFIG["band_edge"]):
    cfg = config
    xs = np.linspace(cfg["x_start"], cfg["x_end"], cfg["n_profiles"])
    s_vals = np.linspace(cfg["s_min"], cfg["s_max"], cfg["n_samples"])
    slope = cfg["slope"]
    intercept = cfg["intercept"]

    theta = np.arctan(slope)
    nx = -np.sin(theta)
    ny = np.cos(theta)

    SS, XX = np.meshgrid(s_vals, xs)
    YY = slope * XX + intercept
    sample_X = XX + SS * nx
    sample_Y = YY + SS * ny
    coords = np.array([sample_Y.ravel(), sample_X.ravel()])
    return map_coordinates(img, coords, order=1).reshape(SS.shape)

def evaluate_real_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tif_path = os.path.join(project_root, "data", "recon0061.tif")
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Missing {tif_path}")

    orig = ct.load_tiff(tif_path)
    H, W = orig.shape

    cfg_air = REAL_DATA_CONFIG["air_roi"]
    cfg_mat = REAL_DATA_CONFIG["matrix_roi"]
    mask_air = np.zeros((H, W), dtype=bool)
    mask_air[cfg_air["y0"]:cfg_air["y1"], cfg_air["x0"]:cfg_air["x1"]] = True
    mask_mat = np.zeros((H, W), dtype=bool)
    mask_mat[cfg_mat["y0"]:cfg_mat["y1"], cfg_mat["x0"]:cfg_mat["x1"]] = True

    # Crop mask for band region to measure method noise structural leakage
    mask_band = np.zeros((H, W), dtype=bool)
    c_band = REAL_DATA_CONFIG["crops"]["band"]
    mask_band[c_band["y0"]:c_band["y1"], c_band["x0"]:c_band["x1"]] = True

    # Initial unfiltered measurements
    orig_air_std = roi_noise_std(orig, mask_air)
    orig_mat_std = roi_noise_std(orig, mask_mat)
    orig_profiles = extract_band_profiles(orig)
    orig_w_lin = edge_width_10_90(orig_profiles)
    orig_sigma, orig_w_erf = fit_edge_erf(orig_profiles)

    filters = [
        ("Gaussian (sigma=1.5)", lambda x: ct.gaussian_filter_2d(x, sigma=1.5, radius=4)),
        ("Median (radius=1)", lambda x: ct.median_filter_2d(x, radius=1)),
        ("Bilateral (s=2.0, r=0.015)", lambda x: ct.bilateral_filter_2d(x, sigma_spatial=2.0, sigma_range=0.015, radius=4)),
        ("Diffusion (kappa=0.015, iters=6)", lambda x: ct.anisotropic_diffusion_2d(x, kappa=0.015, iterations=6, dt=0.20)),
        ("NLM (patch=2, search=5, h=0.015)", lambda x: ct.nlm_filter_2d(x, 2, 5, 0.015)),
        ("Wiener (radius=1)", lambda x: ct.wiener_filter_2d(x, radius=1)),
        ("TV Chambolle (weight=0.05)", lambda x: ct.tv_chambolle_2d(x, weight=0.05, eps=1e-4, max_iterations=40)),
        ("Opening (radius=1, box)", lambda x: ct.grayscale_opening_2d(x, radius=1, element="box")),
        ("Closing (radius=1, box)", lambda x: ct.grayscale_closing_2d(x, radius=1, element="box")),
    ]

    results = []
    filtered_images = {"Original": orig}

    for name, fn in filters:
        f_img = fn(orig)
        filtered_images[name] = f_img

        air_std = roi_noise_std(f_img, mask_air)
        mat_std = roi_noise_std(f_img, mask_mat)
        air_red_pct = 100.0 * (1.0 - air_std / orig_air_std)
        mat_red_pct = 100.0 * (1.0 - mat_std / orig_mat_std)

        # Edge widths on band
        f_profiles = extract_band_profiles(f_img)
        w_lin = edge_width_10_90(f_profiles)
        s_erf, w_erf = fit_edge_erf(f_profiles)

        # Method noise (|input - filtered|)
        diff = np.abs(orig - f_img)
        mn_air = float(np.mean(diff[mask_air]))
        mn_band = float(np.mean(diff[mask_band]))
        leakage_ratio = mn_band / mn_air if mn_air > 0 else np.nan

        results.append({
            "filter": name,
            "air_noise_std": air_std,
            "matrix_noise_std": mat_std,
            "air_noise_red_pct": air_red_pct,
            "matrix_noise_red_pct": mat_red_pct,
            "edge_width_linear": w_lin,
            "erf_sigma": s_erf,
            "edge_width_erf": w_erf,
            "method_noise_air": mn_air,
            "method_noise_band": mn_band,
            "leakage_ratio": leakage_ratio
        })

    # Save real_data_results.csv
    import csv
    csv_path = os.path.join(docs_dir, "real_data_results.csv")
    fieldnames = [
        "filter", "air_std", "matrix_std", "air_noise_red_pct", "matrix_noise_red_pct",
        "edge_width_linear", "erf_sigma", "edge_width_erf", "method_noise_air", "method_noise_band", "leakage_ratio"
    ]
    with open(csv_path, "w", newline="") as f:
        f.write("# Protocol: Real CT Reconstruction Slice recon0061.tif (recon0061, empty slice, noise only; band edge: x in [550, 950], tilt=-5.48 deg)\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerow({
            "filter": "Original",
            "air_std": f"{orig_air_std:.6f}",
            "matrix_std": f"{orig_mat_std:.6f}",
            "air_noise_red_pct": 0.0,
            "matrix_noise_red_pct": 0.0,
            "edge_width_linear": f"{orig_w_lin:.3f}",
            "erf_sigma": f"{orig_sigma:.3f}",
            "edge_width_erf": f"{orig_w_erf:.3f}",
            "method_noise_air": 0.0,
            "method_noise_band": 0.0,
            "leakage_ratio": 1.0
        })
        for r in results:
            writer.writerow({
                "filter": r['filter'],
                "air_std": f"{r['air_noise_std']:.6f}",
                "matrix_std": f"{r['matrix_noise_std']:.6f}",
                "air_noise_red_pct": f"{r['air_noise_red_pct']:.2f}",
                "matrix_noise_red_pct": f"{r['matrix_noise_red_pct']:.2f}",
                "edge_width_linear": f"{r['edge_width_linear']:.3f}",
                "erf_sigma": f"{r['erf_sigma']:.3f}",
                "edge_width_erf": f"{r['edge_width_erf']:.3f}",
                "method_noise_air": f"{r['method_noise_air']:.6f}",
                "method_noise_band": f"{r['method_noise_band']:.6f}",
                "leakage_ratio": f"{r['leakage_ratio']:.3f}"
            })
    print(f"Saved real data CSV to: {csv_path}")

    # (b) Method noise images figure
    fig_mn, axes_mn = plt.subplots(1, len(filters), figsize=(4 * len(filters), 4.5))
    for ax, (name, _) in zip(axes_mn, filters):
        diff = np.abs(orig - filtered_images[name])
        im = ax.imshow(diff, cmap="inferno", vmin=0.0, vmax=0.015)
        ax.set_title(f"{name}\nDiff in Band/Air = {np.mean(diff[mask_band])/np.mean(diff[mask_air]):.2f}x", fontsize=9)
        ax.axis("off")
    fig_mn.suptitle("Method Noise Images |Input - Filtered| (Structural Leakage Comparison)", fontsize=12)
    plt.tight_layout()
    mn_fig_path = os.path.join(docs_dir, "real_data_method_noise.png")
    fig_mn.savefig(mn_fig_path, dpi=180)
    plt.close(fig_mn)
    print(f"Saved method noise figure to: {mn_fig_path}")

    # (c) Shared-window before/after crops of band and rim
    fig_crops, axes_crops = plt.subplots(2, len(filters) + 1, figsize=(3 * (len(filters) + 1), 6))
    vmin, vmax = float(np.percentile(orig, 2)), float(np.percentile(orig, 98))

    cb = REAL_DATA_CONFIG["crops"]["band"]
    cr = REAL_DATA_CONFIG["crops"]["rim"]
    all_names = ["Original"] + [name for name, _ in filters]

    for col_idx, name in enumerate(all_names):
        img = filtered_images[name]
        # Row 0: Band crop
        crop_b = img[cb["y0"]:cb["y1"], cb["x0"]:cb["x1"]]
        axes_crops[0, col_idx].imshow(crop_b, cmap="gray", vmin=vmin, vmax=vmax)
        axes_crops[0, col_idx].set_title(f"{name}\nBand Crop", fontsize=8)
        axes_crops[0, col_idx].axis("off")

        # Row 1: Rim crop
        crop_r = img[cr["y0"]:cr["y1"], cr["x0"]:cr["x1"]]
        axes_crops[1, col_idx].imshow(crop_r, cmap="gray", vmin=vmin, vmax=vmax)
        axes_crops[1, col_idx].set_title("Rim Crop", fontsize=8)
        axes_crops[1, col_idx].axis("off")

    fig_crops.suptitle(f"Before/After Crops with Shared Display Window [{vmin:.4f}, {vmax:.4f}]", fontsize=12)
    plt.tight_layout()
    crops_path = os.path.join(docs_dir, "real_data_crops.png")
    fig_crops.savefig(crops_path, dpi=180)
    plt.close(fig_crops)
    print(f"Saved crops figure to: {crops_path}")

if __name__ == "__main__":
    evaluate_real_data()
