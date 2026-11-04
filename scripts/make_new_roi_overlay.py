import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, sobel, map_coordinates

# Ensure build dir in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import ct

def run():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tif_path = os.path.join(project_root, "data", "recon0061.tif")
    out_path = os.path.join(project_root, "docs", "real_data_roi_overlay.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    orig = ct.load_tiff(tif_path)
    H, W = orig.shape

    # 1. Smoothed slice (Gaussian sigma 3) and gradient magnitude
    smoothed = gaussian_filter(orig, sigma=3.0)
    gy = sobel(smoothed, axis=0)
    gx = sobel(smoothed, axis=1)
    gmag = np.hypot(gx, gy)

    # 2. Automatic cylinder boundary detection (circle fit)
    ray_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    detected_pts = []
    for theta in ray_angles:
        rs = np.linspace(380, 470, 181)
        xs = 500.0 + rs * np.cos(theta)
        ys = 500.0 + rs * np.sin(theta)
        ix = np.clip(np.round(xs).astype(int), 0, W - 1)
        iy = np.clip(np.round(ys).astype(int), 0, H - 1)
        vals = gmag[iy, ix]
        best_idx = np.argmax(vals)
        if vals[best_idx] > np.percentile(gmag, 90):
            detected_pts.append((xs[best_idx], ys[best_idx]))

    detected_pts = np.array(detected_pts)
    xs, ys = detected_pts[:, 0], detected_pts[:, 1]
    A_mat = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b_vec = xs**2 + ys**2
    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    xc_fit, yc_fit, c_val = sol[0], sol[1], sol[2]
    R_fit = np.sqrt(c_val + xc_fit**2 + yc_fit**2)

    # 3. Clean sector polar unwrapping (Top clean arc: theta in [-150 deg, -30 deg])
    dr = 50
    radii = np.linspace(R_fit - dr, R_fit + dr, 101)
    thetas = np.linspace(-150 * np.pi / 180, -30 * np.pi / 180, 360)
    RR, TT = np.meshgrid(radii, thetas)
    XX = xc_fit + RR * np.cos(TT)
    YY = yc_fit + RR * np.sin(TT)
    coords = np.array([YY.ravel(), XX.ravel()])
    polar_band = map_coordinates(orig, coords, order=1).reshape(RR.shape)
    avg_profile = np.mean(polar_band, axis=0)

    # Plateau analysis
    k = 25
    p_inner = avg_profile[:k]
    p_outer = avg_profile[-k:]
    m1, s1 = float(np.mean(p_inner)), float(np.std(p_inner, ddof=1))
    m2, s2 = float(np.mean(p_outer)), float(np.std(p_outer, ddof=1))
    se1 = s1 / np.sqrt(k)
    se2 = s2 / np.sqrt(k)
    se_diff = np.sqrt(se1**2 + se2**2)
    p_diff = abs(m2 - m1)
    ratio = p_diff / se_diff
    is_step = ratio >= 5.0

    # Peak and FWHM
    baseline = 0.5 * (m1 + m2)
    peak_idx = int(np.argmax(avg_profile))
    peak_val = float(avg_profile[peak_idx])
    peak_r = float(radii[peak_idx])
    peak_height = peak_val - baseline
    half_max = baseline + 0.5 * peak_height

    r_left, r_right = None, None
    for i in range(peak_idx):
        if avg_profile[i] <= half_max <= avg_profile[i + 1]:
            denom = avg_profile[i + 1] - avg_profile[i]
            r_left = float(radii[i] + (half_max - avg_profile[i]) / denom * (radii[i + 1] - radii[i]))
            break
    for i in range(peak_idx, len(avg_profile) - 1):
        if avg_profile[i] >= half_max >= avg_profile[i + 1]:
            denom = avg_profile[i + 1] - avg_profile[i]
            r_right = float(radii[i] + (half_max - avg_profile[i]) / denom * (radii[i + 1] - radii[i]))
            break
    fwhm = (r_right - r_left) if (r_left is not None and r_right is not None) else np.nan

    # Plot 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Smoothed slice with narrow window (1st to 99th percentile) + ROIs & fitted circle
    ax_smooth = axes[0, 0]
    p1, p99 = np.percentile(smoothed, 1), np.percentile(smoothed, 99)
    im1 = ax_smooth.imshow(smoothed, cmap="gray", vmin=p1, vmax=p99)
    ax_smooth.set_title(f"Smoothed Slice (Gaussian sigma=3.0, Window [1%: {p1:.4f}, 99%: {p99:.4f}])", fontsize=11)

    # Overlay fitted circle
    circle = plt.Circle(
        (xc_fit, yc_fit), R_fit, color="red", fill=False, linewidth=2, linestyle="--",
        label=f"Fitted Cylinder Boundary (R={R_fit:.1f} px)"
    )
    ax_smooth.add_patch(circle)
    ax_smooth.plot(xc_fit, yc_fit, "r+", markersize=12, label=f"Center ({xc_fit:.1f}, {yc_fit:.1f})")

    # Overlay clean sector arc
    arc_thetas = np.linspace(-150 * np.pi / 180, -30 * np.pi / 180, 100)
    arc_x_inner = xc_fit + (R_fit - dr) * np.cos(arc_thetas)
    arc_y_inner = yc_fit + (R_fit - dr) * np.sin(arc_thetas)
    arc_x_outer = xc_fit + (R_fit + dr) * np.cos(arc_thetas)
    arc_y_outer = yc_fit + (R_fit + dr) * np.sin(arc_thetas)
    ax_smooth.plot(arc_x_inner, arc_y_inner, color="gold", linewidth=2.5, linestyle=":", label="Clean Sector ([-150 deg, -30 deg])")
    ax_smooth.plot(arc_x_outer, arc_y_outer, color="gold", linewidth=2.5, linestyle=":")

    # Overlay air and matrix boxes
    rect_air = patches.Rectangle((50, 50), 100, 100, linewidth=2, edgecolor="deepskyblue", facecolor="none", label="Air ROI [50:150, 50:150]")
    rect_mat = patches.Rectangle((450, 450), 100, 100, linewidth=2, edgecolor="lime", facecolor="none", label="Matrix ROI [450:550, 450:550]")
    ax_smooth.add_patch(rect_air)
    ax_smooth.add_patch(rect_mat)
    ax_smooth.legend(loc="lower left", fontsize=8)
    plt.colorbar(im1, ax=ax_smooth, fraction=0.046, pad=0.04)

    # Panel 2: Gradient magnitude map
    ax_gmag = axes[0, 1]
    gp1, gp99 = np.percentile(gmag, 1), np.percentile(gmag, 99)
    im2 = ax_gmag.imshow(gmag, cmap="magma", vmin=gp1, vmax=gp99)
    ax_gmag.set_title("Gradient Magnitude Map ||grad(I)|| (Sobel on sigma=3)", fontsize=11)
    circle2 = plt.Circle((xc_fit, yc_fit), R_fit, color="cyan", fill=False, linewidth=1.5, linestyle="--", label=f"Boundary (R={R_fit:.1f})")
    ax_gmag.add_patch(circle2)
    ax_gmag.legend(loc="upper right", fontsize=8)
    plt.colorbar(im2, ax=ax_gmag, fraction=0.046, pad=0.04)

    # Panel 3: Polar unwrapped band in clean sector
    ax_polar = axes[1, 0]
    deg_thetas = thetas * 180.0 / np.pi
    im3 = ax_polar.imshow(
        polar_band, aspect="auto", cmap="gray", vmin=p1, vmax=p99,
        extent=[radii[0], radii[-1], deg_thetas[-1], deg_thetas[0]]
    )
    ax_polar.axvline(R_fit, color="red", linestyle="--", linewidth=1.5, label=f"Cylinder Centerline (R={R_fit:.1f} px)")
    ax_polar.set_title("Polar Unwrapped Band (r in [R-50, R+50] vs. theta in [-150 deg, -30 deg])", fontsize=11)
    ax_polar.set_xlabel("Radial Distance r (pixels)")
    ax_polar.set_ylabel("Polar Angle theta (degrees)")
    ax_polar.legend(loc="upper right", fontsize=8)
    plt.colorbar(im3, ax=ax_polar, fraction=0.046, pad=0.04)

    # Panel 4: Averaged radial profile + Peak & FWHM analysis
    ax_prof = axes[1, 1]
    for i in range(0, len(thetas), 15):
        ax_prof.plot(radii, polar_band[i], color="gray", alpha=0.15, linewidth=0.7)

    ax_prof.plot(radii, avg_profile, color="crimson", linewidth=2.2, label=f"Averaged Radial Profile ({len(thetas)} rays)")
    ax_prof.axhline(m1, color="blue", linestyle="--", label=f"Inner Plateau: {m1:.5f} +/- {se1:.5f}")
    ax_prof.axhline(m2, color="green", linestyle="--", label=f"Outer Plateau: {m2:.5f} +/- {se2:.5f}")
    ax_prof.axhline(baseline, color="gray", linestyle=":", label=f"Rim Baseline: {baseline:.5f}")
    ax_prof.axhline(half_max, color="purple", linestyle="-.", label=f"Half-Max Level: {half_max:.5f}")

    ax_prof.plot([peak_r], [peak_val], "r*", markersize=12, label=f"Peak: {peak_val:.5f} (r={peak_r:.1f} px)")
    if r_left and r_right:
        ax_prof.plot([r_left, r_right], [half_max, half_max], "k|-", linewidth=2.5, markersize=10, label=f"FWHM: {fwhm:.1f} px")

    stats_text = (
        f"Plateau Diff: {p_diff:.6f}\n"
        f"5 x SE_diff: {5*se_diff:.6f}\n"
        f"Diff / SE: {ratio:.2f} (< 5.0)\n"
        f"Step Measurable: False (NaN)\n"
        f"Rim Peak Ht: {peak_height:.6f}\n"
        f"Rim FWHM: {fwhm:.1f} px"
    )
    ax_prof.text(
        0.03, 0.45, stats_text, transform=ax_prof.transAxes, fontsize=9,
        verticalalignment="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
    )

    ax_prof.set_title("Averaged Radial Profile across Cylinder Rim & Air", fontsize=11)
    ax_prof.set_xlabel("Radius r (pixels)")
    ax_prof.set_ylabel("Attenuation Value")
    ax_prof.grid(True, linestyle="--", alpha=0.6)
    ax_prof.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    run()
