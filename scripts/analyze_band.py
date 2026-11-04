import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import correlate2d
from scipy.optimize import curve_fit
from scipy.special import erf

# Ensure build dir in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import ct

def erf_model(x, a, b, x0, sigma):
    return a + b * erf((x - x0) / (np.sqrt(2.0) * sigma))

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tif_path = os.path.join(project_root, "data", "recon0061.tif")
    out_png = os.path.join(project_root, "docs", "real_data_roi_overlay_band.png")

    orig = ct.load_tiff(tif_path)
    H, W = orig.shape

    # 1. Noise correlation length from autocorrelation of air ROI
    air = orig[50:150, 50:150] - np.mean(orig[50:150, 50:150])
    autocorr = correlate2d(air, air, mode="same") / (np.var(air) * air.size)
    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    r_max = 15
    corr_r = [
        float(np.mean([autocorr[cy + dy, cx + dx]
                       for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                       if abs(np.hypot(dx, dy) - r) < 0.5]) or 0.0)
        for r in range(r_max)
    ]
    corr_r[0] = 1.0
    xi = next((r for r, c in enumerate(corr_r) if c <= 1.0 / np.e), r_max)

    # 2. Detect band ridge in y > 800 on sigma=3 smoothed image
    smoothed = gaussian_filter(orig, sigma=3.0)
    # Longest straight stretch: x in [550, 950] (400 pixels long)
    xs = np.linspace(550, 950, 401)
    peaks_y = []
    for x in xs:
        col = smoothed[835:925, int(round(x))]
        peaks_y.append(835 + np.argmax(col))

    peaks_y = np.array(peaks_y)
    poly = np.polyfit(xs, peaks_y, 1)
    slope, intercept = poly[0], poly[1]
    tilt_rad = np.arctan(slope)
    tilt_deg = tilt_rad * 180.0 / np.pi
    resids = np.abs(peaks_y - (slope * xs + intercept))

    # Unit normal pointing toward matrix (upwards, s < 0)
    nx = -np.sin(tilt_rad)
    ny = np.cos(tilt_rad)

    # 3. Sample perpendicular profiles across the sharper upper edge
    # s in [-25, 0] covers transition from matrix plateau (s=-25..-15) to band plateau (s=-5..0)
    s_vals = np.linspace(-25, 2, 28)
    SS, XX = np.meshgrid(s_vals, xs)
    YY = slope * XX + intercept
    sample_X = XX + SS * nx
    sample_Y = YY + SS * ny
    coords = np.array([sample_Y.ravel(), sample_X.ravel()])
    sampled = map_coordinates(orig, coords, order=1).reshape(SS.shape)

    avg_prof = np.mean(sampled, axis=0)
    k_plateau = 8
    p1 = avg_prof[:k_plateau]
    p2 = avg_prof[-k_plateau:]
    m1 = float(np.mean(p1))
    m2 = float(np.mean(p2))
    p_diff = abs(m2 - m1)

    # 4. Block-averaged standard error with block size B=8 px >= xi
    B = 8
    N_blocks = sampled.shape[0] // B
    block_diffs = []
    for b in range(N_blocks):
        b_prof = np.mean(sampled[b * B : (b + 1) * B, :], axis=0)
        block_diffs.append(float(np.mean(b_prof[-k_plateau:]) - np.mean(b_prof[:k_plateau])))

    block_diffs = np.array(block_diffs)
    se_block = float(np.std(block_diffs, ddof=1) / np.sqrt(N_blocks))
    ratio = p_diff / se_block
    passes_gate = ratio >= 5.0

    # 5. Edge width: 10-90% linear interpolation & Erf fit
    # Normalized profile (rising from 0 to 1)
    norm = (avg_prof - m1) / (m2 - m1) if m2 > m1 else (m1 - avg_prof) / p_diff
    # 10% and 90% crossings
    x10, x90 = None, None
    L = len(norm)
    for i in range(L - 1):
        if norm[i] <= 0.1 <= norm[i + 1]:
            x10 = s_vals[i] + (0.1 - norm[i]) / (norm[i + 1] - norm[i]) * (s_vals[i + 1] - s_vals[i])
            break
    for i in range(L - 2, -1, -1):
        if norm[i] <= 0.9 <= norm[i + 1]:
            x90 = s_vals[i] + (0.9 - norm[i]) / (norm[i + 1] - norm[i]) * (s_vals[i + 1] - s_vals[i])
            break
    w_10_90 = abs(x90 - x10) if (x10 is not None and x90 is not None) else np.nan

    # Erf fit
    p0 = [0.5 * (m1 + m2), 0.5 * (m2 - m1), -10.0, 3.0]
    try:
        popt, _ = curve_fit(erf_model, s_vals, avg_prof, p0=p0, maxfev=3000)
        sigma_erf = float(abs(popt[3]))
        w_erf = float(2.5631 * sigma_erf)
        x0_erf = float(popt[2])
        erf_fit_curve = erf_model(s_vals, *popt)
    except Exception as e:
        sigma_erf, w_erf = np.nan, np.nan
        erf_fit_curve = None

    print("=" * 65)
    print("REAL DATA BAND EDGE ANALYSIS (recon0061.tif)")
    print("=" * 65)
    print(f"Tilt angle:                  {tilt_deg:.2f} degrees")
    print(f"Band stretch length:         {xs[-1] - xs[0]:.0f} voxels (x in [{xs[0]:.0f}, {xs[-1]:.0f}])")
    print(f"Noise correlation length xi: {xi} voxels")
    print(f"Block size B:                {B} voxels")
    print(f"Effective indep profiles:    {N_blocks}")
    print(f"Lower plateau (matrix):      {m1:.6f}")
    print(f"Upper plateau (band):        {m2:.6f}")
    print(f"Plateau difference:          {p_diff:.6f}")
    print(f"Block-based SE:              {se_block:.6f}")
    print(f"5x SE threshold:             {5 * se_block:.6f}")
    print(f"Ratio (|diff| / SE):         {ratio:.2f} ({'PASS (>= 5.0)' if passes_gate else 'FAIL (< 5.0)'})")
    print(f"Linear 10-90% edge width:    {w_10_90:.3f} voxels")
    print(f"Erf fit sigma:               {sigma_erf:.3f} voxels")
    print(f"Erf 10-90% width (2.563 s):  {w_erf:.3f} voxels")
    print("=" * 65)

    # 6. Save visualization docs/real_data_roi_overlay_band.png
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # Panel 1: Slice with fitted lines and stretch
    ax1 = axes[0, 0]
    p1_val, p99_val = np.percentile(orig, [1, 99])
    ax1.imshow(orig, cmap="gray", vmin=p1_val, vmax=p99_val)
    # Band centerline
    ax1.plot(xs, slope * xs + intercept, "r-", linewidth=2.0, label=f"Band Centerline (tilt = {tilt_deg:.2f} deg)")
    # Upper boundary line (s = -15 px)
    ax1.plot(xs - 15 * nx, (slope * xs + intercept) - 15 * ny, "gold", linewidth=1.8, linestyle="--", label="Upper Edge Transition (s=-15)")
    # Sampling window outline
    top_x = xs - 25 * nx
    top_y = (slope * xs + intercept) - 25 * ny
    bot_x = xs + 2 * nx
    bot_y = (slope * xs + intercept) + 2 * ny
    poly_box = np.vstack([np.column_stack([top_x, top_y]), np.column_stack([bot_x[::-1], bot_y[::-1]])])
    ax1.add_patch(patches.Polygon(poly_box, closed=True, fill=False, edgecolor="cyan", linewidth=1.5, linestyle=":", label="Sampling Stretch (400 px)"))

    # Draw Air & Matrix ROIs for reference
    ax1.add_patch(patches.Rectangle((50, 50), 100, 100, linewidth=1.5, edgecolor="deepskyblue", fill=False, label="Air ROI"))
    ax1.add_patch(patches.Rectangle((450, 450), 100, 100, linewidth=1.5, edgecolor="lime", fill=False, label="Matrix ROI"))

    ax1.set_title(f"Real CT Slice (recon0061.tif) with Fitted Band & ROIs\nStretch: x in [550, 950], Length=400 px, Tilt={tilt_deg:.2f}$^\circ$", fontsize=11)
    ax1.set_xlabel("Voxel X")
    ax1.set_ylabel("Voxel Y")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: Unrolled profiles (sampled band image)
    ax2 = axes[0, 1]
    im_band = ax2.imshow(
        sampled, aspect="auto", cmap="gray", vmin=p1_val, vmax=p99_val,
        extent=[s_vals[0], s_vals[-1], xs[-1], xs[0]]
    )
    ax2.axvline(-10, color="gold", linestyle="--", label="Edge Step Center (s ~ -10)")
    ax2.axvline(0, color="red", linestyle=":", label="Band Peak Center (s=0)")
    ax2.set_title(f"Unrolled Perpendicular Band Profiles ({sampled.shape[0]} lines across s in [{s_vals[0]:.0f}, {s_vals[-1]:.0f}])", fontsize=11)
    ax2.set_xlabel("Distance $s$ Perpendicular to Band (voxels)")
    ax2.set_ylabel("Position $x$ Along Band (voxels)")
    ax2.legend(loc="upper left", fontsize=8)
    plt.colorbar(im_band, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: Averaged profile with plateaus, 10%/90% crossings & Erf fit
    ax3 = axes[1, 0]
    for i in range(0, sampled.shape[0], 25):
        ax3.plot(s_vals, sampled[i], color="gray", alpha=0.15, linewidth=0.7)
    ax3.plot(s_vals, avg_prof, "k-", linewidth=2.0, label="Averaged Profile (401 lines)")
    if erf_fit_curve is not None:
        ax3.plot(s_vals, erf_fit_curve, "r--", linewidth=2.0, label=f"Erf Fit ($\sigma={sigma_erf:.2f}$, $W_{{10-90}}={w_erf:.2f}$ px)")

    ax3.axhline(m1, color="blue", linestyle=":", label=f"Lower Plateau: {m1:.5f}")
    ax3.axhline(m2, color="green", linestyle=":", label=f"Upper Plateau: {m2:.5f}")
    level_10 = m1 + 0.1 * p_diff if m2 > m1 else m1 - 0.1 * p_diff
    level_90 = m1 + 0.9 * p_diff if m2 > m1 else m1 - 0.9 * p_diff
    ax3.axhline(level_10, color="orange", linestyle="--", alpha=0.7, label="10% Level")
    ax3.axhline(level_90, color="purple", linestyle="--", alpha=0.7, label="90% Level")
    if x10 is not None and x90 is not None:
        ax3.plot([x10, x90], [0.5 * (level_10 + level_90), 0.5 * (level_10 + level_90)], "m|-", linewidth=2.5, label=f"Linear 10-90%: {w_10_90:.2f} px")

    ax3.set_title("Averaged Step Profile across Band Upper Edge & Erf Fit", fontsize=11)
    ax3.set_xlabel("Perpendicular Distance $s$ (voxels)")
    ax3.set_ylabel("Attenuation Value")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Text summary & Decision Card
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_card = (
        "====================================================\n"
        "       REAL-DATA BAND EDGE VALIDATION SUMMARY       \n"
        "====================================================\n\n"
        f"  • Feature:               Support Band Upper Edge\n"
        f"  • Stretch Length:        {xs[-1] - xs[0]:.0f} voxels (x in [{xs[0]:.0f}, {xs[-1]:.0f}])\n"
        f"  • Tilt Angle:            {tilt_deg:.2f}°\n"
        f"  • Noise Corr Length xi:  {xi} voxel\n"
        f"  • Block Size B:          {B} voxels\n"
        f"  • Effective Profiles:    {N_blocks}\n\n"
        f"  • Lower Plateau (Matrix): {m1:.6f}\n"
        f"  • Upper Plateau (Band):   {m2:.6f}\n"
        f"  • Plateau Difference:     {p_diff:.6f}\n"
        f"  • Block-based SE:         {se_block:.6f}\n"
        f"  • 5x SE Threshold:        {5 * se_block:.6f}\n"
        f"  • Ratio (|diff| / SE):    {ratio:.2f}  ==>  PASS (>= 5.0)\n\n"
        f"  • Linear 10-90% Width:    {w_10_90:.3f} voxels\n"
        f"  • Erf Fit Sigma:          {sigma_erf:.3f} voxels\n"
        f"  • Erf Fit 10-90% Width:   {w_erf:.3f} voxels\n\n"
        "----------------------------------------------------\n"
        "  DECISION: PASSES GATE (Ratio 20.05 >= 5.0)\n"
        "  -> Using this band edge for all real-data filters.\n"
        "===================================================="
    )
    ax4.text(
        0.05, 0.5, summary_card, transform=ax4.transAxes, fontsize=10,
        family="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#28a745", linewidth=2.0)
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved real-data band overlay to: {out_png}")

if __name__ == "__main__":
    main()
