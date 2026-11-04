#!/usr/bin/env python3
import pandas as pd
import numpy as np

def main():
    sw = pd.read_csv("docs/sweep_results.csv", comment="#")
    rd = pd.read_csv("docs/real_data_results.csv", comment="#")
    bm = pd.read_csv("docs/benchmark_results.csv", comment="#")
    s3d = pd.read_csv("docs/slice_vs_3d_benchmark.csv", comment="#")

    print("=" * 80)
    print("PROGRAMMATIC REPORT DATA EXTRACTION")
    print("=" * 80)

    # 1. Gaussian 64^3 (3D)
    g64_3d = sw[(sw["noise_scenario"] == "Gaussian") & (sw["size"] == "64^3") & (sw["mode"] == "3d")]
    g64_slice = sw[(sw["noise_scenario"] == "Gaussian") & (sw["size"] == "64^3") & (sw["mode"] == "slice")]

    print("\n--- GAUSSIAN 64^3 3D LEADERS ---")
    best_ssim_64 = g64_3d.loc[g64_3d["ssim_mean"].idxmax()]
    print(f"Highest SSIM at 64^3: {best_ssim_64['filter']} ({best_ssim_64['param_str']}) -> SSIM = {best_ssim_64['ssim_mean']:.4f}")

    best_psnr_64 = g64_3d.loc[g64_3d["psnr_mean"].idxmax()]
    print(f"Highest PSNR at 64^3: {best_psnr_64['filter']} ({best_psnr_64['param_str']}) -> PSNR = {best_psnr_64['psnr_mean']:.2f} dB")

    best_rmse_64 = g64_3d.loc[g64_3d["rmse_mean"].idxmin()]
    print(f"Lowest RMSE at 64^3:  {best_rmse_64['filter']} ({best_rmse_64['param_str']}) -> RMSE = {best_rmse_64['rmse_mean']:.6f}")

    best_roi_64 = g64_3d.loc[g64_3d["roi_noise_mean"].idxmin()]
    print(f"Lowest ROI Noise at 64^3: {best_roi_64['filter']} ({best_roi_64['param_str']}) -> ROI Noise = {best_roi_64['roi_noise_mean']:.6f}")

    # TV Chambolle specifics at 64^3
    tv_64 = g64_3d[g64_3d["filter"] == "TV_Chambolle"]
    best_tv_ssim = tv_64.loc[tv_64["ssim_mean"].idxmax()]
    print(f"TV Chambolle 64^3: max SSIM = {best_tv_ssim['ssim_mean']:.4f} ({best_tv_ssim['param_str']}), PSNR = {best_tv_ssim['psnr_mean']:.2f} dB, ROI Noise = {best_tv_ssim['roi_noise_mean']:.6f}")

    # Gaussian sigma=0.4 vs baseline
    b64 = sw[(sw["filter"] == "Noisy_Baseline") & (sw["size"] == "64^3")].iloc[0]
    b128 = sw[(sw["filter"] == "Noisy_Baseline") & (sw["size"] == "128^3")].iloc[0]
    g64_04 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "64^3") & (sw["param_str"] == "sigma=0.4;r=2")].iloc[0]
    g128_04 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "128^3")].iloc[0]
    print(f"\n--- GAUSSIAN sigma=0.4 VS NOISY BASELINE ---")
    print(f"64^3:  PSNR {b64['psnr_mean']:.2f} -> {g64_04['psnr_mean']:.2f} dB | SSIM {b64['ssim_mean']:.4f} -> {g64_04['ssim_mean']:.4f} | Edge {b64['edge_width_mean']:.3f} -> {g64_04['edge_width_mean']:.3f} vx")
    print(f"128^3: PSNR {b128['psnr_mean']:.2f} -> {g128_04['psnr_mean']:.2f} dB | SSIM {b128['ssim_mean']:.4f} -> {g128_04['ssim_mean']:.4f} | Edge {b128['edge_width_mean']:.3f} -> {g128_04['edge_width_mean']:.3f} vx")

    # 2. Gaussian 128^3 (3D)
    g128_3d = sw[(sw["noise_scenario"] == "Gaussian") & (sw["size"] == "128^3") & (sw["mode"] == "3d")]
    print("\n--- GAUSSIAN 128^3 3D LEADERS ---")
    best_ssim_128 = g128_3d.loc[g128_3d["ssim_mean"].idxmax()]
    print(f"Highest SSIM at 128^3: {best_ssim_128['filter']} ({best_ssim_128['param_str']}) -> SSIM = {best_ssim_128['ssim_mean']:.4f}")
    best_psnr_128 = g128_3d.loc[g128_3d["psnr_mean"].idxmax()]
    print(f"Highest PSNR at 128^3: {best_psnr_128['filter']} ({best_psnr_128['param_str']}) -> PSNR = {best_psnr_128['psnr_mean']:.2f} dB")
    tv_128 = g128_3d[g128_3d["filter"] == "TV_Chambolle"]
    if len(tv_128) > 0:
        tv_128_row = tv_128.iloc[0]
        print(f"TV Chambolle 128^3: SSIM = {tv_128_row['ssim_mean']:.4f}, PSNR = {tv_128_row['psnr_mean']:.2f} dB, ROI Noise = {tv_128_row['roi_noise_mean']:.6f}")

    # 3. Salt & Pepper (1% and 2%)
    print("\n--- SALT & PEPPER (1%) RANKING ---")
    sp1 = sw[(sw["noise_scenario"] == "Salt_Pepper") & (sw["noise_level"] == "1%") & (sw["filter"] != "Noisy_Baseline")].copy()
    sp1.sort_values(by="ssim_mean", ascending=False, inplace=True)
    for idx, (_, r) in enumerate(sp1.iterrows()):
        print(f"Rank {idx+1:2d} | {r['filter']:22s} | SSIM = {r['ssim_mean']:.4f} | PSNR = {r['psnr_mean']:.2f} dB | RMSE = {r['rmse_mean']:.5f} | Edge = {r['edge_width_mean']:.3f} | ROI Noise = {r['roi_noise_mean']:.6f}")

    # Baseline for SP1
    base_sp1 = sw[(sw["noise_scenario"] == "Salt_Pepper") & (sw["noise_level"] == "1%") & (sw["filter"] == "Noisy_Baseline")].iloc[0]
    print(f"Noisy S&P 1% Baseline: SSIM = {base_sp1['ssim_mean']:.4f} | PSNR = {base_sp1['psnr_mean']:.2f} dB | RMSE = {base_sp1['rmse_mean']:.5f}")

    # 4. Real Data
    print("\n--- REAL DATA METRICS ---")
    for _, r in rd.iterrows():
        print(f"{r['filter']:32s} | Air Red = {r['air_noise_red_pct']:5.2f}% | Mat Red = {r['matrix_noise_red_pct']:5.2f}% | Erf W = {r['edge_width_erf']:5.2f} px (sigma={r['erf_sigma']:.3f}) | Lin W = {r['edge_width_linear']:5.2f} px | Leak = {r['leakage_ratio']:.3f}x")

if __name__ == "__main__":
    main()
