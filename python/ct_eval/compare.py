import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure build dir is importable
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import ct
from ct_eval.phantom import generate_phantom_3d, extract_planar_edge_profiles
from ct_eval.noise import add_gaussian_noise, add_salt_and_pepper_noise
from ct_eval.metrics import rmse, psnr, ssim, roi_noise_std, edge_width_10_90, z_consistency

def time_filter(fn, sample_input, warmup=1, runs=5):
    """Measures execution time: 1 warm-up run and median of 5 timed runs."""
    for _ in range(warmup):
        _ = fn(sample_input)
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = fn(sample_input)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    return float(np.median(timings))

def get_filter_configs():
    """Generates the parameter sweep configurations for all 8 required filters."""
    configs = []

    # 1. Gaussian: sigma in {0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0}
    for sigma in [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]:
        radius = max(2, int(np.ceil(3.0 * sigma)))
        param_str = f"sigma={sigma:.1f};r={radius}"
        configs.append({
            "family": "Gaussian",
            "mode": "slice",
            "param_str": param_str,
            "fn": lambda x, s=sigma, r=radius: ct.gaussian_filter_slice_by_slice(x, sigma=s, radius=r)
        })
        configs.append({
            "family": "Gaussian",
            "mode": "3d",
            "param_str": param_str,
            "fn": lambda x, s=sigma, r=radius: ct.gaussian_filter_3d(x, sigma=s, radius=r)
        })

    # 2. Median: radius in {1, 2}
    for radius in [1, 2]:
        param_str = f"radius={radius}"
        configs.append({
            "family": "Median",
            "mode": "slice",
            "param_str": param_str,
            "fn": lambda x, r=radius: ct.median_filter_slice_by_slice(x, radius=r)
        })
        configs.append({
            "family": "Median",
            "mode": "3d",
            "param_str": param_str,
            "fn": lambda x, r=radius: ct.median_filter_3d(x, radius=r)
        })

    # 3. Bilateral: spatial sigma in {1.0, 1.5, 2.0, 2.5} x range sigma in {0.04, 0.08, 0.12, 0.16, 0.20}
    for ss in [1.0, 1.5, 2.0, 2.5]:
        for sr in [0.04, 0.08, 0.12, 0.16, 0.20]:
            radius = max(2, int(np.ceil(2.0 * ss)))
            param_str = f"ss={ss:.1f};sr={sr:.2f};r={radius}"
            configs.append({
                "family": "Bilateral",
                "mode": "slice",
                "param_str": param_str,
                "fn": lambda x, s_s=ss, s_r=sr, r=radius: ct.bilateral_filter_slice_by_slice(x, sigma_spatial=s_s, sigma_range=s_r, radius=r)
            })
            configs.append({
                "family": "Bilateral",
                "mode": "3d",
                "param_str": param_str,
                "fn": lambda x, s_s=ss, s_r=sr, r=radius: ct.bilateral_filter_3d(x, sigma_spatial=s_s, sigma_range=s_r, radius=r)
            })

    # 4. Diffusion: kappa in {0.02, 0.04, 0.08, 0.12, 0.16} x T in {0.6, 1.2, 2.4, 3.6}
    for kappa in [0.02, 0.04, 0.08, 0.12, 0.16]:
        for T in [0.6, 1.2, 2.4, 3.6]:
            iters_slice = int(np.round(T / 0.20))
            iters_3d = int(np.round(T / 0.15))
            param_str = f"kappa={kappa:.2f};T={T:.1f}"
            configs.append({
                "family": "Diffusion",
                "mode": "slice",
                "param_str": param_str,
                "fn": lambda x, k=kappa, it=iters_slice: ct.anisotropic_diffusion_slice_by_slice(x, kappa=k, iterations=it, dt=0.20)
            })
            configs.append({
                "family": "Diffusion",
                "mode": "3d",
                "param_str": param_str,
                "fn": lambda x, k=kappa, it=iters_3d: ct.anisotropic_diffusion_3d(x, kappa=k, iterations=it, dt=0.15)
            })

    # 5. Non-Local Means (NLM): patch_radius=1, search_radius=2, h in {0.03, 0.05, 0.08, 0.10}
    for h in [0.03, 0.05, 0.08, 0.10]:
        param_str = f"p=1;s=2;h={h:.2f}"
        configs.append({
            "family": "NLM",
            "mode": "slice",
            "param_str": param_str,
            "fn": lambda x, h_val=h: ct.nlm_filter_slice_by_slice(x, patch_radius=1, search_radius=2, h=h_val)
        })
        configs.append({
            "family": "NLM",
            "mode": "3d",
            "param_str": param_str,
            "fn": lambda x, h_val=h: ct.nlm_filter_3d(x, patch_radius=1, search_radius=2, h=h_val)
        })

    # 6. Adaptive Wiener: radius in {1, 2, 3}
    for radius in [1, 2, 3]:
        param_str = f"radius={radius}"
        configs.append({
            "family": "Wiener",
            "mode": "slice",
            "param_str": param_str,
            "fn": lambda x, r=radius: ct.wiener_filter_slice_by_slice(x, radius=r, noise_variance=-1.0)
        })
        configs.append({
            "family": "Wiener",
            "mode": "3d",
            "param_str": param_str,
            "fn": lambda x, r=radius: ct.wiener_filter_3d(x, radius=r, noise_variance=-1.0)
        })

    # 7. Total Variation (ROF Chambolle): weight in {0.01, 0.02, 0.05, 0.10, 0.20}, eps=1e-4, iters=40
    for weight in [0.01, 0.02, 0.05, 0.10, 0.20]:
        param_str = f"weight={weight:.2f};iters=40"
        configs.append({
            "family": "TV_Chambolle",
            "mode": "slice",
            "param_str": param_str,
            "fn": lambda x, w=weight: ct.tv_chambolle_slice_by_slice(x, weight=w, eps=1e-4, max_iterations=40)
        })
        configs.append({
            "family": "TV_Chambolle",
            "mode": "3d",
            "param_str": param_str,
            "fn": lambda x, w=weight: ct.tv_chambolle_3d(x, weight=w, eps=1e-4, max_iterations=40)
        })

    # 8. Morphology: Opening and Closing with box and ball (radius=1)
    for op_name, fn_slice, fn_3d in [
        ("Opening", ct.grayscale_opening_slice_by_slice, ct.grayscale_opening_3d),
        ("Closing", ct.grayscale_closing_slice_by_slice, ct.grayscale_closing_3d),
    ]:
        for elem in ["box", "ball"]:
            param_str = f"op={op_name.lower()};r=1;elem={elem}"
            configs.append({
                "family": "Morphology",
                "mode": "slice",
                "param_str": param_str,
                "fn": lambda x, f=fn_slice, el=elem: f(x, radius=1, element=el)
            })
            configs.append({
                "family": "Morphology",
                "mode": "3d",
                "param_str": param_str,
                "fn": lambda x, f=fn_3d, el=elem: f(x, radius=1, element=el)
            })

    return configs

    return configs

def evaluate_volume(clean, noisy, filtered, masks, data_range):
    roi_homo = masks["homogeneous_roi_1"]
    
    val_rmse = rmse(clean, filtered)
    val_psnr = psnr(clean, filtered, data_range)
    val_ssim = ssim(clean, filtered, data_range)
    val_roi_noise = roi_noise_std(filtered, roi_homo)
    val_z_cons = z_consistency(filtered, roi_homo)

    profiles = extract_planar_edge_profiles(filtered, masks)
    val_edge_w = edge_width_10_90(profiles)

    return {
        "rmse": val_rmse,
        "psnr": val_psnr,
        "ssim": val_ssim,
        "roi_noise": val_roi_noise,
        "z_consistency": val_z_cons,
        "edge_width": val_edge_w
    }

def run_sweeps():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    seeds = [42, 100, 2024]
    noise_sigma = 0.04

    print("=" * 75)
    print("TASK B5: Multi-Filter Sweeps on 64^3 Phantom (Gaussian & Salt-and-Pepper Noise)")
    print("=" * 75)

    clean_64, masks_64 = generate_phantom_3d((64, 64, 64))
    data_range_64 = float(clean_64.max() - clean_64.min())

    clean_profiles_64 = extract_planar_edge_profiles(clean_64, masks_64)
    clean_edge_w_64 = edge_width_10_90(clean_profiles_64)
    print(f"Clean 64^3 Planar Edge Width: {clean_edge_w_64:.3f} voxels")

    # =========================================================================
    # SCENARIO 1: Gaussian Noise (sigma = 0.04)
    # =========================================================================
    print("\n--- SCENARIO 1: Gaussian Noise (sigma = 0.04) ---")
    noisy_gauss_64 = {seed: add_gaussian_noise(clean_64, sigma=noise_sigma, seed=seed) for seed in seeds}

    base_gauss_list = [evaluate_volume(clean_64, noisy_gauss_64[s], noisy_gauss_64[s], masks_64, data_range_64) for s in seeds]
    base_gauss_agg = {
        k: (float(np.mean([m[k] for m in base_gauss_list])), float(np.std([m[k] for m in base_gauss_list])))
        for k in base_gauss_list[0].keys()
    }
    print(f"Noisy Gaussian Baseline 64^3: RMSE={base_gauss_agg['rmse'][0]:.5f}, PSNR={base_gauss_agg['psnr'][0]:.2f}dB, SSIM={base_gauss_agg['ssim'][0]:.4f}")

    configs = get_filter_configs()
    sweep_results_gauss = []

    print(f"Sweeping {len(configs)} configurations across 8 filter families on 64^3 (Gaussian noise)...")
    for idx, cfg in enumerate(configs):
        family = cfg["family"]
        mode = cfg["mode"]
        param_str = cfg["param_str"]
        fn = cfg["fn"]

        # Benchmark runtime (seed 42)
        runtime_s = time_filter(fn, noisy_gauss_64[42], warmup=1, runs=5)
        runtime_ms = runtime_s * 1000.0

        seed_evals = []
        for seed in seeds:
            filtered = fn(noisy_gauss_64[seed])
            m = evaluate_volume(clean_64, noisy_gauss_64[seed], filtered, masks_64, data_range_64)
            seed_evals.append(m)

        res = {
            "noise_scenario": "Gaussian",
            "noise_level": "sigma=0.04",
            "family": family,
            "mode": mode,
            "param_str": param_str,
            "size": "64^3",
            "runtime_ms": runtime_ms,
        }
        for k in ["rmse", "psnr", "ssim", "roi_noise", "z_consistency", "edge_width"]:
            vals = [e[k] for e in seed_evals]
            res[f"{k}_mean"] = float(np.mean(vals))
            res[f"{k}_std"] = float(np.std(vals))

        sweep_results_gauss.append(res)
        print(f"[{idx+1:02d}/{len(configs):02d}] {family:12s} | {mode:5s} | {param_str:25s} | "
              f"PSNR={res['psnr_mean']:.2f}dB | SSIM={res['ssim_mean']:.4f} | EdgeW={res['edge_width_mean']:.2f}px | Time={runtime_ms:.1f}ms")

    # =========================================================================
    # SCENARIO 2: Salt-and-Pepper Noise (1% and 2% corruptions)
    # =========================================================================
    print("\n--- SCENARIO 2: Salt-and-Pepper Noise (1% and 2% Corrupted Voxels) ---")
    sp_amounts = [0.01, 0.02]
    sweep_results_sp = []

    # Selected representative setting for each filter family to evaluate under impulse noise
    sp_filter_candidates = [
        {"family": "Gaussian", "mode": "3d", "param_str": "sigma=0.8,r=3", "fn": lambda x: ct.gaussian_filter_3d(x, sigma=0.8, radius=3)},
        {"family": "Median", "mode": "3d", "param_str": "radius=1", "fn": lambda x: ct.median_filter_3d(x, radius=1)},
        {"family": "Bilateral", "mode": "3d", "param_str": "ss=1.5,sr=0.12,r=3", "fn": lambda x: ct.bilateral_filter_3d(x, 1.5, 0.12, 3)},
        {"family": "Diffusion", "mode": "3d", "param_str": "kappa=0.08,T=1.2", "fn": lambda x: ct.anisotropic_diffusion_3d(x, 0.08, 8, 0.15)},
        {"family": "NLM", "mode": "3d", "param_str": "p=1,s=2,h=0.05", "fn": lambda x: ct.nlm_filter_3d(x, 1, 2, 0.05)},
        {"family": "Wiener", "mode": "3d", "param_str": "radius=1", "fn": lambda x: ct.wiener_filter_3d(x, radius=1, noise_variance=-1.0)},
        {"family": "TV_Chambolle", "mode": "3d", "param_str": "weight=0.10,iters=40", "fn": lambda x: ct.tv_chambolle_3d(x, 0.10, 1e-4, 40)},
        {"family": "Morphology_Opening", "mode": "3d", "param_str": "op=opening,r=1,box", "fn": lambda x: ct.grayscale_opening_3d(x, radius=1, element="box")},
        {"family": "Morphology_Closing", "mode": "3d", "param_str": "op=closing,r=1,box", "fn": lambda x: ct.grayscale_closing_3d(x, radius=1, element="box")},
        {"family": "Morphology_OpenClose", "mode": "3d", "param_str": "op=open_close,r=1,box", "fn": lambda x: ct.grayscale_closing_3d(ct.grayscale_opening_3d(x, 1, "box"), 1, "box")},
    ]

    sp_baselines = {}
    for amount in sp_amounts:
        pct_label = f"{int(amount*100)}%"
        noisy_sp = {seed: add_salt_and_pepper_noise(clean_64, amount=amount, seed=seed) for seed in seeds}
        base_list = [evaluate_volume(clean_64, noisy_sp[s], noisy_sp[s], masks_64, data_range_64) for s in seeds]
        sp_baselines[pct_label] = {
            k: (float(np.mean([m[k] for m in base_list])), float(np.std([m[k] for m in base_list])))
            for k in base_list[0].keys()
        }
        print(f"Noisy S&P Baseline ({pct_label}): RMSE={sp_baselines[pct_label]['rmse'][0]:.5f}, "
              f"PSNR={sp_baselines[pct_label]['psnr'][0]:.2f}dB, SSIM={sp_baselines[pct_label]['ssim'][0]:.4f}")

        for cand in sp_filter_candidates:
            fam = cand["family"]
            mode = cand["mode"]
            pstr = cand["param_str"]
            fn = cand["fn"]

            runtime_s = time_filter(fn, noisy_sp[42], warmup=1, runs=3)
            runtime_ms = runtime_s * 1000.0

            seed_evals = []
            for seed in seeds:
                filtered = fn(noisy_sp[seed])
                m = evaluate_volume(clean_64, noisy_sp[seed], filtered, masks_64, data_range_64)
                seed_evals.append(m)

            res = {
                "noise_scenario": "Salt_Pepper",
                "noise_level": pct_label,
                "family": fam,
                "mode": mode,
                "param_str": pstr,
                "size": "64^3",
                "runtime_ms": runtime_ms,
            }
            for k in ["rmse", "psnr", "ssim", "roi_noise", "z_consistency", "edge_width"]:
                vals = [e[k] for e in seed_evals]
                res[f"{k}_mean"] = float(np.mean(vals))
                res[f"{k}_std"] = float(np.std(vals))

            sweep_results_sp.append(res)
            print(f"[S&P {pct_label}] {fam:20s} | SSIM={res['ssim_mean']:.4f} | PSNR={res['psnr_mean']:.2f}dB | "
                  f"EdgeW={res['edge_width_mean']:.2f}px | RMSE={res['rmse_mean']:.5f}")

    # =========================================================================
    # SCENARIO 3: Best Filter Settings on 128^3 Phantom (Gaussian Noise)
    # =========================================================================
    print("\n" + "=" * 75)
    print("TASK B5: Running Best Filter Settings on 128^3 Phantom (Gaussian Noise)")
    print("=" * 75)

    clean_128, masks_128 = generate_phantom_3d((128, 128, 128))
    data_range_128 = float(clean_128.max() - clean_128.min())
    noisy_volumes_128 = {seed: add_gaussian_noise(clean_128, sigma=noise_sigma, seed=seed) for seed in seeds}

    base_128_list = [evaluate_volume(clean_128, noisy_volumes_128[s], noisy_volumes_128[s], masks_128, data_range_128) for s in seeds]
    base_128_agg = {
        k: (float(np.mean([m[k] for m in base_128_list])), float(np.std([m[k] for m in base_128_list])))
        for k in base_128_list[0].keys()
    }
    print(f"Noisy Baseline 128^3: RMSE={base_128_agg['rmse'][0]:.5f}, PSNR={base_128_agg['psnr'][0]:.2f}dB")

    results_128 = []
    filter_families = ["Gaussian", "Median", "Bilateral", "Diffusion", "NLM", "Wiener", "TV_Chambolle", "Morphology"]
    for fam in filter_families:
        for mode in ["slice", "3d"]:
            fam_configs = [r for r in sweep_results_gauss if r["family"] == fam and r["mode"] == mode]
            if not fam_configs:
                continue
            best = min(fam_configs, key=lambda x: x["rmse_mean"])

            matching_cfg = next(c for c in configs if c["family"] == fam and c["mode"] == mode and c["param_str"] == best["param_str"])
            fn = matching_cfg["fn"]

            runtime_s = time_filter(fn, noisy_volumes_128[42], warmup=1, runs=3)
            runtime_ms = runtime_s * 1000.0

            seed_evals_128 = []
            for seed in seeds:
                filt = fn(noisy_volumes_128[seed])
                m = evaluate_volume(clean_128, noisy_volumes_128[seed], filt, masks_128, data_range_128)
                seed_evals_128.append(m)

            res128 = {
                "noise_scenario": "Gaussian",
                "noise_level": "sigma=0.04",
                "family": fam,
                "mode": mode,
                "param_str": f"BEST({best['param_str']})",
                "size": "128^3",
                "runtime_ms": runtime_ms,
            }
            for k in ["rmse", "psnr", "ssim", "roi_noise", "z_consistency", "edge_width"]:
                vals = [e[k] for e in seed_evals_128]
                res128[f"{k}_mean"] = float(np.mean(vals))
                res128[f"{k}_std"] = float(np.std(vals))

            results_128.append(res128)
            print(f"[128^3] {fam:12s} | {mode:5s} | {best['param_str']:22s} | "
                  f"PSNR={res128['psnr_mean']:.2f}dB | Time={runtime_ms:.1f}ms")

    # =========================================================================
    # SAVE docs/sweep_results.csv (Using csv.DictWriter with quoted fields)
    # =========================================================================
    import csv
    all_results = sweep_results_gauss + sweep_results_sp + results_128
    csv_path = os.path.join(docs_dir, "sweep_results.csv")
    fieldnames = [
        "noise_scenario", "noise_level", "filter", "mode", "param_str", "size",
        "runtime_ms", "rmse_mean", "rmse_std", "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std", "roi_noise_mean", "roi_noise_std",
        "edge_width_mean", "edge_width_std", "z_consistency_mean", "z_consistency_std"
    ]
    with open(csv_path, "w", newline="") as f:
        f.write("# Protocol: Parameter sweep on 64^3 and 128^3 phantom, Gaussian (sigma=0.04) and Salt-and-Pepper (1%, 2%), 3 noise seeds\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        # Baseline entries
        writer.writerow({
            "noise_scenario": "Gaussian", "noise_level": "sigma=0.04", "filter": "Noisy_Baseline",
            "mode": "none", "param_str": "none", "size": "64^3", "runtime_ms": 0.0,
            "rmse_mean": f"{base_gauss_agg['rmse'][0]:.6f}", "rmse_std": f"{base_gauss_agg['rmse'][1]:.6f}",
            "psnr_mean": f"{base_gauss_agg['psnr'][0]:.2f}", "psnr_std": f"{base_gauss_agg['psnr'][1]:.2f}",
            "ssim_mean": f"{base_gauss_agg['ssim'][0]:.4f}", "ssim_std": f"{base_gauss_agg['ssim'][1]:.4f}",
            "roi_noise_mean": f"{base_gauss_agg['roi_noise'][0]:.6f}", "roi_noise_std": f"{base_gauss_agg['roi_noise'][1]:.6f}",
            "edge_width_mean": f"{base_gauss_agg['edge_width'][0]:.3f}", "edge_width_std": f"{base_gauss_agg['edge_width'][1]:.3f}",
            "z_consistency_mean": f"{base_gauss_agg['z_consistency'][0]:.6f}", "z_consistency_std": f"{base_gauss_agg['z_consistency'][1]:.6f}"
        })
        writer.writerow({
            "noise_scenario": "Gaussian", "noise_level": "sigma=0.04", "filter": "Noisy_Baseline",
            "mode": "none", "param_str": "none", "size": "128^3", "runtime_ms": 0.0,
            "rmse_mean": f"{base_128_agg['rmse'][0]:.6f}", "rmse_std": f"{base_128_agg['rmse'][1]:.6f}",
            "psnr_mean": f"{base_128_agg['psnr'][0]:.2f}", "psnr_std": f"{base_128_agg['psnr'][1]:.2f}",
            "ssim_mean": f"{base_128_agg['ssim'][0]:.4f}", "ssim_std": f"{base_128_agg['ssim'][1]:.4f}",
            "roi_noise_mean": f"{base_128_agg['roi_noise'][0]:.6f}", "roi_noise_std": f"{base_128_agg['roi_noise'][1]:.6f}",
            "edge_width_mean": f"{base_128_agg['edge_width'][0]:.3f}", "edge_width_std": f"{base_128_agg['edge_width'][1]:.3f}",
            "z_consistency_mean": f"{base_128_agg['z_consistency'][0]:.6f}", "z_consistency_std": f"{base_128_agg['z_consistency'][1]:.6f}"
        })
        for amt_lbl in ["1%", "2%"]:
            b = sp_baselines[amt_lbl]
            writer.writerow({
                "noise_scenario": "Salt_Pepper", "noise_level": amt_lbl, "filter": "Noisy_Baseline",
                "mode": "none", "param_str": "none", "size": "64^3", "runtime_ms": 0.0,
                "rmse_mean": f"{b['rmse'][0]:.6f}", "rmse_std": f"{b['rmse'][1]:.6f}",
                "psnr_mean": f"{b['psnr'][0]:.2f}", "psnr_std": f"{b['psnr'][1]:.2f}",
                "ssim_mean": f"{b['ssim'][0]:.4f}", "ssim_std": f"{b['ssim'][1]:.4f}",
                "roi_noise_mean": f"{b['roi_noise'][0]:.6f}", "roi_noise_std": f"{b['roi_noise'][1]:.6f}",
                "edge_width_mean": f"{b['edge_width'][0]:.3f}", "edge_width_std": f"{b['edge_width'][1]:.3f}",
                "z_consistency_mean": f"{b['z_consistency'][0]:.6f}", "z_consistency_std": f"{b['z_consistency'][1]:.6f}"
            })

        for r in all_results:
            writer.writerow({
                "noise_scenario": r['noise_scenario'],
                "noise_level": r['noise_level'],
                "filter": r['family'],
                "mode": r['mode'],
                "param_str": r['param_str'],
                "size": r['size'],
                "runtime_ms": f"{r['runtime_ms']:.2f}",
                "rmse_mean": f"{r['rmse_mean']:.6f}",
                "rmse_std": f"{r['rmse_std']:.6f}",
                "psnr_mean": f"{r['psnr_mean']:.2f}",
                "psnr_std": f"{r['psnr_std']:.2f}",
                "ssim_mean": f"{r['ssim_mean']:.4f}",
                "ssim_std": f"{r['ssim_std']:.4f}",
                "roi_noise_mean": f"{r['roi_noise_mean']:.6f}",
                "roi_noise_std": f"{r['roi_noise_std']:.6f}",
                "edge_width_mean": f"{r['edge_width_mean']:.3f}",
                "edge_width_std": f"{r['edge_width_std']:.3f}",
                "z_consistency_mean": f"{r['z_consistency_mean']:.6f}",
                "z_consistency_std": f"{r['z_consistency_std']:.6f}"
            })
    print(f"\nSaved sweep results to: {csv_path}")

    # =========================================================================
    # PLOT 1: docs/noise_vs_edge_width.png
    # =========================================================================
    colors = {
        "Gaussian": "#e74c3c",
        "Median": "#e67e22",
        "Bilateral": "#2980b9",
        "Diffusion": "#27ae60",
        "NLM": "#8e44ad",
        "Wiener": "#16a085",
        "TV_Chambolle": "#d35400",
        "Morphology": "#2c3e50"
    }

    fig_nw, ax_nw = plt.subplots(figsize=(10, 6.5))
    ax_nw.scatter([base_gauss_agg["edge_width"][0]], [base_gauss_agg["roi_noise"][0]],
                  color="black", marker="x", s=140, linewidth=2.5, label="Noisy Baseline (Gaussian)", zorder=5)

    for fam, col in colors.items():
        for mode, ls, mk in [("slice", "--", "s"), ("3d", "-", "o")]:
            pts = [r for r in sweep_results_gauss if r["family"] == fam and r["mode"] == mode]
            if not pts:
                continue
            pts = sorted(pts, key=lambda x: x["edge_width_mean"])
            xs = [p["edge_width_mean"] for p in pts]
            ys = [p["roi_noise_mean"] for p in pts]
            label = f"{fam} ({mode})"
            ax_nw.plot(xs, ys, linestyle=ls, marker=mk, color=col, label=label, alpha=0.85, linewidth=1.5, markersize=6)

    ax_nw.set_xlabel("Planar Edge Width (voxels) [Lower = Sharper Edges]", fontsize=11)
    ax_nw.set_ylabel("ROI Noise Std Dev [Lower = Better Denoising]", fontsize=11)
    ax_nw.set_title("Noise Suppression vs. Edge Preservation Trade-Off (8 Filters, 64^3 Phantom)", fontsize=12)
    ax_nw.grid(True, linestyle="--", alpha=0.5)
    ax_nw.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    nw_path = os.path.join(docs_dir, "noise_vs_edge_width.png")
    fig_nw.savefig(nw_path, dpi=200)
    plt.close(fig_nw)
    print(f"Saved figure: {nw_path}")

    # =========================================================================
    # PLOT 2: docs/rmse_vs_runtime.png
    # =========================================================================
    fig_rr, ax_rr = plt.subplots(figsize=(10, 6.5))
    ax_rr.axhline(base_gauss_agg["rmse"][0], color="black", linestyle=":", label="Noisy Baseline RMSE", alpha=0.8)

    for fam, col in colors.items():
        for mode, mk in [("slice", "s"), ("3d", "o")]:
            pts = [r for r in sweep_results_gauss if r["family"] == fam and r["mode"] == mode]
            if not pts:
                continue
            xs = [p["runtime_ms"] for p in pts]
            ys = [p["rmse_mean"] for p in pts]
            ax_rr.scatter(xs, ys, color=col, marker=mk, label=f"{fam} ({mode})", s=70, alpha=0.85)

    ax_rr.set_xscale("log")
    ax_rr.set_xlabel("Execution Time per Volume (ms) [Log Scale]", fontsize=11)
    ax_rr.set_ylabel("Reconstruction RMSE [Lower = Higher Quality]", fontsize=11)
    ax_rr.set_title("Filter Efficiency: RMSE vs. Computational Runtime (8 Filters, 64^3 Phantom)", fontsize=12)
    ax_rr.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_rr.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    rr_path = os.path.join(docs_dir, "rmse_vs_runtime.png")
    fig_rr.savefig(rr_path, dpi=200)
    plt.close(fig_rr)
    print(f"Saved figure: {rr_path}")

    # =========================================================================
    # PLOT 3: docs/salt_pepper_ranking.png
    # =========================================================================
    fig_sp, (ax_sp1, ax_sp2) = plt.subplots(1, 2, figsize=(14, 6))

    sp_1pct = [r for r in sweep_results_sp if r["noise_level"] == "1%"]
    sp_1pct_sorted = sorted(sp_1pct, key=lambda x: x["ssim_mean"], reverse=True)

    labels = [r["family"].replace("Morphology_", "Morph_") for r in sp_1pct_sorted]
    ssim_vals = [r["ssim_mean"] for r in sp_1pct_sorted]
    psnr_vals = [r["psnr_mean"] for r in sp_1pct_sorted]

    y_pos = np.arange(len(labels))
    colors_bar = ["#2ecc71" if "Median" in l or "Morph" in l else "#e74c3c" if "Gaussian" in l or "Wiener" in l else "#3498db" for l in labels]

    ax_sp1.barh(y_pos, ssim_vals, color=colors_bar, alpha=0.85)
    ax_sp1.axvline(sp_baselines["1%"]["ssim"][0], color="black", linestyle="--", label=f"Baseline ({sp_baselines['1%']['ssim'][0]:.3f})")
    ax_sp1.set_yticks(y_pos)
    ax_sp1.set_yticklabels(labels, fontsize=9)
    ax_sp1.invert_yaxis()
    ax_sp1.set_xlabel("SSIM [Higher = Better Structure Preservation]", fontsize=10)
    ax_sp1.set_title("Structural Preservation (SSIM) under 1% Salt & Pepper", fontsize=11)
    ax_sp1.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax_sp1.legend(loc="lower right")

    ax_sp2.barh(y_pos, psnr_vals, color=colors_bar, alpha=0.85)
    ax_sp2.axvline(sp_baselines["1%"]["psnr"][0], color="black", linestyle="--", label=f"Baseline ({sp_baselines['1%']['psnr'][0]:.1f} dB)")
    ax_sp2.set_yticks(y_pos)
    ax_sp2.set_yticklabels(labels, fontsize=9)
    ax_sp2.invert_yaxis()
    ax_sp2.set_xlabel("PSNR (dB) [Higher = Lower Reconstruction Error]", fontsize=10)
    ax_sp2.set_title("Reconstruction Quality (PSNR) under 1% Salt & Pepper", fontsize=11)
    ax_sp2.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax_sp2.legend(loc="lower right")

    plt.tight_layout()
    sp_fig_path = os.path.join(docs_dir, "salt_pepper_ranking.png")
    fig_sp.savefig(sp_fig_path, dpi=200)
    plt.close(fig_sp)
    print(f"Saved figure: {sp_fig_path}")

    # Print summary ranking table for Salt and Pepper
    print("\n" + "=" * 75)
    print("SALT-AND-PEPPER NOISE (1%) RANKING BY STRUCTURAL METRICS (SSIM & EDGE WIDTH):")
    print(f"{'Rank':4s} | {'Filter':20s} | {'SSIM':8s} | {'PSNR (dB)':10s} | {'EdgeW (px)':10s} | {'RMSE':8s}")
    print("-" * 75)
    for rank, r in enumerate(sp_1pct_sorted, 1):
        print(f"{rank:4d} | {r['family']:20s} | {r['ssim_mean']:.4f}   | {r['psnr_mean']:6.2f}     | {r['edge_width_mean']:6.2f}     | {r['rmse_mean']:.5f}")
    print("=" * 75)

if __name__ == "__main__":
    run_sweeps()
