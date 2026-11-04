#!/usr/bin/env python3
import subprocess
import os
import sys

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    benchmark_bin = os.path.join(project_root, "build", "benchmark")
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    filters = [
        "gaussian",
        "median",
        "bilateral",
        "diffusion",
        "nlm",
        "wiener",
        "tv",
        "morphology"
    ]

    out_csv = os.path.join(docs_dir, "slice_vs_3d_benchmark.csv")
    results = []

    print("=" * 80)
    print("PURE C++ SLICE-WISE VS TRUE 3D BENCHMARKS (12 Threads, 64^3 Phantom)")
    print("=" * 80)
    print(f"{'Filter':15s} | {'Slice Runtime (ms)':20s} | {'True 3D Runtime (ms)':22s} | {'Ratio (3D / Slice)':18s}")
    print("-" * 80)

    for flt in filters:
        # 1. Run slice-wise
        cmd_slice = [
            benchmark_bin,
            "--filter", flt,
            "--dim", "3D",
            "--size", "64",
            "--threads", "12",
            "--mode", "slice",
            "--warmup", "2",
            "--runs", "5"
        ]
        res_slice = subprocess.run(cmd_slice, capture_output=True, text=True, check=True)
        parts_slice = res_slice.stdout.strip().split(",")
        slice_ms = float(parts_slice[5]) * 1000.0

        # 2. Run True 3D eval
        cmd_3d = [
            benchmark_bin,
            "--filter", flt,
            "--dim", "3D",
            "--size", "64",
            "--threads", "12",
            "--mode", "3d",
            "--variant", "eval",
            "--warmup", "2",
            "--runs", "5"
        ]
        res_3d = subprocess.run(cmd_3d, capture_output=True, text=True, check=True)
        parts_3d = res_3d.stdout.strip().split(",")
        td_ms = float(parts_3d[5]) * 1000.0

        ratio = td_ms / slice_ms if slice_ms > 0 else 0.0
        print(f"{flt.capitalize():15s} | {slice_ms:17.3f} ms | {td_ms:19.3f} ms | {ratio:15.2f}x")

        results.append({
            "filter": flt,
            "slice_ms": slice_ms,
            "true_3d_ms": td_ms,
            "ratio_3d_over_slice": ratio
        })

    print("=" * 80)

    # Save to CSV
    with open(out_csv, "w") as f:
        f.write("# Pure C++ Slice-wise vs True 3D Benchmark under Identical Threading (12 threads, 64^3, 5 runs after 2 warmup)\n")
        f.write("filter,size,threads,slice_runtime_ms,true_3d_runtime_ms,ratio_3d_over_slice\n")
        for r in results:
            f.write(f"{r['filter']},64,12,{r['slice_ms']:.4f},{r['true_3d_ms']:.4f},{r['ratio_3d_over_slice']:.4f}\n")

    print(f"Saved benchmark results to {out_csv}")

if __name__ == "__main__":
    main()
