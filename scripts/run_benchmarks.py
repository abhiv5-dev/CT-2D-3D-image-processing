#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def get_system_info(project_root):
    # 1. CPU Model and Cores
    cores = os.cpu_count() or 1
    cpu_model = "Unknown CPU"
    if sys.platform == "darwin":
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
            cpu_model = res.stdout.strip()
        except Exception:
            cpu_model = platform.processor() or "Apple Silicon"
    elif sys.platform == "win32":
        try:
            res = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True)
            lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip() and "Name" not in line]
            if lines:
                cpu_model = lines[0]
            else:
                cpu_model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Windows CPU")
        except Exception:
            cpu_model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Windows CPU")
    else:
        # Linux
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            cpu_model = platform.processor() or "x86_64"

    # 2. Compiler and Flags from CMakeCache.txt
    build_dir = os.environ.get("CT_BUILD_DIR", os.path.join(project_root, "build"))
    cache_path = os.path.join(build_dir, "CMakeCache.txt")
    compiler = "Unknown C++ Compiler"
    flags = "-O3 -DNDEBUG"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                if line.startswith("CMAKE_CXX_COMPILER:"):
                    compiler = line.split("=", 1)[1].strip()
                elif line.startswith("CMAKE_CXX_FLAGS_RELEASE:"):
                    flags = line.split("=", 1)[1].strip()

    return {
        "cpu_model": cpu_model,
        "cores": cores,
        "compiler": compiler,
        "flags": flags
    }

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.environ.get("CT_BUILD_DIR", os.path.join(project_root, "build"))
    candidate_bins = [
        os.path.join(build_dir, "benchmark"),
        os.path.join(build_dir, "benchmark.exe"),
        os.path.join(build_dir, "Release", "benchmark.exe"),
        os.path.join(build_dir, "Debug", "benchmark.exe"),
    ]
    benchmark_bin = None
    for cand in candidate_bins:
        if os.path.exists(cand):
            benchmark_bin = cand
            break

    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    if not benchmark_bin:
        print(f"Error: benchmark binary not found in {build_dir}. Please build the project first.")
        sys.exit(1)

    sys_info = get_system_info(project_root)
    print("=" * 70)
    print("TASK 8: Running Honest Benchmarks")
    print(f"CPU: {sys_info['cpu_model']} ({sys_info['cores']} cores)")
    print(f"Compiler: {sys_info['compiler']}, Flags: {sys_info['flags']}")
    print("=" * 70)

    max_threads = sys_info["cores"]
    thread_counts = sorted(list(set([1, 2, 4, 8, max_threads])))

    # Define benchmark matrix
    # Format: (filter, dim, size, threads, variant, warmup, runs, timeout_sec)
    jobs = []

    # 1. 2D Benchmarks: sizes 512, 1024, 2048
    sizes_2d = [512, 1024, 2048]
    for sz in sizes_2d:
        # Gaussian naive (single thread baseline)
        jobs.append(("gaussian", "2D", sz, 1, "naive_reference", 1, 5, 30))
        # Gaussian optimized (scaling)
        for th in thread_counts:
            jobs.append(("gaussian", "2D", sz, th, "optimized", 1, 5, 30))

        # Median baseline (1 thread)
        jobs.append(("median", "2D", sz, 1, "baseline", 1, 5, 30))
        # Median fast_parallel
        for th in thread_counts:
            jobs.append(("median", "2D", sz, th, "fast_parallel", 1, 5, 30))

        # Bilateral baseline (1 thread, exact)
        jobs.append(("bilateral", "2D", sz, 1, "baseline", 1, 5, 30))
        # Bilateral fast_parallel (LUT)
        for th in thread_counts:
            jobs.append(("bilateral", "2D", sz, th, "fast_parallel", 1, 5, 30))

        # Diffusion optimized
        for th in thread_counts:
            jobs.append(("diffusion", "2D", sz, th, "optimized", 1, 5, 30))

    # 1b. Additional 2D filters (Item 2: Wiener, TV, NLM, Morphology at 512 and 1024)
    new_filters = ["wiener", "tv", "morphology"]
    for sz in [512, 1024]:
        for flt in new_filters:
            for th in thread_counts:
                jobs.append((flt, "2D", sz, th, "optimized", 1, 5, 30))
        for th in thread_counts:
            runs_nlm = 3 if sz == 512 else 2
            jobs.append(("nlm", "2D", sz, th, "optimized", 1, runs_nlm, 30))

    # 2. 3D Benchmarks: sizes 64, 128, 256, and 512 for Gaussian
    sizes_3d = [64, 128, 256]
    for sz in sizes_3d:
        # Gaussian naive
        jobs.append(("gaussian", "3D", sz, 1, "naive_reference", 1, 5, 30))
        # Gaussian optimized
        for th in thread_counts:
            jobs.append(("gaussian", "3D", sz, th, "optimized", 1, 5, 30))

        # Median baseline (1 thread)
        # Cap 256^3 single thread if expensive
        if sz <= 128:
            jobs.append(("median", "3D", sz, 1, "baseline", 1, 5, 30))
        else:
            jobs.append(("median", "3D", sz, 1, "baseline", 1, 1, 20))  # time-capped

        for th in thread_counts:
            jobs.append(("median", "3D", sz, th, "fast_parallel", 1, 5, 30))

        # Bilateral baseline (1 thread, exact)
        if sz <= 128:
            jobs.append(("bilateral", "3D", sz, 1, "baseline", 1, 5, 30))
        else:
            # 256^3 single thread exact would take minutes -> mark skipped
            jobs.append(("bilateral", "3D", sz, 1, "baseline", 0, 0, -1))

        for th in thread_counts:
            if sz == 256 and th == 1:
                # 256^3 single thread LUT is also very slow -> cap
                jobs.append(("bilateral", "3D", sz, th, "fast_parallel", 0, 0, -1))
            else:
                jobs.append(("bilateral", "3D", sz, th, "fast_parallel", 1, 5, 35))

        # Diffusion optimized
        for th in thread_counts:
            jobs.append(("diffusion", "3D", sz, th, "optimized", 1, 5, 30))

    # 2b. Additional 3D filters (Item 2: Wiener, TV, Morphology at 64, 128, 256; NLM at 64, 128)
    for sz in [64, 128, 256]:
        for flt in new_filters:
            r = 5 if sz <= 128 else 2
            for th in thread_counts:
                jobs.append((flt, "3D", sz, th, "optimized", 1, r, 30))

    # NLM 3D (time-capped: 64 and 128 measured, 256 skipped)
    for th in thread_counts:
        jobs.append(("nlm", "3D", 64, th, "optimized", 1, 2, 30))
    for th in thread_counts:
        # runs=1 for 128^3
        jobs.append(("nlm", "3D", 128, th, "optimized", 0, 1, 45))
    for th in thread_counts:
        jobs.append(("nlm", "3D", 256, th, "optimized", 0, 0, -1))

    # 3. 512^3 ONLY for Gaussian
    jobs.append(("gaussian", "3D", 512, 1, "optimized", 1, 3, 60))
    jobs.append(("gaussian", "3D", 512, 4, "optimized", 1, 3, 60))
    jobs.append(("gaussian", "3D", 512, max_threads, "optimized", 1, 3, 60))

    csv_rows = []
    print(f"Total benchmark configurations: {len(jobs)}")

    for idx, (flt, dim, sz, th, var, warmup, runs, timeout) in enumerate(jobs):
        if timeout == -1:
            print(f"[{idx+1:03d}/{len(jobs):03d}] {flt} {dim} {sz} (th={th}, {var}): SKIPPED (exceeds time limit)")
            csv_rows.append(f"{flt},{dim},{sz},{th},{var},skipped,skipped,skipped,skipped,skipped,skipped\n")
            continue

        cmd = [
            benchmark_bin,
            "--filter", flt,
            "--dim", dim,
            "--size", str(sz),
            "--threads", str(th),
            "--variant", var,
            "--warmup", str(warmup),
            "--runs", str(runs)
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode == 0:
                line = res.stdout.strip()
                csv_rows.append(line + "\n")
                parts = line.split(",")
                sec = float(parts[5])
                mv_s = float(parts[6])
                rss = float(parts[7])
                print(f"[{idx+1:03d}/{len(jobs):03d}] {flt:9s} {dim} {sz:4d} (th={th:2d}, {var:15s}): {sec:8.4f}s ({mv_s:7.1f} MV/s) [RSS={rss:6.1f} MB]")
            else:
                print(f"[{idx+1:03d}/{len(jobs):03d}] {flt} {dim} {sz} (th={th}, {var}): FAILED: {res.stderr}")
                csv_rows.append(f"{flt},{dim},{sz},{th},{var},failed,failed,failed,failed,failed,failed\n")
        except subprocess.TimeoutExpired:
            print(f"[{idx+1:03d}/{len(jobs):03d}] {flt} {dim} {sz} (th={th}, {var}): SKIPPED (timeout > {timeout}s)")
            csv_rows.append(f"{flt},{dim},{sz},{th},{var},skipped,skipped,skipped,skipped,skipped,skipped\n")

    # Write docs/benchmark_results.csv
    csv_path = os.path.join(docs_dir, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        f.write(f"# Compiler: {sys_info['compiler']}, Flags: {sys_info['flags']}\n")
        f.write(f"# CPU: {sys_info['cpu_model']}, Cores: {sys_info['cores']}\n")
        f.write("filter,dim,size,threads,variant,seconds,megavoxels_per_second,peak_rss_mb,min_seconds,max_seconds,std_seconds\n")
        for r in csv_rows:
            f.write(r)
    print(f"\nSaved benchmark results CSV to: {csv_path}")

    # Generate docs/benchmark_scaling.png
    # Plot speedup vs threads for 2D 1024 and 3D 128
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    parsed = []
    for r in csv_rows:
        parts = r.strip().split(",")
        if len(parts) >= 8 and parts[5] not in ["skipped", "failed"]:
            parsed.append({
                "filter": parts[0],
                "dim": parts[1],
                "size": int(parts[2]),
                "threads": int(parts[3]),
                "variant": parts[4],
                "seconds": float(parts[5]),
                "mv_s": float(parts[6]),
                "rss": float(parts[7])
            })

    colors = {
        "gaussian": "#e74c3c", "median": "#e67e22", "bilateral": "#2980b9", "diffusion": "#27ae60",
        "wiener": "#8e44ad", "tv": "#16a085", "morphology": "#d35400", "nlm": "#2c3e50"
    }
    all_filters = ["gaussian", "median", "bilateral", "diffusion", "wiener", "tv", "morphology", "nlm"]

    # Subplot 1: 2D 1024x1024 Scaling
    ax1.plot(thread_counts, thread_counts, "k--", alpha=0.5, label="Ideal Linear")
    for flt in all_filters:
        # Find 1-thread baseline
        v_name = "optimized" if flt != "median" and flt != "bilateral" else "fast_parallel"
        sub = [p for p in parsed if p["filter"] == flt and p["dim"] == "2D" and p["size"] == 1024 and p["variant"] == v_name]
        t1_item = next((p for p in sub if p["threads"] == 1), None)
        if t1_item:
            t1 = t1_item["seconds"]
            th_vals = [p["threads"] for p in sub]
            sp_vals = [t1 / p["seconds"] for p in sub]
            ax1.plot(th_vals, sp_vals, marker="o", color=colors[flt], label=f"{flt.capitalize()}")

    ax1.set_xlabel("Number of Threads", fontsize=11)
    ax1.set_ylabel("Speedup over 1 Thread", fontsize=11)
    ax1.set_title("2D Multithreading Scaling (1024x1024)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper left")

    # Subplot 2: 3D 128^3 Scaling
    ax2.plot(thread_counts, thread_counts, "k--", alpha=0.5, label="Ideal Linear")
    for flt in all_filters:
        v_name = "optimized" if flt != "median" and flt != "bilateral" else "fast_parallel"
        sub = [p for p in parsed if p["filter"] == flt and p["dim"] == "3D" and p["size"] == 128 and p["variant"] == v_name]
        t1_item = next((p for p in sub if p["threads"] == 1), None)
        if t1_item:
            t1 = t1_item["seconds"]
            th_vals = [p["threads"] for p in sub]
            sp_vals = [t1 / p["seconds"] for p in sub]
            ax2.plot(th_vals, sp_vals, marker="o", color=colors[flt], label=f"{flt.capitalize()}")

    ax2.set_xlabel("Number of Threads", fontsize=11)
    ax2.set_ylabel("Speedup over 1 Thread", fontsize=11)
    ax2.set_title("3D Multithreading Scaling (128^3)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    scaling_path = os.path.join(docs_dir, "benchmark_scaling.png")
    fig.savefig(scaling_path, dpi=200)
    plt.close(fig)
    print(f"Saved benchmark scaling plot to: {scaling_path}")

    # Generate and print table of Multithreading Scaling
    print("\n" + "=" * 90)
    print("MEASURED MULTITHREADING PERFORMANCE & SCALING (1024x1024 and 128^3)")
    print("=" * 90)
    print(f"{'Filter':12s} | {'Dim':3s} | {'Size':5s} | {'1-th (s)':10s} | {'Best-th (s)':12s} | {'Best Th':7s} | {'Speedup':8s} | {'12-th (s)':10s} | {'Peak RSS (MB)':13s}")
    print("-" * 90)

    for dim, sz in [("2D", 1024), ("3D", 128)]:
        for flt in all_filters:
            v_name = "optimized" if flt not in ["median", "bilateral"] else "fast_parallel"
            sub = [p for p in parsed if p["filter"] == flt and p["dim"] == dim and p["size"] == sz and p["variant"] == v_name]
            t1_item = next((p for p in sub if p["threads"] == 1), None)
            t12_item = next((p for p in sub if p["threads"] == max_threads), None)
            if sub and t1_item:
                best_item = min(sub, key=lambda x: x["seconds"])
                sp = t1_item["seconds"] / best_item["seconds"]
                t12_str = f"{t12_item['seconds']:10.4f}" if t12_item else "N/A"
                print(f"{flt.capitalize():12s} | {dim:3s} | {sz:5d} | {t1_item['seconds']:10.4f} | {best_item['seconds']:12.4f} | {best_item['threads']:7d} | {sp:7.2f}x | {t12_str} | {best_item['rss']:13.1f}")
        print("-" * 90)

if __name__ == "__main__":
    main()
