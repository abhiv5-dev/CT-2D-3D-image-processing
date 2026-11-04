import csv
import os
import shutil
import subprocess
from pathlib import Path
import pytest
import tifffile
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent

@pytest.fixture
def ctfilter_bin():
    candidates = []
    build_dir_env = os.environ.get("CT_BUILD_DIR")
    if build_dir_env:
        b = Path(build_dir_env)
        candidates.extend([
            b / "ctfilter",
            b / "ctfilter.exe",
            b / "Release" / "ctfilter.exe",
            b / "Debug" / "ctfilter.exe",
            b / "Release" / "ctfilter",
            b / "Debug" / "ctfilter",
        ])

    b_default = repo_root / "build"
    candidates.extend([
        b_default / "ctfilter",
        b_default / "ctfilter.exe",
        b_default / "Release" / "ctfilter.exe",
        b_default / "Debug" / "ctfilter.exe",
        b_default / "Release" / "ctfilter",
        b_default / "Debug" / "ctfilter",
    ])

    for cand in candidates:
        if cand.is_file() and (os.access(cand, os.X_OK) or cand.suffix.lower() == ".exe"):
            return str(cand.resolve())

    which_bin = shutil.which("ctfilter") or shutil.which("ctfilter.exe")
    if which_bin:
        return str(Path(which_bin).resolve())

    pytest.skip("ctfilter binary not found.")

def test_output_naming_and_no_overwrite(ctfilter_bin):
    phantom = repo_root / "data" / "sample" / "phantom_128x128.tif"
    out_dir = repo_root / "ct_output" / "phantom_128x128"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # First run
    res1 = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "gaussian"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res1.returncode == 0, f"Run 1 failed: {res1.stderr}"
    assert "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values." in res1.stdout

    r1 = out_dir / "phantom_128x128_result_float32.tif"
    p1 = out_dir / "phantom_128x128_preview.tif"
    op1 = out_dir / "phantom_128x128_original_preview.tif"
    assert r1.is_file(), "Missing raw float result file"
    assert p1.is_file(), "Missing filtered preview file"
    assert op1.is_file(), "Missing original preview file"

    # Second run: must NOT overwrite existing files, must add numeric suffix _1
    res2 = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "gaussian"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res2.returncode == 0, f"Run 2 failed: {res2.stderr}"

    r2 = out_dir / "phantom_128x128_result_float32_1.tif"
    p2 = out_dir / "phantom_128x128_preview_1.tif"
    op2 = out_dir / "phantom_128x128_original_preview_1.tif"
    assert r2.is_file(), "Missing non-overwritten raw float result file with suffix _1"
    assert p2.is_file(), "Missing non-overwritten filtered preview file with suffix _1"
    assert op2.is_file(), "Missing non-overwritten original preview file with suffix _1"

    # Original files from first run must still exist intact
    assert r1.is_file()
    assert p1.is_file()
    assert op1.is_file()

def test_all_filters_run_on_phantom(ctfilter_bin):
    phantom = repo_root / "data" / "sample" / "phantom_128x128.tif"
    out_dir = repo_root / "ct_output" / "phantom_128x128"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    res = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "all"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res.returncode == 0, f"All filters run failed: {res.stderr}"

    # Verify progress lines printed
    assert "[1/8] Running Gaussian Blur..." in res.stdout
    assert "[8/8] Running Grayscale Morphology (Opening)..." in res.stdout

    # Verify table printed in stdout
    assert "Filter" in res.stdout
    assert "Time (ms)" in res.stdout
    assert "Noise (sigma_hat)" in res.stdout
    assert "Mean Change" in res.stdout
    assert "Caution: The noise estimate measures remaining high-frequency content, not image quality. Smoother is not better: compare the *_preview.tif files for lost detail." in res.stdout

    # Verify summary.csv exists and has rows for all 8 filters with correct columns
    csv_file = out_dir / "summary.csv"
    assert csv_file.is_file()
    with open(csv_file, "r") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 8, f"Expected 8 rows in summary.csv, got {len(reader)}"
        assert "noise_estimate_before" in reader[0]
        assert "noise_estimate_after" in reader[0]
        assert "noise_change_pct" not in reader[0]
        filters = [row["filter"] for row in reader]
        assert "Gaussian Blur" in filters
        assert "Median Filter" in filters
        assert "Bilateral Filter" in filters
        assert "Anisotropic Diffusion" in filters
        assert "Non-Local Means (NLM)" in filters
        assert "Adaptive Wiener Filter" in filters
        assert "Total Variation (TV Chambolle)" in filters
        assert "Grayscale Morphology (Opening)" in filters

    # Check each filter directory contains results
    expected_dirs = ["gaussian", "median", "bilateral", "diffusion", "nlm", "wiener", "tv_chambolle", "morphology"]
    for d in expected_dirs:
        f_dir = out_dir / d
        assert f_dir.is_dir(), f"Expected directory {f_dir} to exist"
        results = list(f_dir.glob("*result_float32*.tif"))
        previews = list(f_dir.glob("*preview*.tif"))
        orig_previews = list(f_dir.glob("*original_preview*.tif"))
        assert len(results) >= 1, f"Missing result_float32 in {f_dir}"
        assert len(previews) >= 1, f"Missing preview in {f_dir}"
        assert len(orig_previews) >= 1, f"Missing original_preview in {f_dir}"

def test_preview_window_equals_percentile_window(ctfilter_bin, tmp_path):
    # Create test image with known 1st and 99th percentiles
    w, h = 64, 64
    np.random.seed(42)
    raw = np.linspace(10.0, 100.0, w * h, dtype=np.float32).reshape((h, w))
    in_tif = tmp_path / "gradient.tif"
    tifffile.imwrite(in_tif, raw)

    out_base = repo_root / "ct_output" / "gradient"
    if out_base.exists():
        shutil.rmtree(out_base)

    res = subprocess.run(
        [ctfilter_bin, "--in", str(in_tif), "--filter", "gaussian", "--sigma", "1.0"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res.returncode == 0, f"Failed: {res.stderr}"

    p_file = out_base / "gradient_preview.tif"
    op_file = out_base / "gradient_original_preview.tif"
    assert p_file.is_file()
    assert op_file.is_file()

    # Load 8-bit preview of original image
    op_data = tifffile.imread(op_file)
    assert op_data.dtype == np.uint8
    # With 1st percentile mapped to 0 and 99th mapped to 255:
    assert np.min(op_data) == 0
    assert np.max(op_data) == 255

    # Cleanup
    if out_base.exists():
        shutil.rmtree(out_base)

def test_console_summary_and_quiet_flag(ctfilter_bin):
    phantom = repo_root / "data" / "sample" / "phantom_128x128.tif"

    # Default run: summary is printed
    res = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "gaussian"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res.returncode == 0
    assert "Mean:" in res.stdout
    assert "Noise:" in res.stdout
    assert "Time:" in res.stdout
    assert "(+" not in res.stdout
    assert "(-" not in res.stdout
    assert "%" not in res.stdout

    # Run with --quiet: summary line and notes are suppressed
    res_quiet = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "gaussian", "--quiet"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res_quiet.returncode == 0
    assert res_quiet.stdout == ""

def test_output_root_option(ctfilter_bin, tmp_path):
    phantom = repo_root / "data" / "sample" / "phantom_128x128.tif"
    custom_root = tmp_path / "custom_outputs"

    res = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "gaussian", "--output-root", str(custom_root)],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res.returncode == 0
    target_dir = custom_root / "phantom_128x128"
    assert (target_dir / "phantom_128x128_result_float32.tif").is_file()
    assert (target_dir / "phantom_128x128_preview.tif").is_file()
    assert (target_dir / "phantom_128x128_original_preview.tif").is_file()

def test_reject_out_with_filter_all(ctfilter_bin):
    phantom = repo_root / "data" / "sample" / "phantom_128x128.tif"

    res = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "all", "--out", "illegal.tif"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res.returncode != 0
    assert "--out is not used with --filter all" in res.stderr

def test_explicit_out_places_previews_next_to_file(ctfilter_bin, tmp_path):
    phantom = repo_root / "data" / "sample" / "phantom_128x128.tif"
    explicit_out = tmp_path / "somewhere" / "custom_gaussian.tif"

    res = subprocess.run(
        [ctfilter_bin, "--in", str(phantom), "--filter", "gaussian", "--out", str(explicit_out)],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert res.returncode == 0
    assert explicit_out.is_file()
    assert (tmp_path / "somewhere" / "custom_gaussian_preview.tif").is_file()
    assert (tmp_path / "somewhere" / "custom_gaussian_original_preview.tif").is_file()
