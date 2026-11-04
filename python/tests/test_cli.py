import os
import shutil
import subprocess
import numpy as np
import tifffile
import pytest
from pathlib import Path

@pytest.fixture
def ctfilter_bin():
    """Finds the ctfilter binary via CT_BUILD_DIR environment variable or repository build dirs."""
    candidates = []

    # 1. Via CT_BUILD_DIR environment variable
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

    # 2. Via default repo build directory
    repo_root = Path(__file__).resolve().parent.parent.parent
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

    pytest.skip("ctfilter binary not found. Set CT_BUILD_DIR environment variable.")

def test_ctfilter_folder_mode_slice_and_3d(tmp_path, ctfilter_bin):
    in_dir = tmp_path / "stack_in"
    out_dir_slice = tmp_path / "stack_out_slice"
    out_dir_3d = tmp_path / "stack_out_3d"
    in_dir.mkdir()

    # Generate 5 slices with numbers like slice_2.tif, slice_10.tif to test natural numeric sort
    filenames = ["slice_1.tif", "slice_2.tif", "slice_9.tif", "slice_10.tif", "slice_11.tif"]
    for i, fname in enumerate(filenames):
        data = np.full((16, 16), fill_value=float(i + 1), dtype=np.float32)
        tifffile.imwrite(in_dir / fname, data)

    # 1. Test slice-wise mode
    cmd_slice = [
        ctfilter_bin,
        "--in-dir", str(in_dir),
        "--out-dir", str(out_dir_slice),
        "--filter", "gaussian",
        "--mode", "slice",
        "--sigma", "1.0"
    ]
    res_slice = subprocess.run(cmd_slice, capture_output=True, text=True)
    assert res_slice.returncode == 0, f"CLI slice mode failed: {res_slice.stderr}"
    assert "Discovered 5 slices in sorted numeric order" in res_slice.stdout

    # Check output filenames and content
    for fname in filenames:
        out_file = out_dir_slice / fname
        assert out_file.exists(), f"Output file missing: {out_file}"
        out_data = tifffile.imread(out_file)
        assert out_data.shape == (16, 16)
        # Uniform slice smoothed by gaussian should preserve mean
        assert np.isclose(np.mean(out_data), float(filenames.index(fname) + 1), atol=1e-3)

    # 2. Test 3d mode
    cmd_3d = [
        ctfilter_bin,
        "--in-dir", str(in_dir),
        "--out-dir", str(out_dir_3d),
        "--filter", "gaussian",
        "--mode", "3d",
        "--sigma", "1.0"
    ]
    res_3d = subprocess.run(cmd_3d, capture_output=True, text=True)
    assert res_3d.returncode == 0, f"CLI 3d mode failed: {res_3d.stderr}"
    assert "Assembling 5 slices into 3D volume" in res_3d.stdout

    for fname in filenames:
        out_file = out_dir_3d / fname
        assert out_file.exists(), f"Output file missing: {out_file}"
        out_data = tifffile.imread(out_file)
        assert out_data.shape == (16, 16)

    # 3. Test size mismatch error handling
    bad_file = in_dir / "slice_12.tif"
    tifffile.imwrite(bad_file, np.zeros((20, 16), dtype=np.float32))

    res_fail = subprocess.run(cmd_slice, capture_output=True, text=True)
    assert res_fail.returncode != 0
    assert "Size mismatch in directory slices" in res_fail.stderr
    assert "expected (16x16)" in res_fail.stderr

    # Test bad mode
    cmd_bad_mode = [
        ctfilter_bin,
        "--in-dir", str(in_dir),
        "--out-dir", str(out_dir_slice),
        "--filter", "gaussian",
        "--mode", "invalid_mode"
    ]
    res_bad = subprocess.run(cmd_bad_mode, capture_output=True, text=True)
    assert res_bad.returncode != 0
    assert "Unknown --mode" in res_bad.stderr

def test_ctfilter_help_and_unknown_options(ctfilter_bin):
    """Verify that --help and -h print usage to stdout and exit 0, while unknown options exit non-zero."""
    # 1. --help exits 0 and prints 'Usage' to stdout
    res_help = subprocess.run([ctfilter_bin, "--help"], capture_output=True, text=True)
    assert res_help.returncode == 0
    assert "Usage" in res_help.stdout
    assert "Options:" in res_help.stdout

    # 2. -h exits 0 and prints 'Usage' to stdout
    res_h = subprocess.run([ctfilter_bin, "-h"], capture_output=True, text=True)
    assert res_h.returncode == 0
    assert "Usage" in res_h.stdout

    # 3. Unknown option exits non-zero (exit code 2) and prints usage/error to stderr
    res_unknown = subprocess.run([ctfilter_bin, "--unknown-option"], capture_output=True, text=True)
    assert res_unknown.returncode != 0
    assert res_unknown.returncode == 2
    assert "Usage" in res_unknown.stderr

    # 4. Missing required arguments exit non-zero (exit code 2) and print usage to stderr
    res_missing = subprocess.run([ctfilter_bin, "--in", "phantom.tif"], capture_output=True, text=True)
    assert res_missing.returncode != 0
    assert res_missing.returncode == 2
    assert "Usage" in res_missing.stderr

