import os
import shutil
import subprocess
import pytest
from pathlib import Path

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

    pytest.skip("ctfilter binary not found.")

def test_interactive_ctfilter_piped_answers(ctfilter_bin):
    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_tif = repo_root / "data" / "sample" / "phantom_128x128.tif"
    assert sample_tif.is_file(), f"Sample file not found at {sample_tif}"

    output_dir = repo_root / "ct_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Scripted answers:
    # 1. Path to input TIFF
    # 2. Filter choice: 1 (Gaussian blur)
    # 3. Gaussian sigma: empty (accept default 1.0)
    # 4. Filter radius: empty (accept default 0/auto)
    scripted_input = f"{sample_tif}\n1\n\n\n"

    res = subprocess.run(
        [ctfilter_bin],
        input=scripted_input,
        text=True,
        capture_output=True,
        cwd=str(repo_root),
    )

    assert res.returncode == 0, f"ctfilter interactive failed: {res.stderr}"
    assert "Mean:" in res.stdout
    assert "Noise:" in res.stdout
    assert "Time:" in res.stdout
    assert "Raw result:" in res.stdout
    assert "preview" in res.stdout

    # Verify files created in ct_output
    assert output_dir.is_dir()
    result_tifs = list(output_dir.rglob("*result_float32*.tif"))
    preview_tifs = list(output_dir.rglob("*preview*.tif"))
    assert len(result_tifs) >= 1
    assert len(preview_tifs) >= 1

    # Cleanup
    shutil.rmtree(output_dir)

def test_interactive_ctfilter_bad_path_retry_quoted(ctfilter_bin):
    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_tif = repo_root / "data" / "sample" / "phantom_128x128.tif"
    assert sample_tif.is_file(), f"Sample file not found at {sample_tif}"

    output_dir = repo_root / "ct_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Scripted answers:
    # 1. Bad path (nonexistent)
    # 2. Quoted valid path
    # 3. Filter choice: 1 (Gaussian blur)
    # 4. Use calibrated parameters? Enter (default Y)
    # 5. Try another filter? Enter (default N)
    bad_path = "nonexistent_bad_path_12345.tif"
    quoted_path = f'"{sample_tif}"'
    scripted_input = f"{bad_path}\n{quoted_path}\n1\n\n\n"

    res = subprocess.run(
        [ctfilter_bin],
        input=scripted_input,
        text=True,
        capture_output=True,
        cwd=str(repo_root),
    )

    assert res.returncode == 0, f"ctfilter interactive retry failed: {res.stderr}"
    # Verify it explained why the first path failed
    assert "Invalid path:" in res.stderr or "Path does not exist" in res.stderr
    assert "Raw result:" in res.stdout
    assert "preview" in res.stdout

    # Verify files created in ct_output
    assert output_dir.is_dir()
    result_tifs = list(output_dir.rglob("*result_float32*.tif"))
    preview_tifs = list(output_dir.rglob("*preview*.tif"))
    assert len(result_tifs) >= 1
    assert len(preview_tifs) >= 1

    # Cleanup
    shutil.rmtree(output_dir)
