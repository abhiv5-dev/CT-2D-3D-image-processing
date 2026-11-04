import os
import pandas as pd
import pytest

def test_sweep_results_csv_structure():
    """Verify that docs/sweep_results.csv exists, loads cleanly with pandas,

    and every row has the exact header field count without missing or corrupted fields.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "docs", "sweep_results.csv")
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"

    # Read with pandas ignoring commented lines
    df = pd.read_csv(csv_path, comment="#")

    expected_columns = [
        "noise_scenario", "noise_level", "filter", "mode", "param_str", "size",
        "runtime_ms", "rmse_mean", "rmse_std", "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std", "roi_noise_mean", "roi_noise_std",
        "edge_width_mean", "edge_width_std", "z_consistency_mean", "z_consistency_std"
    ]

    assert list(df.columns) == expected_columns, f"Columns mismatch: {list(df.columns)} vs {expected_columns}"
    assert df.shape[1] == 19, f"Expected 19 columns, got {df.shape[1]}"
    assert len(df) >= 30, f"Expected at least 30 rows in sweep results, got {len(df)}"

    # Check that no rows have NaN or missing values
    assert df.isna().sum().sum() == 0, f"Found NaN values in CSV: {df.isna().sum()}"

    # Also check raw lines in CSV to guarantee no extra unquoted commas per row
    with open(csv_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    header = lines[0].split(",")
    assert len(header) == 19

def test_real_data_results_csv_structure():
    """Verify that docs/real_data_results.csv exists, loads cleanly with pandas,
    and every row has the exact header field count.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "docs", "real_data_results.csv")
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"

    df = pd.read_csv(csv_path, comment="#")
    expected_columns = [
        "filter", "air_std", "matrix_std", "air_noise_red_pct", "matrix_noise_red_pct",
        "edge_width_linear", "erf_sigma", "edge_width_erf", "method_noise_air", "method_noise_band", "leakage_ratio"
    ]
    assert list(df.columns) == expected_columns, f"Columns mismatch: {list(df.columns)} vs {expected_columns}"
    assert df.shape[1] == 11, f"Expected 11 columns, got {df.shape[1]}"
    assert len(df) == 10, f"Expected 10 rows (1 Original + 9 filters), got {len(df)}"
