import pytest
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ct_eval.metrics import edge_width_10_90, fit_edge_erf
from ct_eval.phantom import generate_phantom_2d, generate_phantom_3d, extract_planar_edge_profiles

def test_blurred_step_matches_erf_width():
    # Acceptance: A step blurred by a known Gaussian sigma gives a width within 5 percent of 2.563 times sigma.
    L = 80
    step = np.zeros(L, dtype=np.float64)
    step[L // 2:] = 1.0

    sigmas = [1.5, 2.0, 3.0, 4.0]
    for s in sigmas:
        blurred = gaussian_filter1d(step, sigma=s, mode="nearest")
        expected_w = 2.5631 * s

        w_lin = edge_width_10_90(blurred)
        assert not np.isnan(w_lin), f"Width is NaN for sigma={s}"
        rel_err_lin = abs(w_lin - expected_w) / expected_w
        assert rel_err_lin <= 0.05, f"Linear width {w_lin:.3f} differs from {expected_w:.3f} by {rel_err_lin*100:.1f}%"

        fitted_s, w_erf = fit_edge_erf(blurred)
        assert not np.isnan(w_erf), f"Erf width is NaN for sigma={s}"
        rel_err_erf = abs(w_erf - expected_w) / expected_w
        assert rel_err_erf <= 0.05, f"Erf width {w_erf:.3f} differs from {expected_w:.3f} by {rel_err_erf*100:.1f}%"

def test_unblurred_step_gives_one_voxel_or_less_not_zero():
    # Acceptance: An unblurred step gives about 1 voxel or less, never exactly 0, and never an exception.
    for step_pos in [20, 25, 30]:
        step = np.zeros(60, dtype=np.float64)
        step[step_pos:] = 1.0

        w = edge_width_10_90(step)
        assert not np.isnan(w)
        assert w > 0.0, "Width cannot be exactly 0.0"
        assert w <= 1.0, f"Width should be <= 1.0 voxel, got {w}"

def test_white_noise_robustness_over_20_lines():
    # Acceptance: White noise added to the step does not change the result by more than 10 percent over 20 or more averaged lines.
    L = 60
    step = np.zeros(L, dtype=np.float64)
    step[L // 2:] = 1.0
    sigma = 2.5
    clean_blurred = gaussian_filter1d(step, sigma=sigma, mode="nearest")
    clean_w = edge_width_10_90(clean_blurred)

    rng = np.random.default_rng(12345)
    N_lines = 30
    noisy_profiles = np.tile(clean_blurred, (N_lines, 1)) + rng.normal(0.0, 0.04, (N_lines, L))

    noisy_w = edge_width_10_90(noisy_profiles)
    assert not np.isnan(noisy_w)
    rel_diff = abs(noisy_w - clean_w) / clean_w
    assert rel_diff <= 0.10, f"Noisy width {noisy_w:.3f} changed by {rel_diff*100:.1f}% from clean {clean_w:.3f}"

def test_flat_profile_returns_nan():
    # Acceptance: A profile with no edge returns NaN.
    flat = np.ones(50, dtype=np.float64) * 42.0
    assert np.isnan(edge_width_10_90(flat))
    assert np.isnan(fit_edge_erf(flat)[0])

    # Pure noise with no underlying edge step
    rng = np.random.default_rng(42)
    pure_noise = rng.normal(10.0, 1.0, (25, 50))
    assert np.isnan(edge_width_10_90(pure_noise))
    assert np.isnan(fit_edge_erf(pure_noise)[0])

def test_phantom_planar_edge_profiles():
    # Verify planar edge slab is present and extractable in 2D and 3D
    img2d, m2d = generate_phantom_2d((128, 128))
    p2d = extract_planar_edge_profiles(img2d, m2d)
    assert p2d.ndim == 2
    assert p2d.shape[0] >= 10
    w2d = edge_width_10_90(p2d)
    assert 0.0 < w2d <= 1.0

    vol3d, m3d = generate_phantom_3d((64, 64, 64))
    p3d = extract_planar_edge_profiles(vol3d, m3d)
    assert p3d.ndim == 2
    assert p3d.shape[0] >= 20
    w3d = edge_width_10_90(p3d)
    assert 0.0 < w3d <= 1.0
