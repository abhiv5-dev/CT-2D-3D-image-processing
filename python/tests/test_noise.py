import numpy as np
import pytest
from pathlib import Path
from scipy.ndimage import correlate

repo_root = Path(__file__).resolve().parent.parent.parent

IMMERKAER_KERNEL = np.array([
    [ 1, -2,  1],
    [-2,  4, -2],
    [ 1, -2,  1]
], dtype=np.float32)

def python_immerkaer(img):
    c = correlate(img, IMMERKAER_KERNEL, mode="reflect")
    return float(np.median(np.abs(c)) / (0.67448975 * 6.0))

def test_noise_estimator_sample_phantom():
    import ct
    phantom_path = repo_root / "data" / "sample" / "phantom_128x128.tif"
    phantom = ct.load_tiff(str(phantom_path))
    py_noise = python_immerkaer(phantom)
    cpp_noise = ct.estimate_noise_sigma(phantom)
    assert abs(cpp_noise - py_noise) < 1e-6
    assert cpp_noise == 0.0

def test_noise_estimator_synthetic_noisy_image():
    import ct
    np.random.seed(42)
    clean = np.outer(np.sin(np.linspace(0, np.pi, 128)), np.cos(np.linspace(0, np.pi, 128))).astype(np.float32)
    noise = np.random.normal(0, 0.05, clean.shape).astype(np.float32)
    noisy = clean + noise

    py_noise = python_immerkaer(noisy)
    cpp_noise = ct.estimate_noise_sigma(noisy)

    # Must match Python estimate within 1e-4 relative
    rel_diff = abs(cpp_noise - py_noise) / py_noise
    assert rel_diff < 1e-4, f"Mismatch: cpp={cpp_noise}, py={py_noise}, rel_diff={rel_diff}"

def test_noise_estimator_recovers_known_sigma():
    import ct
    sigmas = [0.01, 0.02, 0.04, 0.08]
    for sig in sigmas:
        np.random.seed(12345)
        noisy = np.random.normal(0, sig, (256, 256)).astype(np.float32)
        cpp_noise = ct.estimate_noise_sigma(noisy)
        rel_err = abs(cpp_noise - sig) / sig
        assert rel_err < 0.10, f"Failed to recover sigma={sig} within 10%: got {cpp_noise} (rel_err={rel_err})"
