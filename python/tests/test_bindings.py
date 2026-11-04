import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import wiener as scipy_wiener
from skimage.restoration import denoise_tv_chambolle

ct = pytest.importorskip("ct")

# ==============================================================================
# Helper Reference Implementations (brute-force NumPy)
# ==============================================================================

def reflect_idx(idx, size):
    while idx < 0 or idx >= size:
        if idx < 0:
            idx = -idx - 1
        elif idx >= size:
            idx = 2 * size - idx - 1
    return idx

def numpy_bilateral_2d(img, sigma_spatial=2.0, sigma_range=0.1, radius=3):
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    two_ss = 2.0 * sigma_spatial * sigma_spatial
    two_sr = 2.0 * sigma_range * sigma_range

    dys, dxs = np.mgrid[-radius:radius+1, -radius:radius+1]
    sp_weights = np.exp(-(dys**2 + dxs**2) / two_ss)

    for y in range(H):
        for x in range(W):
            c_val = img[y, x]
            sum_w = 0.0
            sum_val = 0.0
            for i, dy in enumerate(range(-radius, radius + 1)):
                sy = reflect_idx(y + dy, H)
                for j, dx in enumerate(range(-radius, radius + 1)):
                    sx = reflect_idx(x + dx, W)
                    val = img[sy, sx]
                    diff = val - c_val
                    rw = np.exp(-(diff * diff) / two_sr)
                    w = sp_weights[i, j] * rw
                    sum_w += w
                    sum_val += w * val
            out[y, x] = sum_val / sum_w
    return out

def numpy_bilateral_3d(vol, sigma_spatial=2.0, sigma_range=0.1, radius=2):
    D, H, W = vol.shape
    out = np.zeros_like(vol, dtype=np.float32)
    two_ss = 2.0 * sigma_spatial * sigma_spatial
    two_sr = 2.0 * sigma_range * sigma_range

    dzs, dys, dxs = np.mgrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    sp_weights = np.exp(-(dzs**2 + dys**2 + dxs**2) / two_ss)

    for z in range(D):
        for y in range(H):
            for x in range(W):
                c_val = vol[z, y, x]
                sum_w = 0.0
                sum_val = 0.0
                for iz, dz in enumerate(range(-radius, radius + 1)):
                    sz = reflect_idx(z + dz, D)
                    for iy, dy in enumerate(range(-radius, radius + 1)):
                        sy = reflect_idx(y + dy, H)
                        for ix, dx in enumerate(range(-radius, radius + 1)):
                            sx = reflect_idx(x + dx, W)
                            val = vol[sz, sy, sx]
                            diff = val - c_val
                            rw = np.exp(-(diff * diff) / two_sr)
                            w = sp_weights[iz, iy, ix] * rw
                            sum_w += w
                            sum_val += w * val
                out[z, y, x] = sum_val / sum_w
    return out

def numpy_diffusion_2d(img, kappa=0.05, iterations=5, dt=0.20):
    H, W = img.shape
    cur = img.astype(np.float64).copy()
    kappa_sq = kappa * kappa

    for _ in range(iterations):
        nxt = np.zeros_like(cur)
        for y in range(H):
            for x in range(W):
                center = cur[y, x]
                dN = (cur[y - 1, x] - center) if y > 0 else 0.0
                dS = (cur[y + 1, x] - center) if y < H - 1 else 0.0
                dW = (cur[y, x - 1] - center) if x > 0 else 0.0
                dE = (cur[y, x + 1] - center) if x < W - 1 else 0.0

                cN = np.exp(-(dN * dN) / kappa_sq)
                cS = np.exp(-(dS * dS) / kappa_sq)
                cW = np.exp(-(dW * dW) / kappa_sq)
                cE = np.exp(-(dE * dE) / kappa_sq)

                flux = cN * dN + cS * dS + cW * dW + cE * dE
                nxt[y, x] = center + dt * flux
        cur = nxt
    return cur.astype(np.float32)

def numpy_diffusion_3d(vol, kappa=0.05, iterations=5, dt=0.15):
    D, H, W = vol.shape
    cur = vol.astype(np.float64).copy()
    kappa_sq = kappa * kappa

    for _ in range(iterations):
        nxt = np.zeros_like(cur)
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    center = cur[z, y, x]
                    dU = (cur[z - 1, y, x] - center) if z > 0 else 0.0
                    dD = (cur[z + 1, y, x] - center) if z < D - 1 else 0.0
                    dN = (cur[z, y - 1, x] - center) if y > 0 else 0.0
                    dS = (cur[z, y + 1, x] - center) if y < H - 1 else 0.0
                    dW = (cur[z, y, x - 1] - center) if x > 0 else 0.0
                    dE = (cur[z, y, x + 1] - center) if x < W - 1 else 0.0

                    cU = np.exp(-(dU * dU) / kappa_sq)
                    cD = np.exp(-(dD * dD) / kappa_sq)
                    cN = np.exp(-(dN * dN) / kappa_sq)
                    cS = np.exp(-(dS * dS) / kappa_sq)
                    cW = np.exp(-(dW * dW) / kappa_sq)
                    cE = np.exp(-(dE * dE) / kappa_sq)

                    flux = cU * dU + cD * dD + cN * dN + cS * dS + cW * dW + cE * dE
                    nxt[z, y, x] = center + dt * flux
        cur = nxt
    return cur.astype(np.float32)

def numpy_wiener_2d(img, radius=1, noise_var=None):
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    lmean = np.zeros_like(img, dtype=np.float64)
    lvar = np.zeros_like(img, dtype=np.float64)

    for y in range(H):
        for x in range(W):
            vals = []
            for dy in range(-radius, radius + 1):
                sy = reflect_idx(y + dy, H)
                for dx in range(-radius, radius + 1):
                    sx = reflect_idx(x + dx, W)
                    vals.append(img[sy, sx])
            arr = np.array(vals, dtype=np.float64)
            m = np.mean(arr)
            v = np.mean(arr ** 2) - m ** 2
            lmean[y, x] = m
            lvar[y, x] = max(0.0, v)

    if noise_var is None or noise_var < 0:
        noise = float(np.mean(lvar))
    else:
        noise = float(noise_var)

    for y in range(H):
        for x in range(W):
            v = lvar[y, x]
            m = lmean[y, x]
            val = img[y, x]
            if v <= noise or v <= 0.0:
                out[y, x] = m
            else:
                out[y, x] = m + ((v - noise) / v) * (val - m)
    return out

def numpy_wiener_3d(vol, radius=1, noise_var=None):
    D, H, W = vol.shape
    out = np.zeros_like(vol, dtype=np.float32)
    lmean = np.zeros_like(vol, dtype=np.float64)
    lvar = np.zeros_like(vol, dtype=np.float64)

    for z in range(D):
        for y in range(H):
            for x in range(W):
                vals = []
                for dz in range(-radius, radius + 1):
                    sz = reflect_idx(z + dz, D)
                    for dy in range(-radius, radius + 1):
                        sy = reflect_idx(y + dy, H)
                        for dx in range(-radius, radius + 1):
                            sx = reflect_idx(x + dx, W)
                            vals.append(vol[sz, sy, sx])
                arr = np.array(vals, dtype=np.float64)
                m = np.mean(arr)
                v = np.mean(arr ** 2) - m ** 2
                lmean[z, y, x] = m
                lvar[z, y, x] = max(0.0, v)

    if noise_var is None or noise_var < 0:
        noise = float(np.mean(lvar))
    else:
        noise = float(noise_var)

    for z in range(D):
        for y in range(H):
            for x in range(W):
                v = lvar[z, y, x]
                m = lmean[z, y, x]
                val = vol[z, y, x]
                if v <= noise or v <= 0.0:
                    out[z, y, x] = m
                else:
                    out[z, y, x] = m + ((v - noise) / v) * (val - m)
    return out

# ==============================================================================
# Pytest Test Cases for Task 9
# ==============================================================================

def test_gaussian_vs_scipy():
    rng = np.random.default_rng(42)
    # 2D test
    arr2d = rng.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
    sigma = 1.2
    radius = 4
    truncate = radius / sigma
    expected2d = ndimage.gaussian_filter(arr2d, sigma=sigma, truncate=truncate, mode='reflect')
    actual2d = ct.gaussian_filter_2d(arr2d, sigma=sigma, radius=radius)
    np.testing.assert_allclose(actual2d, expected2d, atol=1e-5, rtol=1e-5)

    # 3D test
    arr3d = rng.uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    expected3d = ndimage.gaussian_filter(arr3d, sigma=sigma, truncate=truncate, mode='reflect')
    actual3d = ct.gaussian_filter_3d(arr3d, sigma=sigma, radius=radius)
    np.testing.assert_allclose(actual3d, expected3d, atol=1e-5, rtol=1e-5)

def test_median_vs_scipy():
    rng = np.random.default_rng(123)
    # 2D
    arr2d = rng.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
    r = 1
    expected2d = ndimage.median_filter(arr2d, size=2*r + 1, mode='reflect')
    actual2d = ct.median_filter_2d(arr2d, radius=r)
    np.testing.assert_array_equal(actual2d, expected2d)

    # 3D
    arr3d = rng.uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    expected3d = ndimage.median_filter(arr3d, size=2*r + 1, mode='reflect')
    actual3d = ct.median_filter_3d(arr3d, radius=r)
    np.testing.assert_array_equal(actual3d, expected3d)

def test_bilateral_vs_numpy_reference():
    rng = np.random.default_rng(456)
    # 2D 16x16
    arr2d = rng.uniform(0.0, 1.0, (16, 16)).astype(np.float32)
    expected2d = numpy_bilateral_2d(arr2d, sigma_spatial=1.5, sigma_range=0.1, radius=2)
    actual2d = ct.bilateral_filter_2d(arr2d, sigma_spatial=1.5, sigma_range=0.1, radius=2, exact=True)
    np.testing.assert_allclose(actual2d, expected2d, atol=1e-5, rtol=1e-5)

    # 3D 8x8x8
    arr3d = rng.uniform(0.0, 1.0, (8, 8, 8)).astype(np.float32)
    expected3d = numpy_bilateral_3d(arr3d, sigma_spatial=1.5, sigma_range=0.1, radius=2)
    actual3d = ct.bilateral_filter_3d(arr3d, sigma_spatial=1.5, sigma_range=0.1, radius=2, exact=True)
    np.testing.assert_allclose(actual3d, expected3d, atol=1e-5, rtol=1e-5)

def test_wiener_vs_numpy_reference():
    rng = np.random.default_rng(321)
    # 2D 16x16 with default noise (estimated from local variance)
    arr2d = rng.uniform(0.0, 1.0, (16, 16)).astype(np.float32)
    expected2d = numpy_wiener_2d(arr2d, radius=1)
    actual2d = ct.wiener_filter_2d(arr2d, radius=1)
    np.testing.assert_allclose(actual2d, expected2d, atol=1e-5, rtol=1e-5)

    # 2D with fixed noise
    expected2d_fn = numpy_wiener_2d(arr2d, radius=2, noise_var=0.02)
    actual2d_fn = ct.wiener_filter_2d(arr2d, radius=2, noise_variance=0.02)
    np.testing.assert_allclose(actual2d_fn, expected2d_fn, atol=1e-5, rtol=1e-5)

    # 3D 8x8x8
    arr3d = rng.uniform(0.0, 1.0, (8, 8, 8)).astype(np.float32)
    expected3d = numpy_wiener_3d(arr3d, radius=1)
    actual3d = ct.wiener_filter_3d(arr3d, radius=1)
    np.testing.assert_allclose(actual3d, expected3d, atol=1e-5, rtol=1e-5)

def test_wiener_vs_scipy_interior():
    rng = np.random.default_rng(654)
    # Large enough image to test interior well away from borders
    arr2d = rng.uniform(10.0, 50.0, (32, 32)).astype(np.float32)
    radius = 1
    fixed_noise = 2.5
    ct_out = ct.wiener_filter_2d(arr2d, radius=radius, noise_variance=fixed_noise)
    scipy_out = scipy_wiener(arr2d, mysize=2 * radius + 1, noise=fixed_noise)

    # Check interior pixels well inside the boundary (where reflect and zero-pad don't affect)
    interior = slice(radius + 1, -(radius + 1))
    np.testing.assert_allclose(ct_out[interior, interior], scipy_out[interior, interior], atol=1e-5, rtol=1e-5)

def test_morphology_vs_scipy():
    rng = np.random.default_rng(101)
    arr2d = rng.uniform(0.0, 100.0, (25, 25)).astype(np.float32)
    radius = 1

    # Box footprint
    expected_erode = ndimage.grey_erosion(arr2d, size=2 * radius + 1, mode='reflect')
    expected_dilate = ndimage.grey_dilation(arr2d, size=2 * radius + 1, mode='reflect')
    expected_open = ndimage.grey_opening(arr2d, size=2 * radius + 1, mode='reflect')
    expected_close = ndimage.grey_closing(arr2d, size=2 * radius + 1, mode='reflect')

    actual_erode = ct.grayscale_erosion_2d(arr2d, radius=radius, element="box")
    actual_dilate = ct.grayscale_dilation_2d(arr2d, radius=radius, element="box")
    actual_open = ct.grayscale_opening_2d(arr2d, radius=radius, element="box")
    actual_close = ct.grayscale_closing_2d(arr2d, radius=radius, element="box")

    np.testing.assert_array_equal(actual_erode, expected_erode)
    np.testing.assert_array_equal(actual_dilate, expected_dilate)
    np.testing.assert_array_equal(actual_open, expected_open)
    np.testing.assert_array_equal(actual_close, expected_close)

    # Ball / Disk footprint
    dys, dxs = np.mgrid[-radius:radius+1, -radius:radius+1]
    disk_footprint = (dys**2 + dxs**2 <= radius**2)
    expected_disk_erode = ndimage.grey_erosion(arr2d, footprint=disk_footprint, mode='reflect')
    expected_disk_dilate = ndimage.grey_dilation(arr2d, footprint=disk_footprint, mode='reflect')
    actual_disk_erode = ct.grayscale_erosion_2d(arr2d, radius=radius, element="ball")
    actual_disk_dilate = ct.grayscale_dilation_2d(arr2d, radius=radius, element="ball")

    np.testing.assert_array_equal(actual_disk_erode, expected_disk_erode)
    np.testing.assert_array_equal(actual_disk_dilate, expected_disk_dilate)

def test_morphology_properties():
    rng = np.random.default_rng(202)
    arr = rng.uniform(0.0, 50.0, (20, 20)).astype(np.float32)
    for elem in ["box", "ball"]:
        opened = ct.grayscale_opening_2d(arr, radius=1, element=elem)
        closed = ct.grayscale_closing_2d(arr, radius=1, element=elem)
        eroded = ct.grayscale_erosion_2d(arr, radius=1, element=elem)
        dilated = ct.grayscale_dilation_2d(arr, radius=1, element=elem)

        # 1. opening <= input <= closing
        assert np.all(opened <= arr + 1e-6)
        assert np.all(arr <= closed + 1e-6)

        # 2. Idempotence
        np.testing.assert_array_equal(ct.grayscale_opening_2d(opened, radius=1, element=elem), opened)
        np.testing.assert_array_equal(ct.grayscale_closing_2d(closed, radius=1, element=elem), closed)

        # 3. Duality: erosion(x) == -dilation(-x)
        neg_dilated = ct.grayscale_dilation_2d(-arr, radius=1, element=elem)
        np.testing.assert_allclose(eroded, -neg_dilated, atol=1e-6)

def test_tv_chambolle_vs_skimage():
    # 2D test
    rng = np.random.default_rng(777)
    arr2d = rng.uniform(0.0, 1.0, (24, 24)).astype(np.float32)
    weight = 0.08
    eps = 1e-4
    max_iter = 100
    sk_out2d = denoise_tv_chambolle(arr2d, weight=weight, eps=eps, max_num_iter=max_iter)
    ct_out2d = ct.tv_chambolle_2d(arr2d, weight=weight, eps=eps, max_iterations=max_iter)
    data_range2d = float(arr2d.max() - arr2d.min())
    np.testing.assert_allclose(ct_out2d, sk_out2d, atol=1e-4 * data_range2d, rtol=1e-3)

    # 3D test
    arr3d = rng.uniform(0.0, 1.0, (10, 10, 10)).astype(np.float32)
    sk_out3d = denoise_tv_chambolle(arr3d, weight=weight, eps=eps, max_num_iter=60)
    ct_out3d = ct.tv_chambolle_3d(arr3d, weight=weight, eps=eps, max_iterations=60)
    data_range3d = float(arr3d.max() - arr3d.min())
    np.testing.assert_allclose(ct_out3d, sk_out3d, atol=1e-4 * data_range3d, rtol=1e-3)

def test_diffusion_vs_numpy_reference():
    rng = np.random.default_rng(789)
    # 2D 16x16
    arr2d = rng.uniform(0.0, 1.0, (16, 16)).astype(np.float32)
    expected2d = numpy_diffusion_2d(arr2d, kappa=0.05, iterations=4, dt=0.20)
    actual2d = ct.anisotropic_diffusion_2d(arr2d, kappa=0.05, iterations=4, dt=0.20)
    np.testing.assert_allclose(actual2d, expected2d, atol=1e-5, rtol=1e-5)

    # 3D 8x8x8
    arr3d = rng.uniform(0.0, 1.0, (8, 8, 8)).astype(np.float32)
    expected3d = numpy_diffusion_3d(arr3d, kappa=0.05, iterations=4, dt=0.15)
    actual3d = ct.anisotropic_diffusion_3d(arr3d, kappa=0.05, iterations=4, dt=0.15)
    np.testing.assert_allclose(actual3d, expected3d, atol=1e-5, rtol=1e-5)

def test_error_handling():
    # 1. Empty array
    empty = np.array([], dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        ct.gaussian_filter_2d(empty)

    # 2. Wrong dimensions (1D instead of 2D)
    arr1d = np.ones((10,), dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        ct.gaussian_filter_2d(arr1d)

    # 3. 2D instead of 3D
    arr2d = np.ones((10, 10), dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        ct.gaussian_filter_3d(arr2d)

    # 4. NaN sigma
    with pytest.raises((ValueError, RuntimeError)):
        ct.gaussian_filter_2d(arr2d, sigma=float('nan'))

    # 5. Negative/zero sigma
    with pytest.raises((ValueError, RuntimeError)):
        ct.gaussian_filter_2d(arr2d, sigma=-1.0)

def test_copies_input_and_output():
    # Verify input is copied and cannot be mutated by C++
    arr = np.ones((10, 10), dtype=np.float32) * 5.0
    out = ct.gaussian_filter_2d(arr, sigma=1.0)
    assert np.allclose(arr, 5.0)  # Input untouched

    # Mutating out does not change future calls
    out[:] = 999.0
    out2 = ct.gaussian_filter_2d(arr, sigma=1.0)
    assert np.allclose(out2, 5.0)
