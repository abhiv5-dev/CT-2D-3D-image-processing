import numpy as np
from skimage.metrics import structural_similarity
from scipy.optimize import curve_fit
from scipy.special import erf

def _erf_model(x, a, b, x0, sigma):
    return a + b * erf((x - x0) / (np.sqrt(2.0) * sigma))

def rmse(clean, test):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((clean - test)**2)))

def psnr(clean, test, data_range=None):
    """
    Peak Signal-to-Noise Ratio (dB).
    If data_range is not provided, defaults to max(clean) - min(clean).
    """
    if data_range is None:
        data_range = float(np.max(clean) - np.min(clean))
    if data_range <= 0.0:
        data_range = 1.0
    err = rmse(clean, test)
    if err <= 1e-12:
        return float('inf')
    return float(20.0 * np.log10(data_range / err))

def ssim(clean, test, data_range=None):
    """
    Structural Similarity Index (SSIM) supporting both 2D and 3D volumes.
    """
    if data_range is None:
        data_range = float(np.max(clean) - np.min(clean))
    if data_range <= 0.0:
        data_range = 1.0
    return float(structural_similarity(clean, test, data_range=data_range))

def roi_noise_std(data, mask):
    """
    Standard deviation of voxel values inside a homogeneous region of interest.
    """
    vals = data[mask]
    if len(vals) == 0:
        return 0.0
    return float(np.std(vals))

def cnr(data, mask1, mask2):
    """
    Contrast-to-Noise Ratio between two homogeneous regions.
    CNR = |mean1 - mean2| / sqrt(0.5 * (std1^2 + std2^2))
    """
    v1 = data[mask1]
    v2 = data[mask2]
    if len(v1) == 0 or len(v2) == 0:
        return 0.0
    m1, s1 = np.mean(v1), np.std(v1)
    m2, s2 = np.mean(v2), np.std(v2)
    denom = np.sqrt(0.5 * (s1**2 + s2**2))
    if denom <= 1e-12:
        return float('inf')
    return float(np.abs(m1 - m2) / denom)

def edge_width_10_90(profiles, plateau_samples=5):
    """
    Calculates sub-voxel 10%-90% rise/fall distance across a step edge line profile.
    Accepts a 1D line profile or a 2D array of aligned parallel profiles (N_lines, profile_len).
    When 2D, profiles are averaged across lines to suppress noise before computing crossings.
    
    Returns edge width in pixels/voxels via linear sub-voxel interpolation,
    or np.nan if the edge contrast is not significantly larger than the noise.
    """
    arr = np.asarray(profiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.nan

    avg = np.mean(arr, axis=0)
    L = len(avg)
    k = min(plateau_samples, L // 4)
    if k < 2:
        return np.nan

    start_val = float(np.mean(avg[:k]))
    end_val = float(np.mean(avg[-k:]))
    start_std = float(np.std(avg[:k]))
    end_std = float(np.std(avg[-k:]))
    noise_std = 0.5 * (start_std + end_std)

    plateau_diff = abs(end_val - start_val)
    # Check if edge step is clearly larger than background noise
    if plateau_diff < max(1e-4, 3.0 * noise_std):
        return np.nan

    rising = end_val > start_val
    if rising:
        norm = (avg - start_val) / plateau_diff
    else:
        norm = (start_val - avg) / plateau_diff

    # Sub-voxel linear interpolation for 10% and 90% crossings
    # 10% crossing
    x10 = None
    for i in range(L - 1):
        if (norm[i] <= 0.1 <= norm[i + 1]) or (norm[i] >= 0.1 >= norm[i + 1]):
            denom = norm[i + 1] - norm[i]
            if abs(denom) > 1e-12:
                x10 = i + (0.1 - norm[i]) / denom
                break
    if x10 is None:
        x10 = 0.0

    # 90% crossing
    x90 = None
    for i in range(L - 2, -1, -1):
        if (norm[i] <= 0.9 <= norm[i + 1]) or (norm[i] >= 0.9 >= norm[i + 1]):
            denom = norm[i + 1] - norm[i]
            if abs(denom) > 1e-12:
                x90 = i + (0.9 - norm[i]) / denom
                break
    if x90 is None:
        x90 = float(L - 1)

    width = abs(x90 - x10)
    return float(width)

def fit_edge_erf(profiles, plateau_samples=5):
    """
    Fits an error-function step edge model: a + b * erf((x - x0) / (sqrt(2) * sigma)).
    Accepts 1D or 2D profiles (averaged along axis 0).
    
    Returns:
        (sigma, width_10_90): Tuple of fitted Gaussian sigma and equivalent 10-90% width (approx 2.563 * sigma).
        Returns (np.nan, np.nan) if fitting fails or edge contrast is below noise.
    """
    arr = np.asarray(profiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.nan, np.nan

    avg = np.mean(arr, axis=0)
    L = len(avg)
    k = min(plateau_samples, L // 4)
    if k < 2:
        return np.nan, np.nan

    start_val = float(np.mean(avg[:k]))
    end_val = float(np.mean(avg[-k:]))
    start_std = float(np.std(avg[:k]))
    end_std = float(np.std(avg[-k:]))
    noise_std = 0.5 * (start_std + end_std)

    plateau_diff = abs(end_val - start_val)
    if plateau_diff < max(1e-4, 3.0 * noise_std):
        return np.nan, np.nan

    x = np.arange(L, dtype=np.float64)
    halfway = 0.5 * (start_val + end_val)
    x0_guess = float(np.argmin(np.abs(avg - halfway)))
    b_guess = 0.5 * (end_val - start_val)
    a_guess = halfway
    p0 = [a_guess, b_guess, x0_guess, 1.5]

    try:
        popt, _ = curve_fit(
            _erf_model, x, avg, p0=p0,
            bounds=([-np.inf, -np.inf, 0.0, 0.01], [np.inf, np.inf, float(L), float(L)]),
            maxfev=3000
        )
        sigma = float(abs(popt[3]))
        width_10_90 = float(2.5631 * sigma)
        return sigma, width_10_90
    except Exception:
        return np.nan, np.nan

def z_consistency(vol, mask=None):
    """
    Inter-slice consistency (flicker along z):
    Mean absolute difference between adjacent slices in a homogeneous region.
    """
    if vol.ndim != 3 or vol.shape[0] < 2:
        return 0.0
    
    diffs = []
    for z in range(vol.shape[0] - 1):
        s1 = vol[z]
        s2 = vol[z + 1]
        if mask is not None:
            if mask.ndim == 3:
                m = mask[z] & mask[z + 1]
            else:
                m = mask
            v1 = s1[m]
            v2 = s2[m]
            if len(v1) > 0 and len(v2) > 0:
                diffs.append(float(np.mean(np.abs(v1 - v2))))
        else:
            diffs.append(float(np.mean(np.abs(s1 - s2))))
            
    if len(diffs) == 0:
        return 0.0
    return float(np.mean(diffs))
