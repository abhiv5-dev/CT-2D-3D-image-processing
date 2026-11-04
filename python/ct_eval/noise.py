import numpy as np

def add_gaussian_noise(data, sigma=0.05, seed=42):
    """
    Adds zero-mean Gaussian noise with standard deviation `sigma`.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=data.shape).astype(np.float32)
    return (data + noise).astype(np.float32)

def add_poisson_noise(data, peak_photons=10000.0, seed=42):
    """
    Simulates photon transmission/detection quantum noise (Poisson distribution)
    characteristic of CT acquisitions.
    """
    rng = np.random.default_rng(seed)
    # Scale positive intensity to photon count
    min_val = float(np.min(data))
    shifted = data - min_val if min_val < 0 else data
    max_val = float(np.max(shifted))
    if max_val <= 1e-8:
        return data.copy()

    scale = peak_photons / max_val
    counts = np.maximum(0.0, shifted * scale)
    noisy_counts = rng.poisson(lam=counts).astype(np.float32)
    noisy = noisy_counts / scale
    if min_val < 0:
        noisy += min_val
    return noisy.astype(np.float32)

def add_salt_and_pepper_noise(data, amount=0.01, salt_vs_pepper=0.5, seed=42, low=None, high=None):
    """
    Adds salt-and-pepper (impulse) noise to an array.
    amount: fraction of voxels to corrupt (e.g. 0.01 = 1%, 0.02 = 2%).
    salt_vs_pepper: ratio of salt (high) vs pepper (low), default 0.5.
    low, high: intensity values for pepper and salt. Defaults to data min and max (or 0.0 and 1.0).
    """
    rng = np.random.default_rng(seed)
    noisy = data.copy()
    if low is None:
        low = float(np.min(data))
    if high is None:
        high = float(np.max(data))

    total_voxels = data.size
    num_corrupted = int(np.round(amount * total_voxels))
    if num_corrupted == 0:
        return noisy

    indices = rng.choice(total_voxels, size=num_corrupted, replace=False)
    num_salt = int(np.round(salt_vs_pepper * num_corrupted))
    salt_idx = indices[:num_salt]
    pepper_idx = indices[num_salt:]

    flat = noisy.reshape(-1)
    flat[salt_idx] = high
    flat[pepper_idx] = low
    return noisy.astype(np.float32)

