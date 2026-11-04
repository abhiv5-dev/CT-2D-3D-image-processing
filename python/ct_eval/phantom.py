import numpy as np

def generate_phantom_2d(shape=(256, 256)):
    """
    Generates a 2D Shepp-Logan-like numerical phantom with distinct intensities,
    sharp boundaries, flat homogeneous regions, and a small low-contrast target.
    
    Returns:
        phantom (np.ndarray): float32 array in [0, 1]
        masks (dict): Dictionary of ROI masks and profile coords
    """
    H, W = shape
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    img = np.zeros((H, W), dtype=np.float32)

    # Outer head ellipse (a=0.69, b=0.92, intensity=1.0)
    outer = (xx / 0.69)**2 + (yy / 0.92)**2 <= 1.0
    img[outer] = 1.0

    # Large brain/parenchyma background (a=0.6624, b=0.874, intensity=0.2)
    brain = (xx / 0.6624)**2 + (yy / 0.874)**2 <= 1.0
    img[brain] = 0.2

    # Left ventricle (intensity=0.5)
    vent1 = ((xx + 0.22) / 0.16)**2 + ((yy + 0.1) / 0.35)**2 <= 1.0
    img[vent1] = 0.5

    # Right ventricle (intensity=0.7)
    vent2 = ((xx - 0.22) / 0.16)**2 + ((yy + 0.1) / 0.35)**2 <= 1.0
    img[vent2] = 0.7

    # Center tumor/calcification (sharp step, intensity=0.9)
    calc = (xx / 0.08)**2 + ((yy - 0.3) / 0.08)**2 <= 1.0
    img[calc] = 0.9

    # Small low-contrast lesion (intensity=0.24, vs background 0.20)
    lesion = ((xx - 0.2) / 0.05)**2 + ((yy - 0.45) / 0.05)**2 <= 1.0
    img[lesion] = 0.24

    # Planar straight edge slab in upper brain parenchyma (intensity=0.85, straight edge at y=0.52)
    slab = (xx >= -0.25) & (xx <= 0.25) & (yy >= 0.52) & (yy <= 0.72)
    img[slab] = 0.85

    # ROI 1: Homogeneous region in lower brain
    roi1 = ((xx / 0.15)**2 + ((yy + 0.55) / 0.15)**2 <= 1.0) & brain

    # ROI 2: Homogeneous region in left ventricle
    roi2 = ((xx + 0.22) / 0.08)**2 + ((yy + 0.1) / 0.15)**2 <= 1.0

    # Step edge profile line (horizontal line passing through calcification center)
    y_idx = int(np.round((0.3 + 1.0) * 0.5 * (H - 1)))
    x_start = int(np.round((-0.2 + 1.0) * 0.5 * (W - 1)))
    x_end = int(np.round((0.0 + 1.0) * 0.5 * (W - 1)))

    # Planar edge bounds across y = 0.52
    y_edge = int(np.round((0.52 + 1.0) * 0.5 * (H - 1)))
    px0 = int(np.round((-0.18 + 1.0) * 0.5 * (W - 1)))
    px1 = int(np.round((0.18 + 1.0) * 0.5 * (W - 1)))
    py0 = max(0, y_edge - 6)
    py1 = min(H, y_edge + 6)

    masks = {
        "homogeneous_roi_1": roi1,
        "homogeneous_roi_2": roi2,
        "low_contrast_roi": lesion,
        "planar_slab": slab,
        "planar_edge_coords": {"y_range": (py0, py1), "x_range": (px0, px1), "edge_y": y_edge},
        "edge_line": (y_idx, x_start, x_end)
    }

    return img, masks

def generate_phantom_3d(shape=(64, 64, 64)):
    """
    Generates a 3D Shepp-Logan-like volumetric phantom (axis order: z, y, x).
    
    Returns:
        vol (np.ndarray): float32 array in [0, 1]
        masks (dict): Dictionary of 3D ROI masks and profile coords
    """
    D, H, W = shape
    z = np.linspace(-1.0, 1.0, D, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    vol = np.zeros((D, H, W), dtype=np.float32)

    # Outer skull ellipsoid
    outer = (xx / 0.69)**2 + (yy / 0.92)**2 + (zz / 0.85)**2 <= 1.0
    vol[outer] = 1.0

    # Brain parenchyma background
    brain = (xx / 0.66)**2 + (yy / 0.87)**2 + (zz / 0.80)**2 <= 1.0
    vol[brain] = 0.2

    # Left ventricle ellipsoid
    vent1 = ((xx + 0.22) / 0.16)**2 + ((yy + 0.1) / 0.35)**2 + (zz / 0.5)**2 <= 1.0
    vol[vent1] = 0.5

    # Right ventricle ellipsoid
    vent2 = ((xx - 0.22) / 0.16)**2 + ((yy + 0.1) / 0.35)**2 + (zz / 0.5)**2 <= 1.0
    vol[vent2] = 0.7

    # Center calcification sphere
    calc = (xx / 0.08)**2 + ((yy - 0.3) / 0.08)**2 + (zz / 0.08)**2 <= 1.0
    vol[calc] = 0.9

    # Small low-contrast lesion
    lesion = ((xx - 0.2) / 0.06)**2 + ((yy - 0.45) / 0.06)**2 + (zz / 0.06)**2 <= 1.0
    vol[lesion] = 0.24

    # Planar straight edge slab in upper brain parenchyma (intensity=0.85, straight edge at y=0.52)
    slab = (xx >= -0.25) & (xx <= 0.25) & (yy >= 0.52) & (yy <= 0.72) & (zz >= -0.25) & (zz <= 0.25)
    vol[slab] = 0.85

    # 3D Homogeneous region in lower brain
    roi1 = ((xx / 0.15)**2 + ((yy + 0.55) / 0.15)**2 + (zz / 0.3)**2 <= 1.0) & brain

    # 3D Homogeneous region in ventricle
    roi2 = (((xx + 0.22) / 0.08)**2 + ((yy + 0.1) / 0.15)**2 + (zz / 0.2)**2 <= 1.0) & vent1

    y_edge = int(np.round((0.52 + 1.0) * 0.5 * (H - 1)))
    px0 = int(np.round((-0.18 + 1.0) * 0.5 * (W - 1)))
    px1 = int(np.round((0.18 + 1.0) * 0.5 * (W - 1)))
    py0 = max(0, y_edge - 6)
    py1 = min(H, y_edge + 6)
    pz0 = int(np.round((-0.20 + 1.0) * 0.5 * (D - 1)))
    pz1 = int(np.round((0.20 + 1.0) * 0.5 * (D - 1)))

    masks = {
        "homogeneous_roi_1": roi1,
        "homogeneous_roi_2": roi2,
        "low_contrast_roi": lesion,
        "planar_slab": slab,
        "planar_edge_coords": {
            "z_range": (pz0, pz1),
            "y_range": (py0, py1),
            "x_range": (px0, px1),
            "edge_y": y_edge
        },
        "edge_center": (D // 2, int(0.65 * H), int(0.5 * W))
    }

    return vol, masks

def extract_planar_edge_profiles(data, masks=None):
    """
    Extracts a 2D array of aligned parallel line profiles crossing the planar slab edge.
    Supports 2D image (H, W) and 3D volume (D, H, W).
    Returns profiles array of shape (N_lines, profile_len).
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        H, W = arr.shape
        if masks is not None and "planar_edge_coords" in masks:
            c = masks["planar_edge_coords"]
            y0, y1 = c["y_range"]
            x0, x1 = c["x_range"]
        else:
            y_edge = int(np.round((0.52 + 1.0) * 0.5 * (H - 1)))
            x0 = int(np.round((-0.18 + 1.0) * 0.5 * (W - 1)))
            x1 = int(np.round((0.18 + 1.0) * 0.5 * (W - 1)))
            y0 = max(0, y_edge - 6)
            y1 = min(H, y_edge + 6)
        # Vertical profiles: lines along y across [y0, y1] for each x in [x0, x1]
        cols = arr[y0:y1, x0:x1]  # shape: (profile_len, N_lines)
        return cols.T  # shape: (N_lines, profile_len)

    elif arr.ndim == 3:
        D, H, W = arr.shape
        if masks is not None and "planar_edge_coords" in masks:
            c = masks["planar_edge_coords"]
            z0, z1 = c["z_range"]
            y0, y1 = c["y_range"]
            x0, x1 = c["x_range"]
        else:
            y_edge = int(np.round((0.52 + 1.0) * 0.5 * (H - 1)))
            x0 = int(np.round((-0.18 + 1.0) * 0.5 * (W - 1)))
            x1 = int(np.round((0.18 + 1.0) * 0.5 * (W - 1)))
            y0 = max(0, y_edge - 6)
            y1 = min(H, y_edge + 6)
            z0 = int(np.round((-0.20 + 1.0) * 0.5 * (D - 1)))
            z1 = int(np.round((0.20 + 1.0) * 0.5 * (D - 1)))
        
        # Sub-volume shape: (Nz, profile_len, Nx)
        sub = arr[z0:z1, y0:y1, x0:x1]
        # Move profile_len axis (y) to the end: shape (Nz, Nx, profile_len) -> (Nz * Nx, profile_len)
        return np.moveaxis(sub, 1, -1).reshape(-1, y1 - y0)
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={arr.ndim}")
