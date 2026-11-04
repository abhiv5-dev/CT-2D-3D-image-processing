import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import ct

def show_interactive_viewer(orig_vol, filtered_vol, display_min=None, display_max=None):
    """
    Interactive matplotlib viewer with a Z-slider displaying side-by-side:
    [Original Volume Slice]  [Filtered Volume Slice]  [Absolute Difference Map]
    with a shared normalized display window.
    """
    assert orig_vol.shape == filtered_vol.shape, "Volumes must have matching shapes"
    depth, height, width = orig_vol.shape

    if display_min is None:
        display_min = float(orig_vol.min())
    if display_max is None:
        display_max = float(orig_vol.max())

    diff_vol = np.abs(orig_vol - filtered_vol)
    diff_max = float(np.max(diff_vol)) * 0.5

    init_z = depth // 2

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    im1 = ax1.imshow(orig_vol[init_z], cmap="gray", vmin=display_min, vmax=display_max)
    ax1.set_title(f"Original (z={init_z})")
    ax1.axis("off")

    im2 = ax2.imshow(filtered_vol[init_z], cmap="gray", vmin=display_min, vmax=display_max)
    ax2.set_title(f"Filtered (z={init_z})")
    ax2.axis("off")

    im3 = ax3.imshow(diff_vol[init_z], cmap="inferno", vmin=0.0, vmax=diff_max)
    ax3.set_title("Absolute Difference |Orig - Filtered|")
    ax3.axis("off")

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, "Slice Z", 0, depth - 1, valinit=init_z, valstep=1)

    def update(val):
        z = int(slider.val)
        im1.set_data(orig_vol[z])
        im2.set_data(filtered_vol[z])
        im3.set_data(diff_vol[z])
        ax1.set_title(f"Original (z={z})")
        ax2.set_title(f"Filtered (z={z})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    from ct_eval.phantom import generate_phantom_3d
    from ct_eval.noise import add_gaussian_noise

    print("Generating demo phantom volume...")
    clean, _ = generate_phantom_3d(shape=(48, 64, 64))
    noisy = add_gaussian_noise(clean, sigma=0.04)
    print("Filtering with 3D Bilateral filter...")
    filtered = ct.bilateral_filter_3d(noisy, sigma_spatial=1.5, sigma_range=0.08)
    print("Launching interactive slice viewer...")
    show_interactive_viewer(noisy, filtered, 0.0, 1.0)
