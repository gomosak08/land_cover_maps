
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_side_by_side(img_id, base_path, chnls=None, img_cmap=None, mask_cmap="inferno"):
    """
    Show image and mask side-by-side (no overlay).

    Args:
        img_id (str | int): Identifier used in the file name (e.g., 12 -> "12.npy").
        base_path (str): Base folder containing "img_data" and "img_mask".
        chnls (int | None): Channel index to visualize if image is 3D (C, H, W).
                            If None, channel 0 is used.
        img_cmap (str): Matplotlib colormap for the image panel.
        mask_cmap (str): Matplotlib colormap for the mask panel.
    """
    img_path = os.path.join(base_path, "img_data", f"{img_id}.npy")
    msk_path = os.path.join(base_path, "img_mask", f"{img_id}.npy")

    im = np.load(img_path)
    mk = np.load(msk_path)

    # Select 2D image for display
    if im.ndim == 3:  # (C, H, W)
        C, H, W = im.shape
        ch = 0 if chnls is None else int(chnls)
        if ch < 0 or ch >= C:
            raise IndexError(f"Channel {ch} out of range [0, {C-1}]")
        img2show = im[ch]
    elif im.ndim == 2:  # (H, W)
        img2show = im
    else:
        raise ValueError(f"Unsupported image ndim: {im.ndim}")

    # Basic checks for mask
    if mk.ndim != 2:
        raise ValueError(f"Mask must be 2D (H, W), got ndim={mk.ndim}")

    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_img, ax_mask = axes

    im1 = ax_img.imshow(img2show, cmap=img_cmap, interpolation="none")
    ax_img.set_title(f"Image {img_id}" + (f" (ch={ch})" if im.ndim == 3 else ""))
    ax_img.axis("off")

    im2 = ax_mask.imshow(mk, cmap=mask_cmap, interpolation="none")
    ax_mask.set_title(f"Mask {img_id}")
    ax_mask.axis("off")

    # Optional colorbars (comment out if not needed)
    fig.colorbar(im1, ax=ax_img, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()