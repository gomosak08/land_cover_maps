import matplotlib.pyplot as plt
import torch

def _minmax(img:torch.Tensor, eps=1e-6):
    # img: (H,W,3)
    mn = img.amin(dim=(0,1), keepdim=True)
    mx = img.amax(dim=(0,1), keepdim=True)
    return (img - mn) / (mx - mn + eps)

def show_segmentation_results(learn, nrows=1, ncols=3, rgb_indices=(3,2,1), vmax=None):
    """
    Display input (selected bands), true mask, and predicted mask.
    rgb_indices: tuple of 3 band indices to visualize (for 25-band inputs).
    """
    xb, yb = learn.dls.one_batch()
    with torch.no_grad():
        preds = learn.model(xb)

    yb = yb.cpu()
    xb = xb.cpu()
    preds = preds.cpu()

    for i in range(nrows):
        fig, axes = plt.subplots(1, ncols, figsize=(15, 5))

        # input visualization
        sel = xb[i][list(rgb_indices), :, :].permute(1, 2, 0)  # (H,W,3)
        vis = _minmax(sel)
        axes[0].imshow(vis.numpy())
        axes[0].set_title(f"Input bands {rgb_indices}")

        # true mask
        true_mask = yb[i]
        axes[1].imshow(true_mask, cmap='inferno', vmax=vmax)
        axes[1].set_title("True Mask")

        # predicted mask
        pred_mask = preds[i].argmax(dim=0)
        axes[2].imshow(pred_mask, cmap='inferno', vmax=vmax)
        axes[2].set_title("Predicted Mask")

        for ax in axes: ax.axis('off')
        plt.tight_layout()
    plt.show()
