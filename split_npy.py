import os
import numpy as np
import rasterio
import argparse


def load_raster_as_array(file_path):
    """
    Loads a raster file into a NumPy array.

    Args:
        file_path (str): Path to the raster file.

    Returns:
        np.ndarray: Raster data as array (C, H, W).
    """
    with rasterio.open(file_path) as src:
        return src.read()


def preprocess_mask(mask):
    """
    Preprocess a mask by remapping values.

    Args:
        mask (np.ndarray): Input mask array.

    Returns:
        np.ndarray: Preprocessed mask.
    """
    mask = np.squeeze(mask)
    value_map = {
        -5: 0, 2: 1, 3: 1, 6: 1, 12: 1, 28: 3, 29: 2,
        30: 5, 31: 6, 32: 4, 280: 3, 14: 1, 21: 3,
        23: 3, 25: 3, 26: 3, 27: 3, 290: 2
    }
    for old_value, new_value in value_map.items():
        mask[mask == old_value] = new_value
    return mask


def cut_borders(array, border_x, border_y):
    """
    Crop borders from a 2D or 3D array.

    Args:
        array (np.ndarray): Array to crop.
        border_x (tuple): (top, bottom) pixels to remove.
        border_y (tuple): (left, right) pixels to remove.

    Returns:
        np.ndarray: Cropped array.
    """
    x0, x1 = border_x
    y0, y1 = border_y
    x1_slice = None if x1 == 0 else -x1
    y1_slice = None if y1 == 0 else -y1
    return array[..., x0:x1_slice, y0:y1_slice]


def divide_image_np(image, rows, cols):
    """
    Divide a 3D image into smaller tiles.

    Args:
        image (np.ndarray): Array with shape (C, H, W).
        rows (int): Number of rows.
        cols (int): Number of cols.

    Returns:
        np.ndarray: Array of sub-images.
    """
    h, w = image[0].shape
    return np.array([
        image[:, h // rows * row:h // rows * (row + 1),
                 w // cols * col:w // cols * (col + 1)]
        for row in range(rows) for col in range(cols)
    ])


def divide_image_masks(mask, rows, cols):
    """
    Divide a 2D mask into smaller tiles.

    Args:
        mask (np.ndarray): Mask array with shape (H, W).
        rows (int): Number of rows.
        cols (int): Number of cols.

    Returns:
        np.ndarray: Array of sub-masks.
    """
    h, w = mask.shape
    return np.array([
        mask[h // rows * row:h // rows * (row + 1),
             w // cols * col:w // cols * (col + 1)]
        for row in range(rows) for col in range(cols)
    ])


def save_images_and_masks(images, masks, image_dir, mask_dir):
    """
    Save images and masks as .npy files.

    Args:
        images (np.ndarray): Sub-images array.
        masks (np.ndarray): Sub-masks array.
        image_dir (str): Output directory for images.
        mask_dir (str): Output directory for masks.
    """
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        if not np.all(mask == 0):  # skip empty masks
            mask_array = mask.astype(np.float32)
            np.save(f'{mask_dir}/{i}.npy', mask_array)

            img_array = images[i].astype(np.float32)
            np.save(f'{image_dir}/{i}.npy', img_array)


def get_values(xy, px, py):
    """
    Compute cropping margins and number of rows/cols for tiling.

    Args:
        xy (tuple): Shape of array (C, H, W).
        px (int): Tile width.
        py (int): Tile height.

    Returns:
        tuple: (border_x, border_y, rows, cols)
    """
    _, x, y = xy

    rows = x // py
    cols = y // px

    if rows == 0 or cols == 0:
        raise ValueError("Tile size is larger than the image.")

    diffx = x - (py * rows)
    diffy = y - (px * cols)

    bx = (diffx // 2 + 1, diffx // 2 + 1) if (diffx // 2) % 2 != 0 else (diffx // 2, diffx // 2)
    by = (diffy // 2, diffy // 2 + 1) if (diffy // 2) % 2 != 0 else (diffy // 2, diffy // 2)

    return bx, by, rows, cols


def main(mask_path, data_path, image_dir, mask_dir, tile_width, tile_height):
    """
    Main pipeline:
        - Load raster mask and data
        - Preprocess mask
        - Handle NaN in data
        - Crop borders
        - Split into tiles
        - Save to .npy
    """
    mask_array = load_raster_as_array(mask_path)
    image_array = load_raster_as_array(data_path)

    mask_array = preprocess_mask(mask_array)
    image_array = np.nan_to_num(image_array)

    shape = image_array.shape  # (C, H, W)
    border_x, border_y, r, c = get_values(shape, tile_width, tile_height)

    cut_mask = cut_borders(mask_array, border_x, border_y)
    cut_image = cut_borders(image_array, border_x, border_y)

    sub_images = divide_image_np(cut_image, r, c)
    sub_masks = divide_image_masks(cut_mask, r, c)

    save_images_and_masks(sub_images, sub_masks, image_dir, mask_dir)
    print(f"Done ✅\nImages saved in: {image_dir}\nMasks saved in: {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raster tiling pipeline")
    parser.add_argument("--mask_path", required=True, help="Path to mask raster (.tif)")
    parser.add_argument("--data_path", required=True, help="Path to data raster (.tif)")
    parser.add_argument("--image_dir", default="i", help="Output folder for images (.npy)")
    parser.add_argument("--mask_dir", default="m", help="Output folder for masks (.npy)")
    parser.add_argument("--tile_width", type=int, required=True, help="Tile width in pixels")
    parser.add_argument("--tile_height", type=int, required=True, help="Tile height in pixels")

    args = parser.parse_args()

    main(
        args.mask_path,
        args.data_path,
        args.image_dir,
        args.mask_dir,
        args.tile_width,
        args.tile_height,
    )
    """
    python3 split_npy.py --mask_path /home/gomosak/conafor_archivo/segmentacion/cnn/data/mask.tif
    --data_path /home/gomosak/conafor_archivo/segmentacion/cnn/data/mosaic_clip.tif 
    --image_dir /home/gomosak/conafor_archivo/segmentacion/cnn/img_data --mask_dir /home/gomosak/conafor_archivo/segmentacion/cnn/img_mask 
    --tile_width 13175 --tile_height 19152
    """