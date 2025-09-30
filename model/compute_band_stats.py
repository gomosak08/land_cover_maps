# compute_band_stats.py
import numpy as np
from pathlib import Path
import argparse
import random

def main(img_dir: str, out_path: str = "band_stats.npz", sample: int = 200):
    img_dir = Path(img_dir)
    files = sorted([p for p in img_dir.glob("*.npy")], key=lambda p: int(p.stem))
    if not files:
        raise RuntimeError(f"No .npy files found in {img_dir}")

    if sample and sample < len(files):
        files = random.sample(files, sample)

    mean = None
    std = None
    n = 0
    for f in files:
        x = np.load(f).astype(np.float32)  # (C,H,W)
        c = x.shape[0]
        if mean is None:
            mean = np.zeros(c, dtype=np.float64)
            std  = np.zeros(c, dtype=np.float64)
        x2 = x.reshape(c, -1)
        mean += x2.mean(axis=1)
        std  += x2.std(axis=1)
        n += 1

    mean /= n
    std  /= n
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved stats to {out_path}\nmean[:5]={mean[:5]}\nstd[:5]={std[:5]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="Folder with (C,H,W) .npy tiles")
    ap.add_argument("--out_path", default="band_stats.npz")
    ap.add_argument("--sample", type=int, default=200, help="Number of tiles to sample (0 = all)")
    args = ap.parse_args()
    main(args.img_dir, args.out_path, args.sample)
