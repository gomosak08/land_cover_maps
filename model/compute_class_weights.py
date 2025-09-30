import numpy as np
from pathlib import Path
import argparse, json

def main(mask_dir: str, num_classes: int, out_json: str = "class_weights.json", scheme: str = "median_freq"):
    mdir = Path(mask_dir)
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in mdir.glob("*.npy"):
        y = np.load(p)  # (H,W) int mask
        for c in range(num_classes):
            counts[c] += (y == c).sum()

    freqs = counts / max(1, counts.sum())

    if scheme == "median_freq":
        w = np.median(freqs[freqs>0]) / np.clip(freqs, 1e-12, None)
    elif scheme == "inv_log":
        w = 1.0 / np.log1p(np.clip(freqs, 1e-12, None))
    elif scheme == "inv":
        w = 1.0 / np.clip(freqs, 1e-12, None)
    else:
        raise ValueError("Unknown scheme")

    # normalize so mean ≈ 1 (keeps CE scale reasonable)
    w = (w / w.mean()).astype(float)

    json.dump({"counts": counts.tolist(), "weights": w.tolist()},
              open(out_json, "w"))
    print(f"Saved {out_json}\ncounts={counts}\nweights={w}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--out_json", default="class_weights.json")
    ap.add_argument("--scheme", default="median_freq", choices=["median_freq","inv_log","inv"])
    args = ap.parse_args()
    main(args.mask_dir, args.num_classes, args.out_json, args.scheme)
