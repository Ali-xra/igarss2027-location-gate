# -*- coding: utf-8 -*-
"""
prepare_s1.py - read the 27,000 EuroSAT-S1 GeoTIFFs once into a single array.

Raw data: EuroSAT-S1 from Copernicus-Bench (Wang et al., CC BY 4.0), Hugging Face
dataset `wangyi111/Copernicus-Bench`, file `l2_eurosat_s1s2/eurosat_s1.zip`.
Unzip it; --src is the folder that contains `all_imgs/` and
`eurosat-train.txt`, `eurosat-val.txt`, `eurosat-test.txt`.

Writes (default ./data/eurosat_s1_cache.npz):
    images  (N, 2, 64, 64)  float32   VV/VH in dB
    coords  (N, 2)          float32   lon, lat  (EPSG:4326), from the chip centre
    labels  (N,)            int64
    splits  (N,)            str       train/val/test, the Copernicus-Bench lists
    names   (N,)            str       file names (used by prepare_rgb.py for row alignment)
    classes (10,)           str

    python prepare_s1.py --src path/to/eurosat_s1
"""
import argparse, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from pyproj import Transformer


def read_split(root, name):
    out = []
    with (root / f"eurosat-{name}.txt").open() as f:
        for line in f:
            fn = line.strip().replace(".jpg", ".tif")
            if fn:
                out.append(fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="unzipped eurosat_s1 folder (all_imgs/ and eurosat-*.txt)")
    ap.add_argument("--out", default="./data/eurosat_s1_cache.npz")
    a = ap.parse_args()
    root = Path(a.src)
    imgs_dir = root / "all_imgs"
    out_f = Path(a.out)
    out_f.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for s in ("train", "val", "test"):
        entries += [(fn, s) for fn in read_split(root, s)]
    n = len(entries)
    print(f"{n} chips")

    classes = sorted({fn.split('_')[0] for fn, _ in entries})
    cls2idx = {c: i for i, c in enumerate(classes)}
    print("classes:", classes)

    images = np.zeros((n, 2, 64, 64), np.float32)
    coords = np.zeros((n, 2), np.float32)
    labels = np.zeros(n, np.int64)
    splits = np.empty(n, object)
    names = np.empty(n, object)

    tf_cache = {}
    t0 = time.time()
    for i, (fn, sp) in enumerate(entries):
        cls = fn.split("_")[0]
        with rasterio.open(imgs_dir / cls / fn) as src:
            arr = src.read().astype(np.float32)
            cx, cy = src.xy(src.height // 2, src.width // 2)
            crs = src.crs.to_string()
        if crs != "EPSG:4326":
            if crs not in tf_cache:
                tf_cache[crs] = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
            lon, lat = tf_cache[crs].transform(cx, cy)
        else:
            lon, lat = cx, cy

        if arr.shape != (2, 64, 64):          # no chip is skipped silently
            raise ValueError(f"{fn}: shape {arr.shape}")
        images[i] = arr
        coords[i] = (lon, lat)
        labels[i] = cls2idx[cls]
        splits[i] = sp
        names[i] = fn
        if (i + 1) % 3000 == 0:
            el = time.time() - t0
            print(f"   [{i+1}/{n}] {el:.0f}s  eta {el/(i+1)*(n-i-1):.0f}s", flush=True)

    if not np.isfinite(images).all():
        raise ValueError("non-finite pixel values in cache")
    if not np.isfinite(coords).all():
        raise ValueError("non-finite coordinates in cache")

    print("\nsplit counts:", {s: int((splits == s).sum()) for s in ("train", "val", "test")})
    print("class counts:", np.bincount(labels).tolist())
    print(f"lon {coords[:,0].min():.2f}..{coords[:,0].max():.2f}   "
          f"lat {coords[:,1].min():.2f}..{coords[:,1].max():.2f}")
    print(f"VV mean {images[:,0].mean():.3f} std {images[:,0].std():.3f}")
    print(f"VH mean {images[:,1].mean():.3f} std {images[:,1].std():.3f}")

    np.savez(out_f, images=images, coords=coords, labels=labels,
             splits=splits.astype(str), names=names.astype(str),
             classes=np.array(classes))
    print(f"\n-> {out_f}  ({out_f.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
