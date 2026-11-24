# -*- coding: utf-8 -*-
"""
make_geo_split.py - build the location-disjoint train/val split for EuroSAT-S1.

The Copernicus-Bench split is random, so neighbouring chips are in both training and
validation. This script groups the chip coordinates (train + val) into 60 k-means
clusters and puts each whole cluster in training or in validation. The test split is
left as it is and is not used.

Input:  the Sentinel-1 cache from prepare_s1.py (default ./data/eurosat_s1_cache.npz).
Output: geo_split.npz with one array `splits` (train/val/test per row of the cache).
The geo_split.npz shipped with this repository is the output of this script; train.py
uses that copy by default.

    python make_geo_split.py
"""
import argparse, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

SEED = 0
K_CLUSTERS = 60


def km_coords(lonlat):
    lat0 = np.deg2rad(lonlat[:, 1].mean())
    return np.column_stack([lonlat[:, 0] * 111.32 * np.cos(lat0),
                            lonlat[:, 1] * 110.57])


ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="./data/eurosat_s1_cache.npz")
ap.add_argument("--out", default="./data/geo_split.npz")
args = ap.parse_args()
OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

z = np.load(args.cache, allow_pickle=True)
coords, labels, splits = z["coords"], z["labels"], z["splits"]
XY = km_coords(coords)

tr_m, va_m = splits == "train", splits == "val"
keep = tr_m | va_m
print(f"cache {len(coords)} chips   Copernicus-Bench train {tr_m.sum()}  val {va_m.sum()}"
      f"  test {(splits=='test').sum()} (untouched)")

rng = np.random.default_rng(SEED)
km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit(XY[keep])
cl = np.full(len(coords), -1)
cl[keep] = km.labels_

target = va_m.sum() / keep.sum()
order = rng.permutation(K_CLUSTERS)          # first use of rng
sizes = np.array([(cl == k).sum() for k in range(K_CLUSTERS)])
val_clusters, running = [], 0
for k in order:
    if running / keep.sum() < target:
        val_clusters.append(k); running += sizes[k]
val_clusters = set(val_clusters)

new = splits.copy()
new[keep] = np.where(np.isin(cl[keep], list(val_clusters)), "val", "train")

tr2, va2 = new == "train", new == "val"
print(f"geo   clusters {K_CLUSTERS-len(val_clusters)} train / {len(val_clusters)} val")
print(f"geo   chips    {tr2.sum()} train / {va2.sum()} val")

# check that the split separates in space
d_off, _ = cKDTree(XY[tr_m]).query(XY[va_m], k=1)
d_new, _ = cKDTree(XY[tr2]).query(XY[va2], k=1)
print(f"\nmedian distance to nearest train chip:")
print(f"   Copernicus-Bench  {np.median(d_off):8.3f} km")
print(f"   geo               {np.median(d_new):8.3f} km")
if np.median(d_new) < 10 * np.median(d_off):
    raise SystemExit("geo split is not meaningfully more separated - aborting")

# class balance is not guaranteed by a spatial split; report it
print("\nclass counts (train / val):")
for c in range(int(labels.max()) + 1):
    print(f"   class {c}: {int(((labels==c)&tr2).sum()):5d} / {int(((labels==c)&va2).sum()):5d}")
missing = [c for c in range(int(labels.max()) + 1) if ((labels == c) & va2).sum() == 0]
if missing:
    raise SystemExit(f"classes absent from val: {missing} - split unusable")

np.savez(OUT, splits=new.astype(str))
print(f"\n-> {OUT}")
