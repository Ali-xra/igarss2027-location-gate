# -*- coding: utf-8 -*-
"""
train.py - five seeds x four settings per backbone (Tables 1 and 2 of the paper).

Each training run happens in a fresh process (--single); --driver starts the missing
runs one by one and then writes the summary. In one long process GPU memory fills up
from run to run and training becomes several times slower.

    python train.py --driver terramind_v1_large
    python train.py --driver prithvi_eo_v2_300_tl
    python train.py --single terramind_v1_large gate_add dropout 0     # one run

Configuration: blocks 0..17 frozen, 10 epochs, batch 64, AdamW (weight decay 0.05),
one-cycle schedule (max lr 3e-4, pct_start 0.15), random horizontal and vertical
flips, location dropout p = 0.3, accuracy at the end of training (no model selection).
Evaluation on the validation split in three conditions: full (true coordinates),
zero (no location), shuffle (coordinates permuted across validation chips with one
fixed permutation, generator seed 777). The original TerraMind mechanism is also
evaluated with the coordinate tokens dropped from the sequence ("drop").

Outputs in --out-dir:
    <backbone>_cache.json    one entry per finished run (arm|mode|seed)
    <backbone>_result.json   per-seed accuracies, mean, s.d., 95% t-interval,
                             paired differences
    train.log                progress log
"""
import argparse, subprocess, sys, io, json, time, traceback, platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import model as M

FREEZE_N, DIM, EPOCHS, BS, MAX_LR = M.FREEZE_N, M.DIM, 10, 64, 3e-4
P_DROP   = M.P_DROP
CONDS    = ("full", "zero", "shuffle")
SEEDS5   = (0, 1, 2, 3, 4)
CELLS = (("none", "control", SEEDS5), ("gate_add", "dropout", SEEDS5),
         ("loc_add", "dropout", SEEDS5), ("native", "control", SEEDS5))
PAIRS = (("gate_add|dropout", "native|control"),
         ("gate_add|dropout", "none|control"),
         ("gate_add|dropout", "loc_add|dropout"))
ALL_CONDS = ("full", "zero", "drop", "shuffle")
# t_{0.975, n-1}
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}

HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
ARGS = None


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# ------------------------------------------------------------------ data
def load_data(backbone):
    data_dir = Path(ARGS.data_dir)
    if backbone.startswith("terramind"):          # Sentinel-1, location-disjoint split
        z = np.load(data_dir / "eurosat_s1_cache.npz", allow_pickle=True)
        img, crd, lab = z["images"], z["coords"], z["labels"]
        sp = np.load(ARGS.geo_split, allow_pickle=True)["splits"]
        data_desc = "eurosat_s1_cache.npz + geo_split.npz"
    else:                                          # RGB, Copernicus-Bench random split
        z = np.load(data_dir / "eurosat_rgb_cache.npz", allow_pickle=True)
        img, crd, lab, sp = z["images"], z["coords"], z["labels"], z["splits"]
        data_desc = "eurosat_rgb_cache.npz (splits inside)"
    tr, va = sp == "train", sp == "val"
    mu = img[tr].mean(axis=(0, 2, 3), keepdims=True)
    sd = img[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    imgn = (img - mu) / sd
    D = dict(
        Xtr=torch.from_numpy(imgn[tr]).float().to(dev),
        Ctr=torch.from_numpy(crd[tr]).float().to(dev),
        ytr=torch.from_numpy(lab[tr]).long().to(dev),
        Xva=torch.from_numpy(imgn[va]).float().to(dev),
        Cva=torch.from_numpy(crd[va]).float().to(dev),
        yva=lab[va], n_cls=int(lab.max()) + 1,
        n_train=int(tr.sum()), n_val=int(va.sum()), desc=data_desc)
    g = torch.Generator().manual_seed(777)              # one fixed shuffle permutation
    D["perm"] = torch.randperm(len(D["Cva"]), generator=g).to(dev)
    D["Cva_shuf"] = D["Cva"][D["perm"]]
    return D


def _tm_tokens(D):
    """TerraMind coordinate tokens for train, val and shuffled val (same permutation)."""
    if "TokTr" in D:
        return
    D["TokTr"] = M.terramind_coords_tokens(D["Ctr"], dev)
    D["TokVa"] = M.terramind_coords_tokens(D["Cva"], dev)
    pv = D["perm"]
    D["TokVaShuf"] = {"tensor": D["TokVa"]["tensor"][pv], "input_mask": D["TokVa"]["input_mask"][pv]}
    print(f"   coords tokens: train {tuple(D['TokTr']['tensor'].shape)} "
          f"val {tuple(D['TokVa']['tensor'].shape)}", flush=True)


def _take(t, sl):
    return t[sl]


# ------------------------------------------------------------------ one run
def train_and_probe(backbone, D, arm, mode, seed):
    native = arm == "native"
    native_tm = native and backbone.startswith("terramind")
    if native_tm:
        _tm_tokens(D)                                   # before seeding
    torch.manual_seed(seed); np.random.seed(seed)
    bb, mods, head, fwd = M.build_setting(backbone, arm, D["n_cls"], dev)

    params = [p for p in bb.parameters() if p.requires_grad] \
        + list(mods.parameters()) + list(head.parameters())
    inj = sum(p.numel() for p in mods.parameters())
    opt = torch.optim.AdamW(params, lr=MAX_LR, weight_decay=0.05)
    if native_tm:
        Ctr, Cva, Cva_shuf, take = D["TokTr"], D["TokVa"], D["TokVaShuf"], M.slice_tokens
    else:
        Ctr, Cva, Cva_shuf, take = D["Ctr"], D["Cva"], D["Cva_shuf"], _take
    Xtr, ytr = D["Xtr"], D["ytr"]
    n = len(ytr)
    steps = (n + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR,
                                                total_steps=EPOCHS * steps, pct_start=0.15)
    if native:
        train_mode = "full"                             # original mechanism: location always present
    else:
        train_mode = "real" if mode == "control" else "drop"
    t0 = time.time()
    for ep in range(EPOCHS):
        bb.train(); mods.train(); head.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            xb, yb, cb = Xtr[idx], ytr[idx], take(Ctr, idx)
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[3])
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[2])
            loss = F.cross_entropy(fwd(xb, cb, train_mode), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"        ep{ep+1:02d}/{EPOCHS}  {(time.time()-t0)/60:.1f} min", flush=True)

    bb.eval(); mods.eval(); head.eval()
    res = {"inj_params": inj}
    for cond in (ALL_CONDS if native_tm else CONDS):
        src = Cva_shuf if cond == "shuffle" else Cva
        if native:
            emode = cond
        else:
            emode = "zero" if cond == "zero" else "real"
        with torch.no_grad():
            preds = [fwd(D["Xva"][i:i+256], take(src, slice(i, i+256)), emode).argmax(1).cpu()
                     for i in range(0, len(D["Xva"]), 256)]
            pred = torch.cat(preds).numpy()
        res[cond] = float((pred == D["yva"]).mean()) * 100

    del bb, mods, head, fwd, opt
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ bookkeeping
def _read(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def cache_file(backbone):
    return Path(ARGS.out_dir) / f"{backbone}_cache.json"


def get(backbone, D, arm, mode, seed):
    cache_f = cache_file(backbone)
    key = f"{arm}|{mode}|s{seed}"
    cache = _read(cache_f)
    if key in cache:
        print(f"   {key:28s} (cached)", flush=True)
        return cache[key]
    print(f"   {key:28s} running...  ({now()})", flush=True)
    t_start = now(); t0 = time.time()
    res = train_and_probe(backbone, D, arm, mode, seed)
    res.update(seed=seed, started=t_start, finished=now(),
               minutes=round((time.time() - t0) / 60, 2))
    cache = _read(cache_f)
    cache[key] = res
    cache_f.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"   {key:28s} ({res['minutes']:.1f} min)  " +
          "  ".join(f"{c} {res[c]:.2f}" for c in CONDS) +
          f"   inj_params {res['inj_params']}", flush=True)
    return res


def stats(vals):
    a = np.asarray(vals, dtype=float)
    n = len(a)
    out = {"n": n, "mean": float(a.mean()),
           "sd": float(a.std(ddof=1)) if n > 1 else None}
    if n > 1:
        h = T975[n] * out["sd"] / np.sqrt(n)
        out["ci95"] = [out["mean"] - h, out["mean"] + h]
    return out


def run_backbone(backbone):
    cache_f = cache_file(backbone)
    out_f = Path(ARGS.out_dir) / f"{backbone}_result.json"
    t_start = now(); t0 = time.time()
    print(f"\n=== {backbone}  start {t_start}", flush=True)
    D = load_data(backbone)
    print(f"    data {D['desc']}  train {D['n_train']}  val {D['n_val']}", flush=True)

    max_seed = max(len(s) for _, _, s in CELLS)
    for seed in range(max_seed):
        for arm, mode, seeds in CELLS:
            if seed in seeds:
                get(backbone, D, arm, mode, seed)

    cache = _read(cache_f)
    cells = {}
    for arm, mode, seeds in CELLS:
        name = f"{arm}|{mode}"
        per = {f"s{s}": cache[f"{name}|s{s}"] for s in seeds}
        cells[name] = {
            "seeds": list(seeds),
            "inj_params": per["s0"]["inj_params"],
            "per_seed": per,
            "stats": {c: stats([per[f"s{s}"][c] for s in seeds])
                      for c in ALL_CONDS if c in per["s0"]},
        }
    pairs = {}
    for a, b in PAIRS:
        common = [s for s in cells[a]["seeds"] if s in cells[b]["seeds"]]
        d = {c: [cells[a]["per_seed"][f"s{s}"][c] - cells[b]["per_seed"][f"s{s}"][c]
                 for s in common] for c in CONDS
             if c in cells[a]["per_seed"]["s0"] and c in cells[b]["per_seed"]["s0"]}
        pairs[f"{a} - {b}"] = {"seeds": common, "per_seed": d,
                               "stats": {c: stats(d[c]) for c in d}}
    mins = round((time.time() - t0) / 60, 1)

    out_f.write_text(json.dumps({
        "backbone": backbone,
        "config": {"freeze_n": FREEZE_N, "dim": DIM, "epochs": EPOCHS, "bs": BS,
                   "max_lr": MAX_LR, "p_drop": P_DROP, "optimizer": "AdamW wd=0.05",
                   "sched": "OneCycle pct_start=0.15", "shuffle_perm_seed": 777,
                   "accuracy": "end of training (not best epoch)",
                   "gate_input": "mean of patch tokens" if backbone.startswith("terramind") else "CLS",
                   "data": D["desc"], "n_train": D["n_train"], "n_val": D["n_val"]},
        "conditions": "full/zero/shuffle for all; native TerraMind also drop (coords tokens omitted)",
        "ci95": "t-interval, t_{0.975,n-1}, over seeds",
        "cells": cells, "paired_differences": pairs,
        "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                "python": platform.python_version()},
        "started": t_start, "finished": now(), "minutes_this_process": mins,
    }, indent=2), encoding="utf-8")
    print(f"-> {out_f}", flush=True)
    del D
    torch.cuda.empty_cache()


def single(backbone, arm, mode, seed):
    """One run in its own process; GPU memory is fully released when it ends."""
    D = load_data(backbone)
    get(backbone, D, arm, mode, int(seed))


def driver(backbone):
    """Start every missing run in a fresh process, then summarise from the cache."""
    cache_f = cache_file(backbone)
    max_seed = max(len(s) for _, _, s in CELLS)
    for seed in range(max_seed):
        for arm, mode, seeds in CELLS:
            if seed not in seeds or f"{arm}|{mode}|s{seed}" in _read(cache_f):
                continue
            for attempt in (1, 2):
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    "--data-dir", ARGS.data_dir, "--out-dir", ARGS.out_dir,
                                    "--geo-split", ARGS.geo_split,
                                    "--single", backbone, arm, mode, str(seed)],
                                   stdin=subprocess.DEVNULL)
                if f"{arm}|{mode}|s{seed}" in _read(cache_f):
                    break
                print(f"   {arm}|{mode}|s{seed} subprocess exit {r.returncode} (attempt {attempt})", flush=True)
            else:
                raise RuntimeError(f"{arm}|{mode}|s{seed} failed twice in a fresh process")
    run_backbone(backbone)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data",
                    help="folder with eurosat_s1_cache.npz and eurosat_rgb_cache.npz")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--geo-split", default=str(HERE / "geo_split.npz"),
                    help="location-disjoint split for the Sentinel-1 cache")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--driver", nargs="+", metavar="BACKBONE", choices=M.BACKBONES)
    g.add_argument("--single", nargs=4, metavar=("BACKBONE", "ARM", "MODE", "SEED"))
    ARGS = ap.parse_args()
    if ARGS.single:
        bbn, arm, mode, _ = ARGS.single
        if bbn not in M.BACKBONES or (arm, mode) not in M.SETTINGS:
            ap.error(f"unknown setting {bbn} {arm}|{mode}")
    Path(ARGS.out_dir).mkdir(parents=True, exist_ok=True)
    log = io.open(Path(ARGS.out_dir) / "train.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    print(f"\n##### train.py {' '.join(sys.argv[1:])}  start {now()}  device {dev}", flush=True)
    try:
        if ARGS.single:
            single(*ARGS.single)
        else:
            for bbn in ARGS.driver:
                driver(bbn)
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
    print(f"##### train.py end {now()}", flush=True)


if __name__ == "__main__":
    main()
