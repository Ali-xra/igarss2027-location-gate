# One Model With or Without Location

Code for the paper

> **One Model With or Without Location: Gated Mid-Network Injection of Geographic
> Coordinates into Earth Observation Foundation Models.**
> A. Soleymani, IGARSS 2027 (submitted).

The method encodes longitude and latitude with a fixed sinusoidal function. A small
gate, read from the tokens at three quarters of the encoder depth, scales this vector,
and the result is added to every token. During training the location vector is removed
with probability 0.3, so one model works with and without coordinates. It is compared
with no location, with the same injection without the gate, and with the original
location mechanism of each backbone (TerraMind v1 large, Prithvi-EO-2.0 300M TL).

## Files

| File | Purpose |
|---|---|
| `model.py` | location encoding, gate, and the four settings for both backbones |
| `train.py` | training and evaluation, five seeds, summary with 95% intervals |
| `prepare_s1.py` | EuroSAT-S1 GeoTIFFs to `data/eurosat_s1_cache.npz` |
| `prepare_rgb.py` | EuroSAT RGB to `data/eurosat_rgb_cache.npz`, same row order |
| `make_geo_split.py` | location-disjoint split (60 k-means clusters) |
| `geo_split.npz` | the split used in the paper (output of `make_geo_split.py`) |
| `results/` | the numbers in the paper: `terramind_v1_large.json`, `prithvi_eo_v2_300_tl.json` |

## Install

Used for the paper: Python 3.14.6, torch 2.13.0 (CUDA 12.6 build), terratorch 1.2.10,
numpy 2.5.1. Hardware: one NVIDIA GTX 1070 (8 GB).

```
python -m venv venv
venv/Scripts/activate            # Windows; on Linux: source venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The pretrained weights are downloaded by TerraTorch from Hugging Face on the first run
(`ibm-esa-geospatial/TerraMind-1.0-large`, `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-Coords`,
`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`).

## Data

1. EuroSAT-S1: download `l2_eurosat_s1s2/eurosat_s1.zip` from the Hugging Face dataset
   `wangyi111/Copernicus-Bench` and unzip it.
   ```
   python prepare_s1.py --src path/to/eurosat_s1
   ```
2. EuroSAT RGB: download `EuroSAT_RGB.zip` (links at https://github.com/phelber/EuroSAT).
   ```
   python prepare_rgb.py --zip path/to/EuroSAT_RGB.zip
   ```
3. Location-disjoint split for EuroSAT-S1: `geo_split.npz` in this repository is used
   by default. To rebuild it (written to `data/geo_split.npz`):
   ```
   python make_geo_split.py
   ```

TerraMind uses Sentinel-1 with the location-disjoint split (15,859 training, 5,741
validation chips). Prithvi-EO-2.0 uses RGB with the random Copernicus-Bench split
(16,200 training, 5,400 validation chips). The test split is not used.

## Experiments

Both tables of the paper come from these two commands (five seeds x four settings
each; every run in its own process):

```
python train.py --driver terramind_v1_large
python train.py --driver prithvi_eo_v2_300_tl
```

Options: `--data-dir` (default `./data`), `--out-dir` (default `./results`),
`--geo-split` (default `geo_split.npz` next to `train.py`). One run alone:
`python train.py --single terramind_v1_large gate_add dropout 0`.

Results land in `--out-dir`: `<backbone>_cache.json` (one entry per run),
`<backbone>_result.json` (summary) and `train.log`. The result files of the paper,
`results/terramind_v1_large.json` and `results/prithvi_eo_v2_300_tl.json`, have the same
`cells` and `paired_differences` structure.

How the result file maps to the paper:

| Paper | Key in `cells` |
|---|---|
| No location | `none\|control` |
| Original | `native\|control` (`drop` = coordinate tokens left out, TerraMind only) |
| Ungated injection | `loc_add\|dropout` |
| Gated injection | `gate_add\|dropout` |

Conditions: `full` = Full, `zero` = Missing, `shuffle` = Shuffled. "Two separate models"
in the accuracy table is `native|control` for Full and Shuffled and `none|control` for
Missing. The paired-differences table is `paired_differences` (Gated minus Original,
No location and Ungated), mean over seeds with a 95% t-interval.

## Licences and attribution

- Code: Apache-2.0 (see `LICENSE`).
- EuroSAT: Helber et al., MIT licence.
- EuroSAT-S1 from Copernicus-Bench: Wang et al., CC BY 4.0.
- TerraMind and Prithvi-EO-2.0 weights: Apache-2.0.

## Citation

A. Soleymani, "One Model With or Without Location: Gated Mid-Network Injection of
Geographic Coordinates into Earth Observation Foundation Models," IGARSS 2027 (submitted).
