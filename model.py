# -*- coding: utf-8 -*-
"""
model.py - location encoding and the four model settings of the paper.

Settings (arm|mode), the same for both backbones:
    none|control        no location
    gate_add|dropout    gated mid-network injection (ours), trained with location dropout
    loc_add|dropout     ungated mid-network injection, trained with location dropout
    native|control      the backbone's original location mechanism, location always present

Backbones (loaded with TerraTorch):
    terramind_v1_large     Sentinel-1 GRD input; the gate reads the mean of the patch tokens
    prithvi_eo_v2_300_tl   RGB input; the gate reads the class token

The first FREEZE_N encoder blocks are frozen. The injection happens after them,
at three quarters of the encoder depth (block 18 of 24).
"""
import math

import torch
import torch.nn as nn

FREEZE_N, DIM = 18, 1024
P_DROP = 0.3
BACKBONES = ("terramind_v1_large", "prithvi_eo_v2_300_tl")
SETTINGS = (("none", "control"), ("gate_add", "dropout"),
            ("loc_add", "dropout"), ("native", "control"))


def sincos_location(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fixed sinusoidal encoding of (lon, lat) - no learned parameters, so every
    injection method starts from the same representation of location.

    coords: (B, 2) raw lon/lat in degrees.
    """
    lon = coords[:, 0] / 180.0 * math.pi
    lat = coords[:, 1] / 90.0 * math.pi
    half = dim // 4
    freqs = torch.exp(torch.arange(half, device=coords.device, dtype=torch.float32)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    out = []
    for ang in (lon, lat):
        a = ang[:, None] * freqs[None, :] * 10.0
        out += [torch.sin(a), torch.cos(a)]
    e = torch.cat(out, dim=1)                      # (B, 4*half)
    if e.shape[1] < dim:                           # pad if dim not divisible by 4
        e = torch.cat([e, torch.zeros(e.shape[0], dim - e.shape[1],
                                      device=e.device, dtype=e.dtype)], dim=1)
    return e


def loc_vec(coords, mode):
    """Location vector for the injection settings.
    mode: "real" (true coordinates), "zero" (missing location = zero vector),
    "drop" (training: each sample's vector set to zero with probability P_DROP)."""
    if mode == "zero":
        return torch.zeros(coords.shape[0], DIM, device=coords.device)
    e = sincos_location(coords, DIM)
    if mode == "drop":
        keep = (torch.rand(coords.shape[0], 1, device=e.device) >= P_DROP).float()
        e = e * keep
    return e


# ------------------------------------------------------------------ backbone
def build_backbone(backbone, device, native=False):
    """Pretrained backbone; blocks FREEZE_N.. and the final norm are trainable.
    native=True adds TerraMind's coordinate modality (Prithvi-EO-2.0-TL carries its
    location embedding in every build)."""
    from terratorch import BACKBONE_REGISTRY
    if backbone.startswith("terramind"):
        modalities = ["S1GRD", "coords"] if native else ["S1GRD"]
        bb = BACKBONE_REGISTRY.build(backbone, pretrained=True, modalities=modalities).to(device)
        blocks, norm = bb.encoder, bb.encoder_norm
    else:
        from terratorch.datasets import HLSBands
        bb = BACKBONE_REGISTRY.build(backbone, pretrained=True,
                                     bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]).to(device)
        blocks, norm = bb.blocks, bb.norm
    for p in bb.parameters():
        p.requires_grad_(False)
    for blk in blocks[FREEZE_N:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for p in norm.parameters():
        p.requires_grad_(True)
    return bb


# ------------------------------------------------------------------ injection modules
def _gate_mlp():
    """Gate: 1024 -> 256, SiLU, 256 -> 1. Last layer: weights zero, bias 0.1 (262,657 parameters)."""
    net = nn.Sequential(nn.Linear(DIM, DIM // 4), nn.SiLU(), nn.Linear(DIM // 4, 1))
    nn.init.zeros_(net[2].weight); nn.init.constant_(net[2].bias, 0.1)
    return net


def _to_vec():
    """Ungated injection: SiLU, 1024 -> 1024, zero-initialised (1,049,600 parameters)."""
    net = nn.Sequential(nn.SiLU(), nn.Linear(DIM, DIM))
    nn.init.zeros_(net[1].weight); nn.init.zeros_(net[1].bias)
    return net


def make_modules(backbone, arm, device):
    """Build order (and therefore random-number use) per backbone:
    TerraMind: none -> nothing; gate_add -> gate MLP; loc_add -> to_vec.
    Prithvi-EO-2.0: the gate MLP is always built (also for none); loc_add then builds to_vec.
    Only the modules the setting uses are returned."""
    mods = nn.ModuleDict()
    is_tm = backbone.startswith("terramind")
    if is_tm:
        if arm == "gate_add":
            mods["gate_net"] = _gate_mlp()
        elif arm == "loc_add":
            mods["to_vec"] = _to_vec()
    else:
        mods["gate_net"] = _gate_mlp()
        if arm == "loc_add":
            mods["to_vec"] = _to_vec()
    used = nn.ModuleDict()
    if arm == "gate_add":
        used["gate_net"] = mods["gate_net"]
    elif arm == "loc_add":
        used["to_vec"] = mods["to_vec"]
    return used.to(device)


def make_fwd(backbone, bb, mods, head, arm):
    """Forward pass for none / gate_add / loc_add. fwd(images, coords, loc_mode) -> logits."""
    is_tm = backbone.startswith("terramind")
    blocks = bb.encoder if is_tm else bb.blocks
    norm = bb.encoder_norm if is_tm else bb.norm
    n_blocks = len(blocks)
    if is_tm:
        s1_emb = bb.encoder_embeddings[bb.mod_name_mapping["S1GRD"]]

    def fwd(imgb, coordsb, mode):
        if is_tm:
            d = s1_emb(imgb)
            x = d["x"] + d["emb"]
        else:
            x5d = imgb.unsqueeze(2)
            pos = bb.interpolate_pos_encoding(x5d.shape[-3:])
            x = bb.patch_embed(x5d) + pos[:, 1:, :]
            cls = (bb.cls_token + pos[:, :1, :]).expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
        with torch.no_grad():
            for i in range(FREEZE_N):
                x = blocks[i](x)

        if arm != "none":
            loc = loc_vec(coordsb, mode)
            if arm == "gate_add":
                gin = x.mean(dim=1) if is_tm else x[:, 0]      # TerraMind: mean of tokens, Prithvi: class token
                g = mods["gate_net"](gin)
                v = g.unsqueeze(1) * loc.unsqueeze(1)
            else:
                v = mods["to_vec"](loc).unsqueeze(1)
            x = x + v

        for i in range(FREEZE_N, n_blocks):
            x = blocks[i](x)
        h = norm(x)
        return head(h.mean(dim=1) if is_tm else h[:, 0])

    return fwd


# ------------------------------------------------------------------ original mechanisms
def make_native_fwd(backbone, bb, head):
    """Forward pass with the backbone's original location mechanism.
    Prithvi-EO-2.0-TL: location embedding added to the patch tokens at the input;
        fwd(images, coords, cond), cond "zero" leaves the embedding out.
    TerraMind: coordinate tokens appended to the sequence at the input;
        fwd(images, coord_tokens, cond), cond "zero" sets the coordinate-token embeddings
        to zero, cond "drop" leaves the coordinate tokens out of the sequence."""
    if backbone.startswith("terramind"):
        s1_emb = bb.encoder_embeddings[bb.mod_name_mapping["S1GRD"]]
        coords_emb = bb.encoder_embeddings["coords"]
        n_blocks = len(bb.encoder)

        def fwd(imgb, tok, cond):
            d = s1_emb(imgb)
            x = d["x"] + d["emb"]
            if cond != "drop":                       # drop = coordinate tokens not in the sequence
                dc = coords_emb(tok)
                xc = dc["x"] + dc["emb"]
                if cond == "zero":
                    xc = torch.zeros_like(xc)        # tokens present, embeddings zero
                x = torch.cat([x, xc], dim=1)
            for i in range(FREEZE_N):
                x = bb.encoder[i](x)
            for i in range(FREEZE_N, n_blocks):
                x = bb.encoder[i](x)
            return head(bb.encoder_norm(x).mean(dim=1))

        return fwd

    n_blocks = len(bb.blocks)

    def fwd(imgb, coordsb, cond):
        x5d = imgb.unsqueeze(2)
        pos = bb.interpolate_pos_encoding(x5d.shape[-3:])
        x = bb.patch_embed(x5d) + pos[:, 1:, :]
        if cond != "zero":
            x = x + bb.location_embed_enc(coordsb)      # (B,1,dim), added at the input
        cls = (bb.cls_token + pos[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        with torch.no_grad():
            for i in range(FREEZE_N):
                x = bb.blocks[i](x)
        for i in range(FREEZE_N, n_blocks):
            x = bb.blocks[i](x)
        return head(bb.norm(x)[:, 0])

    return fwd


def terramind_coords_tokens(coords_t, device):
    """Tokenise (lon, lat) with TerraMind's coordinate tokenizer, one sample at a time,
    then pad by hand to a common length."""
    from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
        terramind_v1_coords_tokenizer)
    tok = terramind_v1_coords_tokenizer(pretrained=True)
    c = coords_t.cpu()
    per = [tok.encode(c[i:i+1])["tensor"][0].tolist() for i in range(c.shape[0])]
    L = max(len(s) for s in per)
    ids = torch.zeros(len(per), L, dtype=torch.long)
    msk = torch.ones(len(per), L, dtype=torch.bool)
    for i, s in enumerate(per):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        msk[i, :len(s)] = False
    return {"tensor": ids.to(device), "input_mask": msk.to(device)}


def slice_tokens(t, sl):
    return {"tensor": t["tensor"][sl], "input_mask": t["input_mask"][sl]}


# ------------------------------------------------------------------ one setting
def build_setting(backbone, arm, n_cls, device):
    """Backbone, injection modules, linear head and forward function for one setting.
    Call torch.manual_seed(seed) first; the build order fixes the random initialisation."""
    if arm == "native":
        bb = build_backbone(backbone, device, native=True)
        mods = nn.ModuleDict()
        head = nn.Linear(DIM, n_cls).to(device)
        return bb, mods, head, make_native_fwd(backbone, bb, head)
    bb = build_backbone(backbone, device)
    mods = make_modules(backbone, arm, device)
    head = nn.Linear(DIM, n_cls).to(device)
    return bb, mods, head, make_fwd(backbone, bb, mods, head, arm)
