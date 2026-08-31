"""Multi-view hypothesis extraction, scored vs Faro like extractor_variants.py.

Reprojects sensor depth from neighboring frames into the current camera and
builds each band pixel's candidate set from the pooled multi-view samples:
a surface that is missing/corrupted in this frame's depth map is usually seen
cleanly by a neighbor.

Self-check: reprojected neighbor depth must agree with the current sensor
depth off-band (median gap printed; if large, the camera convention is wrong).

  python extractor_multiview.py --scene <scene_dir> [--max-frames 15]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kt_common import (
    EDGE_DEFS,
    dilate,
    load_depth,
    local_fg_bg,
    mushroom_frames,
    subsample,
    valid_mask,
)
from kt_config import CONFIG


def load_transforms(base: Path):
    d = json.loads((base / "transformations.json").read_text())
    out = {}
    for f in d["frames"]:
        stem = Path(f["file_path"]).stem
        K = (f["fl_x"], f["fl_y"], f["cx"], f["cy"])
        out[stem] = (K, torch.tensor(f["transform_matrix"], dtype=torch.float64))
    return out


def unproject(depth, K):
    """OpenGL/nerfstudio convention: camera looks down -z, y up."""
    fx, fy, cx, cy = K
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                          torch.arange(W, dtype=torch.float64), indexing="ij")
    d = depth.double()
    x = (u - cx) / fx * d
    y = -(v - cy) / fy * d
    z = -d
    return torch.stack([x, y, z, torch.ones_like(d)], dim=-1)  # [H,W,4]


def reproject(depth_j, K_j, c2w_j, K_i, c2w_i, hw_i):
    """Neighbor j's depth -> two maps (amin, amax) of depths seen in camera i."""
    P = unproject(depth_j, K_j).reshape(-1, 4)
    valid = valid_mask(depth_j).reshape(-1)
    P = P[valid]
    w2c_i = torch.linalg.inv(c2w_i)
    pc = (w2c_i @ (c2w_j @ P.T)).T  # [N,4] in camera i
    d_i = -pc[:, 2]
    ok = d_i > CONFIG["depth_tolerance"]
    pc, d_i = pc[ok], d_i[ok]
    fx, fy, cx, cy = K_i
    u = (cx + fx * pc[:, 0] / d_i).round().long()
    v = (cy - fy * pc[:, 1] / d_i).round().long()
    H, W = hw_i
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = (v[inb] * W + u[inb])
    d_i = d_i[inb].float()
    amin = torch.full((H * W,), torch.inf).scatter_reduce(
        0, idx, d_i, reduce="amin", include_self=True)
    amax = torch.full((H * W,), -torch.inf).scatter_reduce(
        0, idx, d_i, reduce="amax", include_self=True)
    amin = torch.where(torch.isfinite(amin), amin, torch.nan).view(H, W)
    amax = torch.where(torch.isfinite(amax), amax, torch.nan).view(H, W)
    return amin, amax


def window_samples(maps, ys, xs, r=1):
    """Gather maps values in a (2r+1)^2 window around band pixels -> [N, K]."""
    H, W = maps[0].shape
    cols = []
    for m in maps:
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy = (ys + dy).clamp(0, H - 1)
                xx = (xs + dx).clamp(0, W - 1)
                cols.append(m[yy, xx])
    return torch.stack(cols, dim=-1)  # [N, K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--max-frames", type=int, default=15)
    ap.add_argument("--offsets", type=int, nargs="+", default=[-15, -10, -5, 5, 10, 15])
    args = ap.parse_args()

    base = args.scene / "iphone" / args.capture
    tfs = load_transforms(base)
    frames = [f for f in mushroom_frames(args.scene, args.capture)
              if f["sensor_depth"] and f["reference_depth"] and f["stem"] in tfs]
    order = {f["stem"]: k for k, f in enumerate(frames)}
    eval_frames = subsample(frames, args.max_frames)

    names = ["own_p10_90", "mv_p10_90", "union_own_mv", "union_full",
             "control_own_x3"]
    acc = {n: {"gate": 0, "cov": 0} for n in names}
    selfcheck = []

    for fr in eval_frames:
        prior = load_depth(fr["sensor_depth"])
        hw = prior.shape[:2]
        ref = load_depth(fr["reference_depth"], hw=hw)
        vref = valid_mask(ref)
        band = dilate(EDGE_DEFS["grad_logd"](prior), CONFIG["hyp_band_dilation"])
        gt_fg, gt_bg = local_fg_bg(ref)
        gt_gap = gt_bg - gt_fg
        meaningful = ((gt_gap > CONFIG["min_gap_abs"])
                      | (gt_gap > CONFIG["min_gap_rel"] * gt_fg)) & torch.isfinite(gt_gap)
        gate = band & vref & meaningful
        ys, xs = torch.nonzero(band, as_tuple=True)
        g_b = gate[ys, xs]
        ref_b = ref[ys, xs]
        tol_b = (CONFIG["commit_tolerance"] * gt_gap)[ys, xs]

        K_i, c2w_i = tfs[fr["stem"]]
        i = order[fr["stem"]]
        maps = []
        for off in args.offsets:
            j = i + off
            if not (0 <= j < len(frames)):
                continue
            nb = frames[j]
            K_j, c2w_j = tfs[nb["stem"]]
            dj = load_depth(nb["sensor_depth"])
            amin, amax = reproject(dj, K_j, c2w_j, K_i, c2w_i, hw)
            maps.extend([amin, amax])
        if not maps:
            continue
        # convention self-check on off-band pixels of the first neighbor map
        off_ok = ~band & valid_mask(prior) & torch.isfinite(maps[0])
        if off_ok.any():
            selfcheck.append(float((maps[0] - prior).abs()[off_ok].median()))

        samp = window_samples(maps, ys, xs, r=1)  # [N, K] depths (nan-padded)
        inv = torch.where(torch.isfinite(samp) & (samp > 0), 1.0 / samp, torch.nan)
        q = torch.nanquantile(inv, torch.tensor([0.10, 0.90], dtype=inv.dtype), dim=-1)
        mv_bg = 1.0 / q[0].clamp(min=1e-6)
        mv_fg = 1.0 / q[1].clamp(min=1e-6)

        # own-frame percentile pair (same recipe as production extractor)
        v = valid_mask(prior)
        invp = torch.where(v, 1.0 / prior.clamp(min=1e-6), torch.nan)
        win, pad = CONFIG["hyp_window"], CONFIG["hyp_window"] // 2
        patches = F.pad(invp.view(1, 1, *hw), (pad,) * 4, value=torch.nan)
        patches = patches.unfold(2, win, 1).unfold(3, win, 1).reshape(*hw, win * win)
        pb = patches[ys, xs]
        qo = torch.nanquantile(pb, torch.tensor([0.10, 0.90], dtype=pb.dtype), dim=-1)
        own_bg, own_fg = 1.0 / qo[0].clamp(min=1e-6), 1.0 / qo[1].clamp(min=1e-6)

        qo2 = torch.nanquantile(pb, torch.tensor([0.05, 0.95], dtype=pb.dtype), dim=-1)
        own2_bg, own2_fg = 1.0 / qo2[0].clamp(min=1e-6), 1.0 / qo2[1].clamp(min=1e-6)

        cands = {"own_p10_90": [own_fg, own_bg],
                 "mv_p10_90": [mv_fg, mv_bg],
                 "union_own_mv": [own_fg, own_bg, mv_fg, mv_bg],
                 "union_full": [own_fg, own_bg, own2_fg, own2_bg, mv_fg, mv_bg],
                 # control: how much coverage K near-duplicate candidates buy
                 "control_own_x3": [own_fg, own_bg, own_fg, own_bg, own_fg, own_bg]}
        for name, ds in cands.items():
            e = torch.stack([(torch.nan_to_num(d, nan=1e9) - ref_b).abs()
                             for d in ds]).min(dim=0).values
            acc[name]["gate"] += int(g_b.sum())
            acc[name]["cov"] += int((g_b & (e < tol_b)).sum())

    print(f"{args.scene.name}  ({len(eval_frames)} frames)  "
          f"selfcheck median |reproj-sensor| off-band: "
          f"{np.median(selfcheck):.3f} m" if selfcheck else "no selfcheck")
    for n in names:
        a = acc[n]
        print(f"  {n:<14}{a['cov'] / max(a['gate'], 1) * 100:>7.1f}%")


if __name__ == "__main__":
    with torch.no_grad():
        main()
