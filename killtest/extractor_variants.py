"""Single-frame extractor variants, scored like extractor_killtest.py.

Variants: percentile pairs / window sizes of the windowed-inverse-percentile
extractor, two-sided, and UNIONS of candidate sets (min-error over all
candidates -> what a K-hypothesis loss would see).

  python extractor_variants.py --scene <scene_dir> [--max-frames 20]
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from kt_common import (
    EDGE_DEFS,
    dilate,
    load_depth,
    local_fg_bg,
    mushroom_frames,
    subsample,
    two_sided_hypotheses,
    valid_mask,
)
from kt_config import CONFIG


def band_patches(depth, ys, xs, win):
    v = valid_mask(depth)
    inv = torch.where(v, 1.0 / depth.clamp(min=1e-6), torch.nan)
    pad = win // 2
    p = F.pad(inv.view(1, 1, *inv.shape), (pad,) * 4, value=torch.nan)
    p = p.unfold(2, win, 1).unfold(3, win, 1).reshape(*inv.shape, win * win)
    return p[ys, xs]  # [N, win*win] inverse depth


def quantile_pair(patches, lo, hi):
    q = torch.nanquantile(patches, torch.tensor([lo, hi], dtype=patches.dtype), dim=-1)
    d_bg = 1.0 / q[0].clamp(min=1e-6)
    d_fg = 1.0 / q[1].clamp(min=1e-6)
    return d_fg, d_bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--max-frames", type=int, default=20)
    args = ap.parse_args()

    frames = [f for f in mushroom_frames(args.scene, args.capture)
              if f["sensor_depth"] and f["reference_depth"]]
    frames = subsample(frames, args.max_frames)

    variants = ["p10_90_w21", "p05_95_w21", "p20_80_w21", "p10_90_w31",
                "two_sided", "union_p_2s", "union_all"]
    acc = {n: {"gate": 0, "cov": 0} for n in variants}

    for fr in frames:
        prior = load_depth(fr["sensor_depth"])
        ref = load_depth(fr["reference_depth"], hw=prior.shape[:2])
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

        cands = {}
        p21 = band_patches(prior, ys, xs, 21)
        p31 = band_patches(prior, ys, xs, 31)
        cands["p10_90_w21"] = quantile_pair(p21, 0.10, 0.90)
        cands["p05_95_w21"] = quantile_pair(p21, 0.05, 0.95)
        cands["p20_80_w21"] = quantile_pair(p21, 0.20, 0.80)
        cands["p10_90_w31"] = quantile_pair(p31, 0.10, 0.90)
        t_fg, t_bg, _ = two_sided_hypotheses(prior, band)
        cands["two_sided"] = (t_fg[ys, xs], t_bg[ys, xs])
        cands["union_p_2s"] = cands["p10_90_w21"] + cands["two_sided"]
        cands["union_all"] = (cands["p10_90_w21"] + cands["p05_95_w21"]
                              + cands["p10_90_w31"] + cands["two_sided"])

        for name, ds in cands.items():
            e = torch.stack([(torch.nan_to_num(d, nan=1e9) - ref_b).abs()
                             for d in ds]).min(dim=0).values
            acc[name]["gate"] += int(g_b.sum())
            acc[name]["cov"] += int((g_b & (e < tol_b)).sum())

    print(f"{args.scene.name}  ({len(frames)} frames)")
    for n in variants:
        a = acc[n]
        print(f"  {n:<14}{a['cov'] / max(a['gate'], 1) * 100:>7.1f}%")


if __name__ == "__main__":
    with torch.no_grad():
        main()
