"""Extractor kill test: candidate coverage vs Faro, no training required.

For every gated band pixel (band from the sensor prior, Faro-valid, meaningful
local GT gap), a hypothesis set {d_fg, d_bg} is scored by

    e_candidate = min(|d_fg - d*|, |d_bg - d*|)          (d* = Faro depth)
    covered     = e_candidate < tol,   tol = commit_tolerance * local GT gap

Extractors compared:
  current    cached bimodal_hyp npz (21x21 p10/p90 of inverse depth)
  two_sided  edge-normal two-sided sampling (kt_common.two_sided_hypotheses)
  oracle     local_fg_bg of the Faro GT itself (upper bound)

A pixel with an invalid/degenerate hypothesis set counts as NOT covered (the
loss would fall back to unimodal there), so coverage is comparable across
extractors with different validity gates. Median min-error is over valid-
hypothesis pixels only (validity rate reported alongside).

  python extractor_killtest.py --scene <scene_dir> [--max-frames 50] --out out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kt_common import (
    EDGE_DEFS,
    dilate,
    kmeans2_hypotheses,
    load_depth,
    local_fg_bg,
    mushroom_frames,
    subsample,
    two_sided_hypotheses,
    valid_mask,
)
from kt_config import CONFIG


def score(d_fg, d_bg, ok, ref, tol, gate, acc, name):
    e = torch.minimum((d_fg - ref).abs(), (d_bg - ref).abs())
    g_ok = gate & ok
    acc[name]["n_gate"] += int(gate.sum())
    acc[name]["n_valid"] += int(g_ok.sum())
    acc[name]["n_cov"] += int((g_ok & (e < tol)).sum())
    acc[name]["errs"].append(e[g_ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--hyp-name", default="bimodal_hyp")
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--r-min", type=int, default=3)
    ap.add_argument("--r-max", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--extractors", nargs="+",
                    default=["current", "two_sided", "kmeans2", "oracle"])
    args = ap.parse_args()

    base = args.scene / "iphone" / args.capture
    hyp_dir = base / args.hyp_name
    frames = [f for f in mushroom_frames(args.scene, args.capture)
              if f["sensor_depth"] and f["reference_depth"]]
    frames = subsample(frames, args.max_frames)

    names = args.extractors
    acc = {n: {"n_gate": 0, "n_valid": 0, "n_cov": 0, "errs": []} for n in names}

    for fr in frames:
        prior = load_depth(fr["sensor_depth"])
        hw = prior.shape[:2]
        ref = load_depth(fr["reference_depth"], hw=hw)
        vref = valid_mask(ref)

        band = dilate(EDGE_DEFS["grad_logd"](prior), CONFIG["hyp_band_dilation"])
        gt_fg, gt_bg = local_fg_bg(ref)
        gt_gap = gt_bg - gt_fg
        gt_meaningful = (
            (gt_gap > CONFIG["min_gap_abs"]) | (gt_gap > CONFIG["min_gap_rel"] * gt_fg)
        ) & torch.isfinite(gt_gap)
        tol = CONFIG["commit_tolerance"] * gt_gap
        gate = band & vref & gt_meaningful

        # current: cached npz (mm)
        hyp_p = hyp_dir / f"{fr['stem']}.npz"
        if "current" in names and hyp_p.exists():
            hyp = np.load(hyp_p)
            h_fg = torch.from_numpy(hyp["d_fg"].astype(np.float32)) * 1e-3
            h_bg = torch.from_numpy(hyp["d_bg"].astype(np.float32)) * 1e-3
            degen = torch.from_numpy(hyp["degenerate"]) > 0.5
            ok = ~degen & (h_fg > 0) & (h_bg > 0)
            score(h_fg, h_bg, ok, ref, tol, gate, acc, "current")

        # two-sided edge-normal
        if "two_sided" in names:
            t_fg, t_bg, t_ok = two_sided_hypotheses(prior, band, args.r_min, args.r_max)
            t_gap = t_bg - t_fg
            t_ok = t_ok & ((t_gap > CONFIG["min_gap_abs"])
                           | (t_gap > CONFIG["min_gap_rel"] * t_fg))
            score(torch.nan_to_num(t_fg), torch.nan_to_num(t_bg), t_ok,
                  ref, tol, gate, acc, "two_sided")

        # 2-means in the percentile extractor's window (modes, not quantiles)
        if "kmeans2" in names:
            k_fg, k_bg, k_ok = kmeans2_hypotheses(prior, band)
            k_gap = k_bg - k_fg
            k_ok = k_ok & ((k_gap > CONFIG["min_gap_abs"])
                           | (k_gap > CONFIG["min_gap_rel"] * k_fg))
            score(torch.nan_to_num(k_fg), torch.nan_to_num(k_bg), k_ok,
                  ref, tol, gate, acc, "kmeans2")

        # oracle: GT's own local surfaces
        if "oracle" in names:
            score(gt_fg, gt_bg, torch.isfinite(gt_gap), ref, tol, gate, acc, "oracle")

    out = {"scene": args.scene.name, "n_frames": len(frames), "extractors": {}}
    print(f"{args.scene.name}  ({len(frames)} frames)")
    print(f"{'extractor':<12}{'coverage':>9}{'valid':>8}{'min-err med':>12}{'p90':>8}")
    for n in names:
        a = acc[n]
        if a["n_gate"] == 0:
            continue
        errs = torch.cat(a["errs"]) if a["errs"] else torch.zeros(0)
        row = {
            "coverage": a["n_cov"] / a["n_gate"],
            "validity": a["n_valid"] / a["n_gate"],
            "min_err_median": float(errs.median()) if errs.numel() else None,
            "min_err_p90": float(torch.quantile(errs, 0.9)) if errs.numel() else None,
            "n_gate": a["n_gate"],
        }
        out["extractors"][n] = row
        print(f"{n:<12}{row['coverage'] * 100:>8.1f}%{row['validity'] * 100:>7.1f}%"
              f"{row['min_err_median']:>12.3f}{row['min_err_p90']:>8.3f}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
