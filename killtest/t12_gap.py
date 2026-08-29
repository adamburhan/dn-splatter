"""T1.2 — Gap/noise ratio: prior-vs-GT error in the edge band vs off band.

Usage:
  python t12_gap.py --scene /path/to/room_datasets/koivu --out t12_koivu.json

Uses frames that have Faro reference depth. The prior is used exactly as the
loss sees it (sensor depth as shipped; mono depth as the repo's own
Zoe+scale/shift-aligned *_aligned.npy — no refitting).

Per depth source x edge definition x band width:
  - |prior - GT|: median/p90 in band vs off band, and their ratios
  - mechanism fingerprint: band error binned by local contrast d_bg/d_fg
    (7px window percentiles of the PRIOR), off-band error in the same bins
  - flying-pixel fraction: band pixels whose prior depth lies strictly between
    local fg and bg by margin (only where the local gap is meaningful)
  - median px distance from prior edges to nearest GT edge (edge placement offset)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kt_common import (
    EDGE_DEFS,
    dilate,
    load_depth,
    local_fg_bg,
    mushroom_frames,
    subsample,
    summarize,
    valid_mask,
)
from kt_config import CONFIG

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frames = mushroom_frames(args.scene, args.capture)
    frames = [f for f in frames if f["reference_depth"] is not None]
    frames = subsample(frames, CONFIG["max_frames_per_scene"])
    print(f"{args.scene.name}/{args.capture}: {len(frames)} frames with GT")

    bins = CONFIG["contrast_bins"]
    # accumulate raw error samples (subsampled per frame to bound memory)
    err_band = defaultdict(list)
    err_off = defaultdict(list)
    err_band_binned = defaultdict(lambda: defaultdict(list))
    err_off_binned = defaultdict(lambda: defaultdict(list))
    gap_binned = defaultdict(lambda: defaultdict(list))
    fly = defaultdict(lambda: {"fly": 0.0, "gapped": 0.0, "band": 0.0})
    edge_dist = defaultdict(list)

    for i, fr in enumerate(frames):
        gt = load_depth(fr["reference_depth"])
        hw = gt.shape[:2]
        vgt = valid_mask(gt)
        for source in ["sensor_depth", "mono_depth"]:
            if fr[source] is None:
                continue
            d = load_depth(fr[source], hw=hw)
            v = valid_mask(d) & vgt
            err = torch.abs(d - gt)
            d_fg, d_bg = local_fg_bg(d)
            gap = d_bg - d_fg
            contrast = d_bg / d_fg.clamp(min=1e-6)
            meaningful_gap = (gap > CONFIG["min_gap_abs"]) | (
                gap > CONFIG["min_gap_rel"] * d_fg
            )
            margin = CONFIG["intermediate_margin"] * gap
            intermediate = (d > d_fg + margin) & (d < d_bg - margin) & meaningful_gap

            for edef, fn in EDGE_DEFS.items():
                edge = fn(d)
                # GT-vs-prior edge placement offset (band-width independent)
                if distance_transform_edt is not None:
                    gt_edge = fn(gt)
                    if gt_edge.any() and edge.any():
                        dt = distance_transform_edt(~gt_edge.numpy())
                        edge_dist[(source, edef)].append(
                            float(np.median(dt[edge.numpy()]))
                        )
                for bw in CONFIG["band_dilations"]:
                    band = dilate(edge, bw) & v
                    off = v & ~band
                    key = (source, edef, bw)
                    err_band[key].append(err[band])
                    # subsample off-band (it is huge)
                    e_off = err[off]
                    if e_off.numel() > 50_000:
                        e_off = e_off[torch.randperm(e_off.numel())[:50_000]]
                    err_off[key].append(e_off)
                    # contrast-binned fingerprint
                    c_band, e_bandpx = contrast[band], err[band]
                    c_off, e_offpx = contrast[off], err[off]
                    g_band = gap[band]
                    for b in range(len(bins) - 1):
                        mb = (c_band >= bins[b]) & (c_band < bins[b + 1])
                        mo = (c_off >= bins[b]) & (c_off < bins[b + 1])
                        err_band_binned[key][b].append(e_bandpx[mb])
                        eo = e_offpx[mo]
                        if eo.numel() > 20_000:
                            eo = eo[torch.randperm(eo.numel())[:20_000]]
                        err_off_binned[key][b].append(eo)
                        gap_binned[key][b].append(g_band[mb])
                    # flying pixels
                    fly[key]["fly"] += float((intermediate & band).sum())
                    fly[key]["gapped"] += float((meaningful_gap & band).sum())
                    fly[key]["band"] += float(band.sum())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frames)}")

    result = {"scene": str(args.scene), "capture": args.capture, "n_frames": len(frames)}
    rows = []
    for key in sorted(err_band.keys()):
        source, edef, bw = key
        eb = torch.cat(err_band[key])
        eo = torch.cat(err_off[key])
        sb, so = summarize(eb), summarize(eo)
        row = {
            "source": source,
            "edge_def": edef,
            "band_px": bw,
            "band_err": sb,
            "off_err": so,
            "ratio_median": sb["median"] / so["median"] if so["median"] else None,
            "ratio_p90": sb["p90"] / so["p90"] if so["p90"] else None,
            "fly_frac_of_band": fly[key]["fly"] / max(fly[key]["band"], 1),
            "fly_frac_of_gapped_band": fly[key]["fly"] / max(fly[key]["gapped"], 1),
            "bins": [],
        }
        for b in range(len(bins) - 1):
            ebb = torch.cat(err_band_binned[key][b])
            eob = torch.cat(err_off_binned[key][b])
            gbb = torch.cat(gap_binned[key][b])
            row["bins"].append(
                {
                    "contrast": [bins[b], bins[b + 1]],
                    "band_err_median": summarize(ebb)["median"],
                    "off_err_median": summarize(eob)["median"],
                    "band_gap_median": summarize(gbb)["median"],
                    "n_band": int(ebb.numel()),
                }
            )
        rows.append(row)
    result["rows"] = rows
    result["edge_offset_px_median"] = {
        f"{s}/{e}": float(np.median(v)) for (s, e), v in edge_dist.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    for r in rows:
        print(
            f"\n{r['source']} {r['edge_def']} band={r['band_px']}px: "
            f"band med={r['band_err']['median']:.4f} p90={r['band_err']['p90']:.4f} | "
            f"off med={r['off_err']['median']:.4f} p90={r['off_err']['p90']:.4f} | "
            f"ratio med={r['ratio_median']:.2f} p90={r['ratio_p90']:.2f} | "
            f"fly={r['fly_frac_of_band']:.3f} ({r['fly_frac_of_gapped_band']:.3f} of gapped)"
        )
        for b in r["bins"]:
            print(
                f"    c[{b['contrast'][0]:.2f},{b['contrast'][1]:.2f}) "
                f"band={b['band_err_median']} off={b['off_err_median']} "
                f"gap={b['band_gap_median']} n={b['n_band']}"
            )
    print("\nedge offset (px, prior->GT):", result["edge_offset_px_median"])


if __name__ == "__main__":
    with torch.no_grad():
        main()
