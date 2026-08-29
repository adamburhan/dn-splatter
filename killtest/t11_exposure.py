"""T1.1 — Exposure: fraction of supervised pixels inside the prior's depth-edge band.

Usage:
  python t11_exposure.py --scene /path/to/room_datasets/koivu --out t11_koivu.json

Reports, per depth source (sensor / mono) x edge definition x band width:
  - raw exposure: band_valid_px / valid_px
  - weighted exposure: same but pixels weighted by the EdgeAwareLogL1 RGB weight
    exp(-|grad RGB|) (how much supervision actually lands in the band under the
    repo's default loss).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from kt_common import (
    EDGE_DEFS,
    dilate,
    edge_aware_weight,
    load_depth,
    load_rgb,
    mushroom_frames,
    subsample,
    valid_mask,
)
from kt_config import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frames = mushroom_frames(args.scene, args.capture)
    frames = [f for f in frames if f["sensor_depth"] is not None]
    frames = subsample(frames, CONFIG["max_frames_per_scene"])
    print(f"{args.scene.name}/{args.capture}: {len(frames)} frames")

    acc = defaultdict(lambda: {"band": 0.0, "valid": 0.0, "band_w": 0.0, "valid_w": 0.0})

    for i, fr in enumerate(frames):
        rgb = load_rgb(fr["image"])
        hw = rgb.shape[:2]
        w_ea = edge_aware_weight(rgb)
        for source in ["sensor_depth", "mono_depth"]:
            if fr[source] is None:
                continue
            d = load_depth(fr[source], hw=hw)
            v = valid_mask(d)
            for edef, fn in EDGE_DEFS.items():
                edge = fn(d)
                for bw in CONFIG["band_dilations"]:
                    band = dilate(edge, bw) & v
                    key = (source, edef, bw)
                    acc[key]["band"] += float(band.sum())
                    acc[key]["valid"] += float(v.sum())
                    acc[key]["band_w"] += float(w_ea[band].sum())
                    acc[key]["valid_w"] += float(w_ea[v].sum())
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(frames)}")

    result = {"scene": str(args.scene), "capture": args.capture, "n_frames": len(frames)}
    rows = []
    for (source, edef, bw), a in sorted(acc.items()):
        if a["valid"] == 0:
            continue
        rows.append(
            {
                "source": source,
                "edge_def": edef,
                "band_px": bw,
                "exposure": a["band"] / a["valid"],
                "exposure_edgeaware_weighted": a["band_w"] / a["valid_w"],
            }
        )
    result["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n{'source':<14}{'edge_def':<14}{'band':<6}{'exposure':<10}{'weighted'}")
    for r in rows:
        print(
            f"{r['source']:<14}{r['edge_def']:<14}{r['band_px']:<6}"
            f"{r['exposure']:<10.4f}{r['exposure_edgeaware_weighted']:.4f}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
