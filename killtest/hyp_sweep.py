"""WIN sweep for hypothesis extraction, scored against Faro GT on the tuning
scene (koivu) only. For each candidate window size, over active-band pixels
(band & ~degenerate & valid GT):

  coverage  fraction whose GT depth is within tol of EITHER hypothesis
            (tol = max(5cm, 5% of GT)) — can the bimodal loss reach the truth?
  min_err   median of min(|GT - d_fg|, |GT - d_bg|) — how close is the closer
            hypothesis?
  unimodal baseline: same metrics with the single prior value (what the
            unimodal loss offers) for reference.

Usage: python hyp_sweep.py --scene .../koivu
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from hyp_extract import extract_one
from kt_common import load_depth, mushroom_frames, subsample, valid_mask
from kt_config import CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--source", default="sensor")
    ap.add_argument("--n-frames", type=int, default=40)
    args = ap.parse_args()

    frames = [
        f
        for f in mushroom_frames(args.scene, args.capture)
        if f[f"{args.source}_depth"] and f["reference_depth"]
    ]
    frames = subsample(frames, args.n_frames)
    print(f"{args.scene.name}: {len(frames)} GT frames; wins {CONFIG['hyp_window_sweep']}")

    for win in CONFIG["hyp_window_sweep"]:
        cov, merr, ucov, uerr, npx = [], [], [], [], 0
        for fr in frames:
            gt = load_depth(fr["reference_depth"]).to(DEVICE)
            d = load_depth(fr[f"{args.source}_depth"], hw=gt.shape).to(DEVICE)
            maps = extract_one(d, args.source, win)
            active = maps["band"] & ~maps["degenerate"] & valid_mask(gt) & valid_mask(d)
            if not active.any():
                continue
            g = gt[active]
            tol = torch.maximum(torch.full_like(g, 0.05), 0.05 * g)
            e_bi = torch.minimum(
                (g - maps["d_fg"][active]).abs(), (g - maps["d_bg"][active]).abs()
            )
            e_uni = (g - d[active]).abs()
            cov.append((e_bi < tol).float().sum().item())
            ucov.append((e_uni < tol).float().sum().item())
            merr.append(e_bi.cpu())
            uerr.append(e_uni.cpu())
            npx += int(active.sum())
        merr = torch.cat(merr)
        uerr = torch.cat(uerr)
        print(
            f"win={win:>2}  n={npx/1e6:.2f}M  "
            f"bimodal: cov {sum(cov)/npx*100:.1f}%  min_err med {merr.median():.3f}  "
            f"| unimodal: cov {sum(ucov)/npx*100:.1f}%  err med {uerr.median():.3f}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
