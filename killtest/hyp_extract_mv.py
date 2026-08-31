"""Production multi-view union hypothesis maps -> bimodal_hyp_mv/<stem>.npz.

Candidate set per band pixel (K=6, the union that won extractor_multiview.py):
  own-frame windowed inverse-depth p10/p90 (reused from bimodal_hyp npz),
  own-frame p05/p95,
  multi-view p10/p90 from sensor depth of 12 neighbors reprojected into this
  camera (amin/amax splats, 3x3 pooling).

Band and degenerate gates are copied unchanged from the existing bimodal_hyp
extraction so the only difference vs the bimodal arm is the candidate set.
npz keys: band u8, degenerate u8, d_hyp [H,W,6] float16 mm (0 = invalid).

  python hyp_extract_mv.py --scene <scene_dir> --capture long_capture
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kt_common import load_depth, valid_mask
from kt_config import CONFIG
from extractor_multiview import load_transforms, reproject, window_samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def own_pair(prior, ys, xs, lo, hi):
    v = valid_mask(prior)
    inv = torch.where(v, 1.0 / prior.clamp(min=1e-6), torch.nan)
    win, pad = CONFIG["hyp_window"], CONFIG["hyp_window"] // 2
    p = F.pad(inv.view(1, 1, *inv.shape), (pad,) * 4, value=torch.nan)
    p = p.unfold(2, win, 1).unfold(3, win, 1).reshape(*inv.shape, win * win)[ys, xs]
    q = torch.nanquantile(p, torch.tensor([lo, hi], dtype=p.dtype, device=p.device), dim=-1)
    return 1.0 / q[1].clamp(min=1e-6), 1.0 / q[0].clamp(min=1e-6)  # fg, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--src-name", default="bimodal_hyp")
    ap.add_argument("--out-name", default="bimodal_hyp_mv")
    ap.add_argument("--offsets", type=int, nargs="+",
                    default=[-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30])
    args = ap.parse_args()

    base = args.scene / "iphone" / args.capture
    src_dir, out_dir = base / args.src_name, base / args.out_name
    out_dir.mkdir(exist_ok=True)
    tfs = load_transforms(base)
    stems = sorted(p.stem for p in src_dir.glob("frame_*.npz"))
    stems = [s for s in stems if s in tfs]
    depths = {}  # lazy CPU cache

    def dep(stem):
        if stem not in depths:
            depths[stem] = load_depth(base / "depth" / f"{stem}.png")
        return depths[stem]

    print(f"{args.scene.name}/{args.capture}: {len(stems)} frames -> {out_dir}")
    for i, stem in enumerate(stems):
        dst = out_dir / f"{stem}.npz"
        if dst.exists():
            continue
        src = np.load(src_dir / f"{stem}.npz")
        band_np = src["band"] > 0
        prior = dep(stem)
        hw = prior.shape[:2]
        band = torch.from_numpy(band_np)
        ys, xs = torch.nonzero(band, as_tuple=True)
        if ys.numel() == 0:  # frame with no depth edges: nothing to extract
            H, W = prior.shape[:2]
            np.savez_compressed(
                dst, band=src["band"], degenerate=src["degenerate"],
                d_hyp=np.zeros((H, W, 6), dtype=np.float16))
            continue

        # own-frame pairs
        o10_fg, o10_bg = (torch.from_numpy(src["d_fg"].astype(np.float32)) * 1e-3,
                          torch.from_numpy(src["d_bg"].astype(np.float32)) * 1e-3)
        o10_fg, o10_bg = o10_fg[ys, xs], o10_bg[ys, xs]
        o05_fg, o05_bg = own_pair(prior, ys, xs, 0.05, 0.95)

        # multi-view pair
        K_i, c2w_i = tfs[stem]
        maps = []
        for off in args.offsets:
            j = i + off
            if not (0 <= j < len(stems)):
                continue
            nstem = stems[j]
            K_j, c2w_j = tfs[nstem]
            amin, amax = reproject(dep(nstem), K_j, c2w_j, K_i, c2w_i, hw)
            maps.extend([amin, amax])
        if maps:
            samp = window_samples(maps, ys, xs, r=1)
            inv = torch.where(torch.isfinite(samp) & (samp > 0), 1.0 / samp, torch.nan)
            q = torch.nanquantile(inv, torch.tensor([0.10, 0.90], dtype=inv.dtype), dim=-1)
            mv_bg, mv_fg = 1.0 / q[0].clamp(min=1e-6), 1.0 / q[1].clamp(min=1e-6)
        else:
            mv_bg = mv_fg = torch.full_like(o10_fg, torch.nan)

        H, W = hw
        d_hyp = torch.zeros(H, W, 6)
        for k, vals in enumerate([o10_fg, o10_bg, o05_fg, o05_bg, mv_fg, mv_bg]):
            m = torch.zeros(H, W)
            m[ys, xs] = torch.nan_to_num(vals.float(), nan=0.0, posinf=0.0)
            d_hyp[..., k] = m
        np.savez_compressed(
            dst,
            band=src["band"],
            degenerate=src["degenerate"],
            d_hyp=(d_hyp.numpy() * 1000.0).astype(np.float16),  # mm
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(stems)}")
    print("EXTRACT_MV_DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
