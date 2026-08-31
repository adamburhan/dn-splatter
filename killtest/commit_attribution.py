"""Why do wrong commitments happen: bad optimization, or bad hypotheses?

A wrong-commit pixel (t13_eval.py) is one whose rendered depth snapped to the
OTHER local GT surface. Two mutually exclusive causes:

  OPT   the prior's hypothesis set contained the correct surface (GT's own
        value is within tol of d_fg or d_bg) -> the loss offered the right
        target and the optimizer chose the other mode. A softer mixture can
        fix this.
  EXTR  the correct surface is in neither hypothesis -> the loss never offered
        it. No temperature can fix this; the extractor (or the prior) is the
        bound.

Also reports the same split over ALL gated band pixels, so the wrong-commit
rate can be compared against the rate at which hypotheses are simply unusable.

  python commit_attribution.py --render-dir <renders/scene_arm> --scene <scene>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kt_common import EDGE_DEFS, dilate, load_depth, local_fg_bg, valid_mask
from kt_config import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--hyp-name", default="bimodal_hyp")
    ap.add_argument("--band-px", type=int, default=2)
    ap.add_argument("--edge-def", default="grad_logd")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    base = args.scene / "iphone" / args.capture
    hyp_dir = base / args.hyp_name
    pred_depth_dir = args.render_dir / "eval" / "pred" / "depth" / "raw"
    prefix = f"{args.capture}_"

    stems = sorted(p.stem for p in pred_depth_dir.glob(f"{prefix}*.npy"))
    n = dict(gated=0, wc=0, wc_opt=0, wc_extr=0, hyp_ok=0)
    for s in stems:
        name = s[len(prefix):]
        prior_p, ref_p = base / "depth" / f"{name}.png", base / "reference_depth" / f"{name}.png"
        hyp_p = hyp_dir / f"{name}.npz"
        if not (prior_p.exists() and ref_p.exists() and hyp_p.exists()):
            continue
        pdepth = torch.from_numpy(
            np.load(pred_depth_dir / f"{s}.npy").squeeze()).float() * 1e-3
        hw = pdepth.shape[:2]
        prior = load_depth(prior_p, hw=hw)
        ref = load_depth(ref_p, hw=hw)
        hyp = np.load(hyp_p)

        def hmap(k):
            a = hyp[k].astype(np.float32)
            if a.shape[:2] != hw:
                import cv2
                a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(a)

        h_fg, h_bg = hmap("d_fg") * 1e-3, hmap("d_bg") * 1e-3  # mm -> m
        degen = hmap("degenerate") > 0.5

        gt_fg, gt_bg = local_fg_bg(ref)
        gt_gap = gt_bg - gt_fg
        gt_meaningful = (
            (gt_gap > CONFIG["min_gap_abs"]) | (gt_gap > CONFIG["min_gap_rel"] * gt_fg)
        ) & torch.isfinite(gt_gap)
        tol = CONFIG["commit_tolerance"] * gt_gap
        own_is_fg = (ref - gt_fg).abs() < (ref - gt_bg).abs()
        gt_other = torch.where(own_is_fg, gt_bg, gt_fg)
        wrong = ((pdepth - gt_other).abs() < tol) & ((pdepth - ref).abs() > tol)

        band = dilate(EDGE_DEFS[args.edge_def](prior), args.band_px)
        gated = band & valid_mask(ref) & gt_meaningful & ~degen & (h_fg > 0) & (h_bg > 0)

        # was the correct surface (GT's own depth) available among the prior's
        # two hypotheses at all?
        avail = torch.minimum((ref - h_fg).abs(), (ref - h_bg).abs()) < tol
        n["gated"] += int(gated.sum())
        n["hyp_ok"] += int((gated & avail).sum())
        n["wc"] += int((gated & wrong).sum())
        n["wc_opt"] += int((gated & wrong & avail).sum())
        n["wc_extr"] += int((gated & wrong & ~avail).sum())

    wc = max(n["wc"], 1)
    out = {
        "render_dir": str(args.render_dir),
        "n_frames": len(stems),
        "gated_px": n["gated"],
        "hyp_available_rate": n["hyp_ok"] / max(n["gated"], 1),
        "wrong_commit_rate": n["wc"] / max(n["gated"], 1),
        "wc_share_optimization": n["wc_opt"] / wc,
        "wc_share_extractor": n["wc_extr"] / wc,
    }
    print(f"{args.render_dir.name}: frames {out['n_frames']}  gated {n['gated']:,}\n"
          f"  hypotheses contain the true surface: {out['hyp_available_rate'] * 100:.1f}%\n"
          f"  wrong-commit rate:                  {out['wrong_commit_rate'] * 100:.1f}%\n"
          f"    -> OPT  (right target offered):   {out['wc_share_optimization'] * 100:.1f}%\n"
          f"    -> EXTR (never offered):          {out['wc_share_extractor'] * 100:.1f}%")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
