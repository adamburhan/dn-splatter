"""T1.3 — Artifact attribution: band-sliced eval of trained models.

Consumes the output of dn_splatter/scripts/render_model.py --split eval:
  <render_dir>/eval/pred/rgb/<stem>.png        rendered rgb
  <render_dir>/eval/gt/rgb/<stem>.png          gt rgb
  <render_dir>/eval/pred/depth/raw/<stem>.npy  rendered depth (scene units = m)

Band masks are computed from the SUPERVISING prior (sensor depth) of the scene,
GT depth error uses the Faro reference depth. Metrics per edge-def x band-width:
PSNR/SSIM/LPIPS restricted to band vs off-band, rendered-depth error vs faro
GT in band vs off-band.

Usage:
  python t13_eval.py --render-dir <dir> --scene <scene_dir> --out out.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kt_common import (
    EDGE_DEFS,
    dilate,
    load_depth,
    load_rgb,
    local_fg_bg,
    summarize,
    valid_mask,
)
from kt_config import CONFIG

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


def masked_psnr(pred, gt, mask):
    if mask.sum() == 0:
        return None
    mse = ((pred - gt) ** 2).mean(dim=-1)[mask].mean()
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def ssim_map(pred, gt, kernel=11):
    """Per-pixel SSIM (uniform window, same shape as input), [H,W]."""
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)
    pad = kernel // 2
    mu_p = F.avg_pool2d(p, kernel, 1, pad)
    mu_g = F.avg_pool2d(g, kernel, 1, pad)
    sig_p = F.avg_pool2d(p * p, kernel, 1, pad) - mu_p**2
    sig_g = F.avg_pool2d(g * g, kernel, 1, pad) - mu_g**2
    sig_pg = F.avg_pool2d(p * g, kernel, 1, pad) - mu_p * mu_g
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) / (
        (mu_p**2 + mu_g**2 + C1) * (sig_p + sig_g + C2)
    )
    return ssim.mean(dim=1).squeeze(0)  # avg over channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = args.scene / "iphone" / args.capture
    pred_rgb_dir = args.render_dir / "eval" / "pred" / "rgb"
    gt_rgb_dir = args.render_dir / "eval" / "gt" / "rgb"
    pred_depth_dir = args.render_dir / "eval" / "pred" / "depth" / "raw"

    stems = sorted(p.stem for p in pred_rgb_dir.glob("*.png"))
    # rendered stems are '<capture>_frame_XXXXX'; keep this capture's frames
    # that have a prior (+ faro GT when present)
    frames = []
    prefix = f"{args.capture}_"
    for s in stems:
        if not s.startswith(prefix):
            continue
        name = s[len(prefix):]
        prior = base / "depth" / f"{name}.png"
        ref = base / "reference_depth" / f"{name}.png"
        if prior.exists():
            frames.append({"stem": s, "prior": prior, "ref": ref if ref.exists() else None})
    print(f"{args.render_dir}: {len(frames)} eval frames matched to {args.capture}")

    lp = lpips_lib.LPIPS(net="alex", spatial=True) if lpips_lib else None

    psnr_band = defaultdict(list)
    psnr_off = defaultdict(list)
    ssim_band = defaultdict(list)
    ssim_off = defaultdict(list)
    lp_band = defaultdict(list)
    lp_off = defaultdict(list)
    derr_band = defaultdict(list)
    derr_off = defaultdict(list)
    commit_n = defaultdict(int)
    wrong_commit = defaultdict(int)
    phantom = defaultdict(int)
    full = {"psnr": [], "ssim": [], "lpips": []}

    for fr in frames:
        pred = load_rgb(pred_rgb_dir / f"{fr['stem']}.png")
        gt = load_rgb(gt_rgb_dir / f"{fr['stem']}.png")
        hw = gt.shape[:2]
        if pred.shape != gt.shape:
            continue
        prior = load_depth(fr["prior"], hw=hw)
        # save_depth writes npy in mm (utils.py save_depth divides by SCALE_FACTOR)
        pdepth = np.load(pred_depth_dir / f"{fr['stem']}.npy") * 1e-3
        pdepth = torch.from_numpy(pdepth.squeeze()).float()

        smap = ssim_map(pred, gt)
        if lp is not None:
            with torch.no_grad():
                lmap = lp(
                    pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1,
                    gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1,
                ).squeeze()
        else:
            lmap = None

        full["psnr"].append(masked_psnr(pred, gt, torch.ones(hw, dtype=torch.bool)))
        full["ssim"].append(float(smap.mean()))
        if lmap is not None:
            full["lpips"].append(float(lmap.mean()))

        ref = load_depth(fr["ref"], hw=hw) if fr["ref"] else None
        vref = valid_mask(ref) if ref is not None else None
        if ref is not None:
            # local GT surfaces (7px window, as in T1.2) for commitment metrics
            gt_fg, gt_bg = local_fg_bg(ref)
            gt_gap = gt_bg - gt_fg
            gt_meaningful = (
                (gt_gap > CONFIG["min_gap_abs"])
                | (gt_gap > CONFIG["min_gap_rel"] * gt_fg)
            ) & torch.isfinite(gt_gap)
            tol = CONFIG["commit_tolerance"] * gt_gap
            own_is_fg = (ref - gt_fg).abs() < (ref - gt_bg).abs()
            gt_other = torch.where(own_is_fg, gt_bg, gt_fg)
            wrong_commit_map = ((pdepth - gt_other).abs() < tol) & (
                (pdepth - ref).abs() > tol
            )
            phantom_map = (pdepth > gt_fg + tol) & (pdepth < gt_bg - tol)

        for edef, fn in EDGE_DEFS.items():
            edge = fn(prior)
            for bw in CONFIG["band_dilations"]:
                band = dilate(edge, bw)
                off = ~band
                key = (edef, bw)
                psnr_band[key].append(masked_psnr(pred, gt, band))
                psnr_off[key].append(masked_psnr(pred, gt, off))
                ssim_band[key].append(float(smap[band].mean()) if band.any() else None)
                ssim_off[key].append(float(smap[off].mean()))
                if lmap is not None:
                    lp_band[key].append(float(lmap[band].mean()) if band.any() else None)
                    lp_off[key].append(float(lmap[off].mean()))
                if ref is not None:
                    vb = band & vref
                    vo = off & vref
                    derr_band[key].append(torch.abs(pdepth - ref)[vb])
                    derr_off[key].append(torch.abs(pdepth - ref)[vo])
                    gated = vb & gt_meaningful
                    commit_n[key] += int(gated.sum())
                    wrong_commit[key] += int((wrong_commit_map & gated).sum())
                    phantom[key] += int((phantom_map & gated).sum())

    def agg(d, key):
        vals = [v for v in d[key] if v is not None]
        return float(np.mean(vals)) if vals else None

    result = {
        "render_dir": str(args.render_dir),
        "scene": str(args.scene),
        "n_frames": len(frames),
        "full_image": {k: (float(np.mean(v)) if v else None) for k, v in full.items()},
        "rows": [],
    }
    for key in sorted(psnr_band.keys()):
        edef, bw = key
        row = {
            "edge_def": edef,
            "band_px": bw,
            "psnr_band": agg(psnr_band, key),
            "psnr_off": agg(psnr_off, key),
            "ssim_band": agg(ssim_band, key),
            "ssim_off": agg(ssim_off, key),
            "lpips_band": agg(lp_band, key),
            "lpips_off": agg(lp_off, key),
        }
        if derr_band[key]:
            row["depth_err_band"] = summarize(torch.cat(derr_band[key]))
            row["depth_err_off"] = summarize(torch.cat(derr_off[key]))
        if commit_n[key] > 0:
            row["wrong_commit_rate"] = wrong_commit[key] / commit_n[key]
            row["phantom_middle_rate"] = phantom[key] / commit_n[key]
            row["commit_n"] = commit_n[key]
        result["rows"].append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["full_image"]))
    for r in result["rows"]:
        line = (
            f"{r['edge_def']:<16} band={r['band_px']}px  "
            f"PSNR {r['psnr_band']:.2f}/{r['psnr_off']:.2f}  "
            f"SSIM {r['ssim_band']:.3f}/{r['ssim_off']:.3f}  "
        )
        if r.get("lpips_band") is not None:
            line += f"LPIPS {r['lpips_band']:.3f}/{r['lpips_off']:.3f}  "
        if "depth_err_band" in r:
            line += (
                f"derr med {r['depth_err_band']['median']:.3f}/"
                f"{r['depth_err_off']['median']:.3f}  "
            )
        if "wrong_commit_rate" in r:
            line += (
                f"wrongcommit {r['wrong_commit_rate']:.3f} "
                f"phantom {r['phantom_middle_rate']:.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()
