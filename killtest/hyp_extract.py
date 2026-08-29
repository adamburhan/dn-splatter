"""Offline extraction of two-surface hypothesis maps for the bimodal depth loss.

For each prior depth map produce and cache, at the prior's native resolution:
  band        edge-band mask (uint8), per-source definition + dilation
  d_fg, d_bg  near/far surface hypotheses (float32, mm) = windowed percentiles
              of the prior's INVERSE depth (robust to flying pixels)
  degenerate  uint8, band pixels whose local gap is not meaningful -> the loss
              falls back to unimodal there

Cache location: <capture>/<out-name>/<stem>.npz  (sibling of the depth dir, so
the dataloader can find it from the depth filename with no dataparser changes),
plus <out-name>/extract_config.json recording every threshold used.

Usage:
  python hyp_extract.py --scene <scene_dir> --capture long_capture --source sensor
  python hyp_extract.py --selftest
  python hyp_extract.py --scene ... --viz-only --n-viz 5 --viz-out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kt_common import EDGE_DEFS, dilate, load_depth, load_rgb, mushroom_frames, valid_mask
from kt_config import CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def windowed_inv_percentiles(depth: torch.Tensor, win: int):
    """d_fg (near), d_bg (far) from p{bg}/p{fg} percentiles of inverse depth
    in a win x win window. depth [H,W] meters on any device. Returns two [H,W]
    tensors (nan where the window has no valid pixels)."""
    v = valid_mask(depth)
    inv = torch.where(v, 1.0 / depth.clamp(min=1e-6), torch.nan)
    pad = win // 2
    patches = F.pad(inv.view(1, 1, *inv.shape), (pad, pad, pad, pad), value=torch.nan)
    patches = patches.unfold(2, win, 1).unfold(3, win, 1).reshape(*inv.shape, win * win)
    # high inverse-depth percentile = near surface, low = far surface
    q = torch.tensor(
        [CONFIG["hyp_fg_percentile"] / 100.0, CONFIG["hyp_bg_percentile"] / 100.0],
        device=depth.device,
        dtype=depth.dtype,
    )
    lo_hi = torch.nanquantile(patches, q, dim=-1)  # [2,H,W]: p10(inv), p90(inv)
    d_bg = 1.0 / lo_hi[0].clamp(min=1e-6)  # far
    d_fg = 1.0 / lo_hi[1].clamp(min=1e-6)  # near
    return d_fg, d_bg


def extract_one(depth: torch.Tensor, source: str, win: int):
    """depth [H,W] meters -> dict of maps (band, d_fg, d_bg, degenerate)."""
    band_def = CONFIG["hyp_band_def"][source]
    edge = EDGE_DEFS[band_def](depth)
    band = dilate(edge, CONFIG["hyp_band_dilation"])
    d_fg, d_bg = windowed_inv_percentiles(depth, win)
    gap = d_bg - d_fg
    meaningful = (gap > CONFIG["min_gap_abs"]) | (gap > CONFIG["min_gap_rel"] * d_fg)
    degenerate = ~(meaningful & torch.isfinite(gap))
    return {
        "band": band,
        "d_fg": torch.nan_to_num(d_fg, nan=0.0),
        "d_bg": torch.nan_to_num(d_bg, nan=0.0),
        "degenerate": degenerate,
    }


def selftest():
    """Synthetic step edge (2m/4m plateaus) with flying pixels injected across
    the boundary: extraction must recover both plateau depths in the band."""
    H, W, win = 64, 64, CONFIG["hyp_window"]
    d = torch.full((H, W), 2.0)
    d[:, 32:] = 4.0
    # flying pixels: a 3px column of intermediate values straddling the edge
    torch.manual_seed(0)
    d[:, 31:34] = 2.0 + 2.0 * torch.rand(H, 3)
    maps = extract_one(d, "sensor", win)
    band = maps["band"] & ~maps["degenerate"]
    assert band.any(), "band empty on a step edge"
    # check hypotheses at band pixels near the edge straddle the plateaus
    fg = maps["d_fg"][band]
    bg = maps["d_bg"][band]
    assert (fg - 2.0).abs().median() < 0.15, f"d_fg off: {fg.median()}"
    assert (bg - 4.0).abs().median() < 0.15, f"d_bg off: {bg.median()}"
    # flying pixels must NOT contaminate hypotheses (p10/p90 vs min/max)
    assert fg.min() > 1.5 and bg.max() < 4.5
    # degenerate gate: far from the edge, windows are flat -> degenerate
    flat = maps["degenerate"][:, :10]
    assert flat.all(), "flat region not gated degenerate"
    print("SELFTEST_OK  d_fg med %.3f  d_bg med %.3f  band px %d" % (
        fg.median(), bg.median(), int(band.sum())))


def viz(scene: Path, capture: str, source: str, win: int, out: Path, n: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [f for f in mushroom_frames(scene, capture) if f[f"{source}_depth"]]
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out.mkdir(parents=True, exist_ok=True)
    for i in idx:
        fr = frames[i]
        rgb = load_rgb(fr["image"])
        depth = load_depth(fr[f"{source}_depth"], hw=rgb.shape[:2]).to(DEVICE)
        maps = extract_one(depth, source, win)
        band = (maps["band"] & ~maps["degenerate"]).cpu().numpy()
        over = rgb.numpy().copy()
        over[band] = [1.0, 0.1, 0.1]
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
        ax[0].imshow(over); ax[0].set_title("rgb + active band")
        dm = depth.cpu().numpy()
        vmax = np.percentile(dm[dm > 0], 98)
        ax[1].imshow(dm, cmap="turbo", vmin=0, vmax=vmax); ax[1].set_title("prior depth")
        ax[2].imshow(maps["d_fg"].cpu().numpy(), cmap="turbo", vmin=0, vmax=vmax)
        ax[2].set_title("d_fg (near hyp)")
        ax[3].imshow(maps["d_bg"].cpu().numpy(), cmap="turbo", vmin=0, vmax=vmax)
        ax[3].set_title("d_bg (far hyp)")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"hyp_{fr['stem']}.jpg", dpi=110)
        plt.close(fig)
        print("saved", out / f"hyp_{fr['stem']}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--source", choices=["sensor", "mono"], default="sensor")
    ap.add_argument("--win", type=int, default=None)
    ap.add_argument("--out-name", default="bimodal_hyp")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--viz-only", action="store_true")
    ap.add_argument("--viz-out", type=Path, default=None)
    ap.add_argument("--n-viz", type=int, default=5)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    win = args.win or CONFIG["hyp_window"]
    if args.viz_only:
        viz(args.scene, args.capture, args.source, win,
            args.viz_out or Path("hyp_viz"), args.n_viz)
        return

    base = args.scene / "iphone" / args.capture
    out_dir = base / args.out_name
    out_dir.mkdir(exist_ok=True)
    cfg_dump = {
        "source": args.source, "win": win,
        "band_def": CONFIG["hyp_band_def"][args.source],
        "band_dilation": CONFIG["hyp_band_dilation"],
        "fg_percentile": CONFIG["hyp_fg_percentile"],
        "bg_percentile": CONFIG["hyp_bg_percentile"],
        "min_gap_abs": CONFIG["min_gap_abs"], "min_gap_rel": CONFIG["min_gap_rel"],
        "depth_tolerance": CONFIG["depth_tolerance"], "max_depth": CONFIG["max_depth"],
    }
    (out_dir / "extract_config.json").write_text(json.dumps(cfg_dump, indent=2))

    frames = [f for f in mushroom_frames(args.scene, args.capture) if f[f"{args.source}_depth"]]
    print(f"{args.scene.name}/{args.capture} [{args.source}] win={win}: {len(frames)} frames -> {out_dir}")
    for i, fr in enumerate(frames):
        depth = load_depth(fr[f"{args.source}_depth"]).to(DEVICE)
        maps = extract_one(depth, args.source, win)
        np.savez_compressed(
            out_dir / f"{fr['stem']}.npz",
            band=maps["band"].cpu().numpy().astype(np.uint8),
            degenerate=maps["degenerate"].cpu().numpy().astype(np.uint8),
            d_fg=(maps["d_fg"].cpu().numpy() * 1000.0).astype(np.float32),  # mm
            d_bg=(maps["d_bg"].cpu().numpy() * 1000.0).astype(np.float32),  # mm
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames)}")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
