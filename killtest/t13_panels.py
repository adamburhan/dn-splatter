"""T1.3 qualitative panels: gt rgb | rendered rgb (+prior edge overlay) | rendered depth | depth error.

Usage:
  python t13_panels.py --render-dirs <dir_arm1> <dir_arm2> ... --scene <scene_dir> \
      --out panels_dir --n-views 5

One row per model arm, one figure per selected view. Prior edge band (grad_logd,
2px) overlaid in red on the rendered rgb; rendered depth with shared color range;
|rendered - faro| error map when faro GT exists.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kt_common import EDGE_DEFS, dilate, load_depth, load_rgb, valid_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-views", type=int, default=5)
    ap.add_argument("--edge-def", default="grad_logd")
    ap.add_argument("--band-px", type=int, default=2)
    args = ap.parse_args()

    labels = args.labels or [d.name for d in args.render_dirs]
    base = args.scene / "iphone" / args.capture
    prefix = f"{args.capture}_"
    stems = sorted(
        p.stem for p in (args.render_dirs[0] / "eval" / "pred" / "rgb").glob("*.png")
    )
    stems = [
        s
        for s in stems
        if s.startswith(prefix)
        and (base / "depth" / f"{s[len(prefix):]}.png").exists()
    ]
    sel = [stems[i] for i in np.linspace(0, len(stems) - 1, args.n_views).round().astype(int)]
    args.out.mkdir(parents=True, exist_ok=True)

    for stem in sel:
        name = stem[len(prefix):]
        prior = load_depth(base / "depth" / f"{name}.png")
        edge = EDGE_DEFS[args.edge_def](prior)
        band = dilate(edge, args.band_px)
        ref_p = base / "reference_depth" / f"{name}.png"
        ref = load_depth(ref_p, hw=prior.shape) if ref_p.exists() else None

        n_arms = len(args.render_dirs)
        fig, axes = plt.subplots(n_arms, 4, figsize=(16, 3.2 * n_arms), squeeze=False)
        # shared depth range from prior
        vmax = float(prior[valid_mask(prior)].quantile(0.98))
        for i, (rdir, lab) in enumerate(zip(args.render_dirs, labels)):
            pred = load_rgb(rdir / "eval" / "pred" / "rgb" / f"{stem}.png")
            gt = load_rgb(rdir / "eval" / "gt" / "rgb" / f"{stem}.png")
            pdepth = (
                np.load(rdir / "eval" / "pred" / "depth" / "raw" / f"{stem}.npy").squeeze()
                * 1e-3
            )

            over = pred.clone()
            b = band.numpy()
            over[..., 0][b] = 1.0
            over[..., 1][b] *= 0.2
            over[..., 2][b] *= 0.2

            axes[i][0].imshow(gt.numpy())
            axes[i][0].set_ylabel(lab, fontsize=10)
            axes[i][0].set_title("gt rgb" if i == 0 else "")
            axes[i][1].imshow(over.numpy())
            axes[i][1].set_title("render + prior edge band" if i == 0 else "")
            axes[i][2].imshow(pdepth, cmap="turbo", vmin=0, vmax=vmax)
            axes[i][2].set_title("rendered depth" if i == 0 else "")
            if ref is not None:
                err = np.abs(pdepth - ref.numpy())
                err[~valid_mask(ref).numpy()] = 0
                axes[i][3].imshow(err, cmap="magma", vmin=0, vmax=0.5)
                axes[i][3].set_title("|depth - faro| (0-0.5m)" if i == 0 else "")
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out / f"panel_{stem}.jpg", dpi=110)
        plt.close(fig)
        print("saved", args.out / f"panel_{stem}.jpg")


if __name__ == "__main__":
    with torch.no_grad():
        main()
