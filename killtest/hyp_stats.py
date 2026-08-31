"""Extractor validation: how much of the active band is genuinely two-surface?

Reads cached bimodal_hyp npz for each scene's long capture and reports, per
scene and pooled:
  - band coverage (band px / all px)
  - gate survival: fraction of band px where the fg/bg gap is meaningful
    (~degenerate & both hypotheses valid) -> the pixels the bimodal loss acts on
  - hypothesis separation |d_bg - d_fg| on surviving px: percentiles + histogram
    (absolute cm and relative gap / d_fg)

  python hyp_stats.py --data-root <room_datasets> --scenes koivu vr_room ... \
      --out hyp_stats.json [--fig hyp_stats.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np

HIST_EDGES_CM = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1e9])


def scene_stats(hyp_dir: Path):
    n_px = n_band = n_active = 0
    gaps = []
    for f in sorted(hyp_dir.glob("*.npz")):
        d = np.load(f)
        band = d["band"] > 0
        active = band & (d["degenerate"] == 0) & (d["d_fg"] > 0) & (d["d_bg"] > 0)
        n_px += band.size
        n_band += int(band.sum())
        n_active += int(active.sum())
        if active.any():
            gap_cm = (d["d_bg"][active] - d["d_fg"][active]) / 10.0  # mm -> cm
            rel = (d["d_bg"][active] - d["d_fg"][active]) / d["d_fg"][active]
            gaps.append((gap_cm, rel))
    gap_cm = np.concatenate([g for g, _ in gaps])
    rel = np.concatenate([r for _, r in gaps])
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "n_frames": len(list(hyp_dir.glob("*.npz"))),
        "band_coverage": n_band / n_px,
        "gate_survival": n_active / max(n_band, 1),
        "gap_cm": {p: q(gap_cm, p) for p in [10, 25, 50, 75, 90]},
        "gap_rel": {p: q(rel, p) for p in [10, 25, 50, 75, 90]},
        "hist_cm": np.histogram(gap_cm, bins=HIST_EDGES_CM)[0].tolist(),
        "_gap_cm": gap_cm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--capture", default="long_capture")
    ap.add_argument("--hyp-name", default="bimodal_hyp")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    print(f"{'scene':<13}{'frames':>7}{'band%':>7}{'survive%':>9}"
          f"{'gap_med_cm':>11}{'gap_p10':>8}{'gap_p90':>9}{'rel_med':>8}")
    for s in args.scenes:
        hyp = args.data_root / s / "iphone" / args.capture / args.hyp_name
        r = scene_stats(hyp)
        results[s] = r
        print(f"{s:<13}{r['n_frames']:>7}{r['band_coverage'] * 100:>6.1f}%"
              f"{r['gate_survival'] * 100:>8.1f}%{r['gap_cm'][50]:>11.1f}"
              f"{r['gap_cm'][10]:>8.1f}{r['gap_cm'][90]:>9.1f}{r['gap_rel'][50]:>8.2f}")

    pooled = np.concatenate([r.pop("_gap_cm") for r in results.values()])
    results["_pooled"] = {
        "gap_cm": {p: float(np.percentile(pooled, p)) for p in [10, 25, 50, 75, 90]},
        "hist_cm": np.histogram(pooled, bins=HIST_EDGES_CM)[0].tolist(),
        "hist_edges_cm": HIST_EDGES_CM[:-1].tolist(),
    }
    print(f"\npooled gap cm p10/50/90: {results['_pooled']['gap_cm'][10]:.1f} / "
          f"{results['_pooled']['gap_cm'][50]:.1f} / {results['_pooled']['gap_cm'][90]:.1f}")
    args.out.write_text(json.dumps(results, indent=2))

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(pooled, 0, 640), bins=80, log=True, color="#444")
        ax.set_xlabel("hypothesis separation |d_bg - d_fg| on active band (cm)")
        ax.set_ylabel("pixels (log)")
        ax.axvline(10, color="r", ls="--", lw=1, label="10 cm")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.fig, dpi=130)
        print("fig:", args.fig)


if __name__ == "__main__":
    main()
