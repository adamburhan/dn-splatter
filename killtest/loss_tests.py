"""Pre-training sanity tests for the bimodal/masked depth losses.

A) --loss-descent: optimize a free depth image under BimodalLogL1 on a synthetic
   step edge whose prior contains flying pixels. Band pixels must converge to one
   of the two plateaus (not the phantom middle); off-band pixels to the prior.
B) --dataloader: build the real MuSHRoom datamanager, pull training batches, and
   verify the cached hypothesis maps arrive pixel-aligned and unit-consistent
   with the sensor depth the loss uses; saves an overlay jpg for eyeballing.

Usage:
  python loss_tests.py --loss-descent
  python loss_tests.py --dataloader --scene /path/to/koivu --out align.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def loss_descent():
    from dn_splatter.losses import BimodalLogL1, MaskedLogL1

    H, W = 32, 32
    gt = torch.full((H, W, 1), 2.0)
    gt[:, 16:] = 4.0
    gt[:, 15:18] = 3.0  # flying pixels in the prior
    band = torch.zeros(H, W, 1, dtype=torch.bool)
    band[:, 13:20] = True
    d_fg = torch.full((H, W, 1), 2.0)
    d_bg = torch.full((H, W, 1), 4.0)
    degen = torch.zeros_like(band)
    valid = torch.ones_like(band)

    pred = torch.full((H, W, 1), 3.01, requires_grad=True)  # near the phantom middle
    loss_fn = BimodalLogL1()
    opt = torch.optim.Adam([pred], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(pred, gt, valid, band, d_fg, d_bg, degen)
        loss.backward()
        opt.step()
    p = pred.detach()
    in_band = p[band]
    dist_to_plateau = torch.minimum((in_band - 2.0).abs(), (in_band - 4.0).abs())
    assert dist_to_plateau.max() < 0.1, f"band pixels stuck mid: {dist_to_plateau.max()}"
    assert not ((in_band > 2.5) & (in_band < 3.5)).any(), "phantom middle survived"
    off = p[~band & (gt == 2.0)]
    assert (off - 2.0).abs().max() < 0.1
    # masked loss: band pixels must receive zero gradient
    pred2 = torch.full((H, W, 1), 3.0, requires_grad=True)
    m = MaskedLogL1()(pred2, gt, valid, band)
    m.backward()
    g = pred2.grad
    assert g[band].abs().max() == 0, "masked arm leaks gradient into band"
    assert g[~band].abs().max() > 0
    # soft mixture: must still escape the phantom middle, and asymmetric side
    # mass must break the tie toward the heavier mode
    from dn_splatter.losses import SoftBimodalLogL1

    pi = torch.full((H, W, 1), 0.8)  # fg-heavy windows
    pred3 = torch.full((H, W, 1), 3.0, requires_grad=True)  # exact midpoint
    soft = SoftBimodalLogL1(tau=0.1)
    opt3 = torch.optim.Adam([pred3], lr=0.05)
    for _ in range(400):
        opt3.zero_grad()
        loss = soft(pred3, gt, valid, band, d_fg, d_bg, degen, pi)
        assert torch.isfinite(loss), "soft loss not finite"
        loss.backward()
        opt3.step()
    p3 = pred3.detach()[band]
    dist3 = torch.minimum((p3 - 2.0).abs(), (p3 - 4.0).abs())
    assert dist3.max() < 0.15, f"soft: band pixels stuck mid: {dist3.max()}"
    frac_fg = ((p3 - 2.0).abs() < (p3 - 4.0).abs()).float().mean()
    assert frac_fg > 0.5, f"soft: pi=0.8 fg did not break midpoint tie ({frac_fg})"
    # multi-hyp: K candidates (two real plateaus + one invalid slot) must
    # behave like bimodal on the synthetic edge
    from dn_splatter.losses import MultiHypLogL1

    d_hyp = torch.cat([d_fg, d_bg, torch.zeros_like(d_fg)], dim=-1)  # [H,W,3]
    pred4 = torch.full((H, W, 1), 3.01, requires_grad=True)
    mh = MultiHypLogL1()
    opt4 = torch.optim.Adam([pred4], lr=0.05)
    for _ in range(400):
        opt4.zero_grad()
        loss = mh(pred4, gt, valid, band, d_hyp, degen)
        assert torch.isfinite(loss), "multihyp loss not finite"
        loss.backward()
        opt4.step()
    p4 = pred4.detach()[band]
    dist4 = torch.minimum((p4 - 2.0).abs(), (p4 - 4.0).abs())
    assert dist4.max() < 0.1, f"multihyp: band pixels stuck mid: {dist4.max()}"
    print(f"LOSS_DESCENT_OK  band range [{in_band.min():.2f},{in_band.max():.2f}], "
          f"masked band grad = 0, soft frac_fg={frac_fg:.2f}, multihyp ok")


def dataloader_check(scene: Path, out: Path):
    from dn_splatter.data.mushroom_dataparser import MushroomDataParserConfig
    from dn_splatter.dn_datamanager import DNSplatterManagerConfig

    dm_cfg = DNSplatterManagerConfig(
        dataparser=MushroomDataParserConfig(data=scene, mode="iphone")
    )
    dm = dm_cfg.setup(device="cpu", test_mode="val")
    n_checked, band_fracs = 0, []
    overlay_saved = False
    for i in range(len(dm.train_dataset)):
        batch = dm.train_dataset[i]
        if "hyp_band" not in batch:
            continue
        img = batch["image"]
        band = batch["hyp_band"] > 0.5
        sensor = batch["sensor_depth"]
        d_fg, d_bg = batch["hyp_d_fg"], batch["hyp_d_bg"]
        assert band.shape[:2] == img.shape[:2] == sensor.shape[:2] == d_fg.shape[:2]
        # unit consistency: hypotheses bracket the sensor depth off-degenerate
        v = (sensor > 0.1) & (d_fg > 0) & (d_bg > 0)
        ratio = (sensor[v] / ((d_fg[v] + d_bg[v]) / 2)).median()
        assert 0.5 < ratio < 2.0, f"unit mismatch: sensor/hyp ratio {ratio}"
        band_fracs.append(float(band.float().mean()))
        if not overlay_saved:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            over = img.numpy().copy()
            over[band.squeeze(-1).numpy()] = [1.0, 0.1, 0.1]
            plt.imsave(out, over)
            overlay_saved = True
        n_checked += 1
        if n_checked >= 10:
            break
    assert n_checked > 0, "no batches carried hypothesis maps"
    print(f"DATALOADER_OK  {n_checked} batches, band frac "
          f"{np.mean(band_fracs)*100:.2f}% (min {min(band_fracs)*100:.2f}, "
          f"max {max(band_fracs)*100:.2f}), overlay -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-descent", action="store_true")
    ap.add_argument("--dataloader", action="store_true")
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--out", type=Path, default=Path("hyp_align_check.jpg"))
    args = ap.parse_args()
    if args.loss_descent:
        loss_descent()
    if args.dataloader:
        dataloader_check(args.scene, args.out)
