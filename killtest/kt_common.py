"""Shared loaders and band computation for the bimodal-depth kill tests.

Conventions:
- all depths in meters, torch float32, shape [H, W]
- edge/band masks are bool [H, W]
- everything computed at the RGB camera resolution (what the loss sees):
  depth maps are resized with nearest interpolation exactly like
  nerfstudio's get_depth_image_from_path (cv2.INTER_NEAREST).
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kt_config import CONFIG

MM_TO_M = 1e-3


# ---------------------------------------------------------------- loading


def load_depth(path: Path, hw=None) -> torch.Tensor:
    """Load a depth map (png in mm, or npy in mm) -> [H, W] float32 meters.

    Mirrors nerfstudio get_depth_image_from_path / dn_splatter depth_path_to_tensor:
    png uint16 mm, npy float mm, nearest-neighbor resize to target (w, h).
    """
    path = Path(path)
    if path.suffix == ".npy":
        d = np.load(str(path)).astype(np.float64)
        if d.ndim == 3:
            d = d[..., 0]
        d = d * MM_TO_M
    else:
        d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float64) * MM_TO_M
    if hw is not None and d.shape[:2] != tuple(hw):
        d = cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(d).float()


def load_rgb(path: Path) -> torch.Tensor:
    """[H, W, 3] float32 in [0, 1]."""
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float() / 255.0


# ---------------------------------------------------------------- masks


def valid_mask(depth: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(depth)
        & (depth > CONFIG["depth_tolerance"])
        & (depth < CONFIG["max_depth"])
    )


def edge_map_grad_logd(depth: torch.Tensor) -> torch.Tensor:
    """Ours: ||grad log D|| > thr (forward differences, invalid-aware)."""
    v = valid_mask(depth)
    logd = torch.where(v, torch.log(depth.clamp(min=1e-6)), torch.nan)
    gx = torch.abs(logd[:, 1:] - logd[:, :-1])
    gy = torch.abs(logd[1:, :] - logd[:-1, :])
    gx = F.pad(gx, (0, 1), value=torch.nan)
    gy = F.pad(gy, (0, 0, 0, 1), value=torch.nan)
    mag = torch.sqrt(torch.nan_to_num(gx, nan=0.0) ** 2 + torch.nan_to_num(gy, nan=0.0) ** 2)
    return (mag > CONFIG["grad_logd_threshold"]) & v


def edge_map_repo_laplacian(depth: torch.Tensor) -> torch.Tensor:
    """Repo's find_edges: laplacian of inverse depth > thr (signed), no dilation here."""
    v = valid_mask(depth)
    inv = 1.0 / (depth.clamp(min=1e-6) + 1e-6)
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    lap = F.conv2d(inv.view(1, 1, *inv.shape), k, padding=1).view(*inv.shape)
    return (lap > CONFIG["repo_laplacian_threshold"]) & v


def edge_map_window_contrast(depth: torch.Tensor) -> torch.Tensor:
    """Smear-robust: local p90/p10 depth contrast in a small window > thr."""
    v = valid_mask(depth)
    d_fg, d_bg = local_fg_bg(depth)
    c = d_bg / d_fg.clamp(min=1e-6)
    return (torch.nan_to_num(c, nan=0.0) > CONFIG["window_contrast_threshold"]) & v


EDGE_DEFS = {
    "grad_logd": edge_map_grad_logd,
    "repo_laplace": edge_map_repo_laplacian,
    "window_contrast": edge_map_window_contrast,
}


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Square-kernel binary dilation by `px` pixels."""
    k = 2 * px + 1
    m = mask.float().view(1, 1, *mask.shape)
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=px)
    return out.view(*mask.shape) > 0.5


def edge_aware_weight(rgb: torch.Tensor) -> torch.Tensor:
    """Per-pixel EdgeAwareLogL1-style weight exp(-|grad RGB|), avg of x/y terms.

    Summary of the repo's separate x/y weighting (losses.py:200-207) into one
    per-pixel scalar for exposure bookkeeping.
    """
    gx = torch.mean(torch.abs(rgb[:, 1:, :] - rgb[:, :-1, :]), dim=-1)
    gy = torch.mean(torch.abs(rgb[1:, :, :] - rgb[:-1, :, :]), dim=-1)
    gx = F.pad(gx, (0, 1), mode="replicate")
    gy = F.pad(gy.T, (0, 1), mode="replicate").T
    return 0.5 * (torch.exp(-gx) + torch.exp(-gy))


# ---------------------------------------------------------------- local fg/bg structure


def local_fg_bg(depth: torch.Tensor):
    """Per-pixel local fg/bg depths from a small window of the prior itself.

    Returns (d_fg, d_bg): [H, W] tensors = p10/p90 of valid depths in the window
    (percentiles, not min/max, so the extracted surfaces are robust to the very
    flying pixels we are trying to count).
    """
    w = CONFIG["contrast_window"]
    pad = w // 2
    v = valid_mask(depth)
    d = torch.where(v, depth, torch.nan)
    patches = F.pad(d.view(1, 1, *d.shape), (pad, pad, pad, pad), value=torch.nan)
    patches = patches.unfold(2, w, 1).unfold(3, w, 1)  # [1,1,H,W,w,w]
    patches = patches.reshape(*d.shape, w * w)
    d_fg = torch.nanquantile(patches, CONFIG["fg_percentile"] / 100.0, dim=-1)
    d_bg = torch.nanquantile(patches, CONFIG["bg_percentile"] / 100.0, dim=-1)
    return d_fg, d_bg


def summarize(x: torch.Tensor):
    """median/p90/mean/count of a 1D tensor (empty-safe)."""
    if x.numel() == 0:
        return {"median": None, "p90": None, "mean": None, "n": 0}
    return {
        "median": float(x.median()),
        "p90": float(torch.quantile(x, 0.9)) if x.numel() < 16_000_000 else float(
            np.percentile(x.numpy(), 90)
        ),
        "mean": float(x.mean()),
        "n": int(x.numel()),
    }


# ---------------------------------------------------------------- mushroom layout


def mushroom_frames(scene_dir: Path, capture: str = "long_capture"):
    """List frames of a MuSHRoom iphone capture.

    Returns list of dicts with paths: image, sensor_depth, mono_depth (may be
    None), reference_depth (may be None).
    """
    base = Path(scene_dir) / "iphone" / capture
    images = sorted((base / "images").glob("*"))
    frames = []
    for img in images:
        stem = img.stem
        sensor = base / "depth" / f"{stem}.png"
        # KT_MONO_DIR selects an alternative mono-depth source (e.g. mono_depth_dav2)
        mono_dir = os.environ.get("KT_MONO_DIR", "mono_depth")
        mono = base / mono_dir / f"{stem}_aligned.npy"
        if not mono.exists():
            mono = base / mono_dir / f"{stem}.npy"
        ref = base / "reference_depth" / f"{stem}.png"
        frames.append(
            {
                "stem": stem,
                "image": img,
                "sensor_depth": sensor if sensor.exists() else None,
                "mono_depth": mono if mono.exists() else None,
                "reference_depth": ref if ref.exists() else None,
            }
        )
    return frames


def subsample(frames, max_n):
    if len(frames) <= max_n:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_n).round().astype(int)
    return [frames[i] for i in idx]


# ------------------------------------------- two-sided hypothesis extraction


def two_sided_hypotheses(depth: torch.Tensor, band: torch.Tensor,
                         r_min: int = 3, r_max: int = 8):
    """Edge-normal two-sided sampling: for each band pixel, walk along +/- the
    depth-edge normal (direction of grad log D, box-blurred so it is defined
    across the dilated band), skip the mixed pixels (r < r_min), take the
    median valid depth on each side, and return (d_fg, d_bg) = (min, max) of
    the two side medians. Unlike the windowed-percentile extractor this asks
    for the surface immediately on EACH side of THIS edge, not two extreme
    depth populations somewhere in a square window.

    depth [H,W] meters; band bool [H,W]. Returns (d_fg, d_bg, ok) full-size
    maps: nan outside band or where a side had no valid samples; ok = band
    pixels with both sides valid.
    """
    H, W = depth.shape
    v = valid_mask(depth)
    logd = torch.where(v, torch.log(depth.clamp(min=1e-6)), torch.nan)
    gx = F.pad(logd[:, 1:] - logd[:, :-1], (0, 1), value=torch.nan)
    gy = F.pad(logd[1:, :] - logd[:-1, :], (0, 0, 0, 1), value=torch.nan)
    gx = torch.nan_to_num(gx, nan=0.0)
    gy = torch.nan_to_num(gy, nan=0.0)
    k = 5  # spread direction over the dilated band
    gx = F.avg_pool2d(gx.view(1, 1, H, W), k, 1, k // 2).view(H, W)
    gy = F.avg_pool2d(gy.view(1, 1, H, W), k, 1, k // 2).view(H, W)
    mag = torch.sqrt(gx**2 + gy**2).clamp(min=1e-12)
    nx, ny = gx / mag, gy / mag  # (col, row) components of the edge normal

    ys, xs = torch.nonzero(band, as_tuple=True)
    n_dir_x, n_dir_y = nx[ys, xs], ny[ys, xs]
    sides = []
    for sign in (-1.0, 1.0):
        samples = []
        for r in range(r_min, r_max + 1):
            yy = (ys + sign * r * n_dir_y).round().long().clamp(0, H - 1)
            xx = (xs + sign * r * n_dir_x).round().long().clamp(0, W - 1)
            d = torch.where(v[yy, xx], depth[yy, xx],
                            torch.full_like(depth[yy, xx], torch.nan))
            samples.append(d)
        sides.append(torch.nanmedian(torch.stack(samples), dim=0).values)
    d_lo = torch.minimum(sides[0], sides[1])  # nan if either side empty
    d_hi = torch.maximum(sides[0], sides[1])

    d_fg = torch.full((H, W), torch.nan, dtype=depth.dtype)
    d_bg = torch.full((H, W), torch.nan, dtype=depth.dtype)
    d_fg[ys, xs] = d_lo
    d_bg[ys, xs] = d_hi
    ok = band & torch.isfinite(d_fg) & torch.isfinite(d_bg)
    return d_fg, d_bg, ok


def kmeans2_hypotheses(depth: torch.Tensor, band: torch.Tensor,
                       win: int = None, iters: int = 8):
    """2-means in inverse depth over the same window as the percentile
    extractor, initialized at its p10/p90: cluster MODES instead of tail
    quantiles. Returns (d_fg, d_bg, ok) full-size maps (nan/False off band).
    """
    win = win or CONFIG["hyp_window"]
    H, W = depth.shape
    v = valid_mask(depth)
    inv = torch.where(v, 1.0 / depth.clamp(min=1e-6), torch.nan)
    pad = win // 2
    patches = F.pad(inv.view(1, 1, H, W), (pad,) * 4, value=torch.nan)
    patches = patches.unfold(2, win, 1).unfold(3, win, 1).reshape(H, W, win * win)
    ys, xs = torch.nonzero(band, as_tuple=True)
    p = patches[ys, xs]  # [N, win*win]
    finite = torch.isfinite(p)
    q = torch.tensor([CONFIG["hyp_fg_percentile"] / 100.0,
                      CONFIG["hyp_bg_percentile"] / 100.0], dtype=p.dtype)
    lo_hi = torch.nanquantile(p, q, dim=-1)  # [2,N]: low inv (far), high inv (near)
    c_lo, c_hi = lo_hi[0], lo_hi[1]
    for _ in range(iters):
        mid = 0.5 * (c_lo + c_hi)
        hi_m = finite & (p > mid.unsqueeze(-1))
        lo_m = finite & ~hi_m

        def cmean(m):
            return torch.where(m, p, torch.zeros_like(p)).sum(-1) / m.sum(-1).clamp(min=1)

        c_hi = torch.where(hi_m.any(-1), cmean(hi_m), c_hi)
        c_lo = torch.where(lo_m.any(-1), cmean(lo_m), c_lo)
    d_fg_b = 1.0 / c_hi.clamp(min=1e-6)  # high inverse depth = near
    d_bg_b = 1.0 / c_lo.clamp(min=1e-6)
    d_fg = torch.full((H, W), torch.nan, dtype=depth.dtype)
    d_bg = torch.full((H, W), torch.nan, dtype=depth.dtype)
    d_fg[ys, xs] = d_fg_b
    d_bg[ys, xs] = d_bg_b
    ok = band & torch.isfinite(d_fg) & torch.isfinite(d_bg)
    return d_fg, d_bg, ok
