"""Mesh eval with GT-depth-boundary slicing, for the four-arm bimodal ablation.

Mirrors the repo's MuSHRoom protocol exactly (eval_mesh_mushroom_vis_cull:
icp_iphone alignment of GT, cut_mesh to GT bounds, go-surf visibility culling
over long-capture frames), then computes accuracy/completion/Chamfer-L1/F1@5cm
over three point subsets: all / boundary / interior, where "boundary" = within
`boundary_radius` of any 3D point backprojected from FARO GT depth edges
(grad_logd on reference_depth) — boundary defined from GT, never the prior.

Culled meshes are cached: GT per scene, pred per arm.

Usage:
  python mesh_boundary_eval.py --scene <scene_dir> --pred-mesh <ply> --out <json>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dn_splatter.eval.eval_mesh_mushroom_vis_cull import (  # noqa: E402
    cull_mesh,
    cut_mesh,
    open3d_mesh_from_trimesh,
    trimesh_from_open3d_mesh,
)

from kt_common import edge_map_grad_logd, load_depth, valid_mask  # noqa: E402

N_SAMPLES = 500_000
F1_THRESH = 0.05


def culled_meshes(scene: Path, pred_mesh_path: Path, cache_dir: Path):
    """Aligned + cut + culled (gt, pred) trimeshes, with caching."""
    dataset_path = scene / "iphone" / "long_capture"
    tf_file = dataset_path / "transformations_colmap.json"
    test_ids = [
        x.strip() for x in (dataset_path / "test.txt").read_text().splitlines() if x.strip()
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)

    gt_cache = cache_dir / f"gt_culled_{scene.name}.ply"
    if gt_cache.exists():
        gt_mesh = trimesh.load(gt_cache, force="mesh", process=False)
    else:
        gt_mesh = trimesh.load(str(scene / "gt_mesh.ply"), force="mesh", process=False)
        T = np.array(
            json.load(open(scene / "icp_iphone.json"))["gt_transformation"]
        ).reshape(4, 4)
        gt_mesh = gt_mesh.apply_transform(np.linalg.inv(T))
        gt_mesh = cull_mesh(
            dataset_path, gt_mesh, tf_file, test_ids,
            remove_missing_depth=True, remove_occlusion=True,
            subdivide=True, max_edge=0.015,
        )
        gt_mesh.export(str(gt_cache))

    pred_cache = cache_dir / (pred_mesh_path.parent.name + "_culled.ply")
    if pred_cache.exists():
        pred_mesh = trimesh.load(pred_cache, force="mesh", process=False)
    else:
        pred_mesh = trimesh.load(str(pred_mesh_path), force="mesh", process=False)
        gt_o3d = open3d_mesh_from_trimesh(gt_mesh)
        pred_o3d = cut_mesh(gt_o3d, open3d_mesh_from_trimesh(pred_mesh), kernel_size=15, dilate=True)
        pred_mesh = trimesh_from_open3d_mesh(pred_o3d)
        pred_mesh = cull_mesh(
            dataset_path, pred_mesh, tf_file, test_ids,
            remove_missing_depth=True, remove_occlusion=True,
            subdivide=True, max_edge=0.015,
        )
        pred_mesh.export(str(pred_cache))
    return gt_mesh, pred_mesh


def gt_boundary_points(scene: Path, max_points: int = 300_000):
    """3D points backprojected from Faro reference_depth edge pixels (grad_logd),
    using the same OpenGL->OpenCV pose convention as the culling code."""
    dataset_path = scene / "iphone" / "long_capture"
    tf = json.load(open(dataset_path / "transformations_colmap.json"))
    glob_K = {k: tf[k] for k in ("h", "w", "fl_x", "fl_y", "cx", "cy") if k in tf}
    pts = []
    for frame in tf["frames"]:
        name = Path(frame["file_path"]).stem
        ref_path = dataset_path / "reference_depth" / f"{name}.png"
        if not ref_path.exists():
            continue
        f = {**glob_K, **{k: frame[k] for k in ("h", "w", "fl_x", "fl_y", "cx", "cy") if k in frame}}
        d = load_depth(ref_path)
        edge = (edge_map_grad_logd(d) & valid_mask(d)).numpy()
        if not edge.any():
            continue
        v, u = np.nonzero(edge)
        z = d.numpy()[v, u]
        x = (u - f["cx"]) / f["fl_x"] * z
        y = (v - f["cy"]) / f["fl_y"] * z
        cam = np.stack([x, y, z], axis=1)
        c2w = np.array(frame["transform_matrix"]).astype(np.float64)
        c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL -> OpenCV
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        pts.append(world.astype(np.float32))
    pts = np.concatenate(pts, axis=0)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return pts


def sliced_metrics(pred_mesh, gt_mesh, boundary_pts, radius):
    pred_pts = trimesh.sample.sample_surface(pred_mesh, N_SAMPLES)[0].astype(np.float32)
    gt_pts = trimesh.sample.sample_surface(gt_mesh, N_SAMPLES)[0].astype(np.float32)
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)
    tree_bnd = cKDTree(boundary_pts)

    acc, acc_idx = tree_gt.query(pred_pts, k=1)  # pred -> gt
    comp = tree_pred.query(gt_pts, k=1)[0]  # gt -> pred
    gt_is_bnd = tree_bnd.query(gt_pts, k=1)[0] < radius
    # pred points inherit the boundary label of their nearest GT point, so both
    # slices are anchored to GT regions (junk far from GT can't self-select a slice)
    pred_is_bnd = gt_is_bnd[acc_idx]

    def stats(sel_acc, sel_comp):
        a, c = acc[sel_acc], comp[sel_comp]
        precision = float((a < F1_THRESH).mean()) if len(a) else None
        recall = float((c < F1_THRESH).mean()) if len(c) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else None
        )
        return {
            "acc_mean": float(a.mean()) if len(a) else None,
            "comp_mean": float(c.mean()) if len(c) else None,
            "acc_median": float(np.median(a)) if len(a) else None,
            "comp_median": float(np.median(c)) if len(c) else None,
            "chamfer_l1_median": float(0.5 * (np.median(a) + np.median(c))) if len(a) and len(c) else None,
            "chamfer_l1": float(0.5 * (a.mean() + c.mean())) if len(a) and len(c) else None,
            "precision@5cm": precision,
            "recall@5cm": recall,
            "f1@5cm": f1,
            "n_acc": int(sel_acc.sum()),
            "n_comp": int(sel_comp.sum()),
        }

    return {
        "all": stats(np.ones_like(pred_is_bnd), np.ones_like(gt_is_bnd)),
        "boundary": stats(pred_is_bnd, gt_is_bnd),
        "interior": stats(~pred_is_bnd, ~gt_is_bnd),
        "boundary_frac_gt": float(gt_is_bnd.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--pred-mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("/scratch/adam/experiments/bimodal-depth/mesh_cache"))
    ap.add_argument("--boundary-radius", type=float, default=0.05)
    args = ap.parse_args()

    gt_mesh, pred_mesh = culled_meshes(args.scene, args.pred_mesh, args.cache_dir)
    bnd = gt_boundary_points(args.scene)
    print(f"boundary points: {len(bnd)}")
    res = sliced_metrics(pred_mesh, gt_mesh, bnd, args.boundary_radius)
    res["scene"] = str(args.scene)
    res["pred_mesh"] = str(args.pred_mesh)
    res["boundary_radius"] = args.boundary_radius
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for k in ["all", "boundary", "interior"]:
        s = res[k]
        print(
            f"{k:<9} chamfer {s['chamfer_l1']:.4f}  acc {s['acc_mean']:.4f}  "
            f"comp {s['comp_mean']:.4f}  F1@5cm {s['f1@5cm']:.3f}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
