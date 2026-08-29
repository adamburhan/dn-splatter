# DN-Splatter loss-surface map (Task 0 recon)

Repo state: commit `97588b4` (main), clean clone. All paths relative to repo root.

## 1. Where the depth loss lives

**Loss definitions** — [dn_splatter/losses.py](../dn_splatter/losses.py)

| Variant | Location | Notes |
|---|---|---|
| `DepthLossType` enum | losses.py:20–32 | MSE, L1, LogL1, HuberL1, TV, EdgeAwareLogL1, EdgeAwareTV, PearsonDepth, LocalPearsonDepthLoss, AdaptiveDepth |
| `L1` / `LogL1` | losses.py:155–184 | plain per-pixel or scalar mean |
| `EdgeAwareLogL1` | losses.py:187–224 | **default depth loss**. Per-pixel LogL1 × `exp(-|∇RGB|)`: weights depth residuals down where the **RGB image** (not the depth prior) has gradients. Forward differences in x/y, mask zeroes invalid pixels, then `.mean()` over surviving pixels |
| `HuberL1` | losses.py:227–248 | threshold relative to max residual in batch |
| `EdgeAwareTV` | losses.py:251–276 | smoothness on rendered depth, weighted by RGB gradients (not a supervision term) |
| `AdaptiveDepth` | losses.py:386–401 | EdgeAwareLogL1; after step 7000 additionally zeroes gt where `confidence_map <= 0` (AGS-Mesh style) |
| `SensorDepthLoss` | losses.py:299–352 | SDF-model leftover (free-space/truncation); NOT used by the splatter model |
| Pearson variants | losses.py:428–485 | scale-invariant correlation losses (SparseGS) |

**Loss application** — [dn_splatter/regularization_strategy.py](../dn_splatter/regularization_strategy.py)

- `DNRegularization.get_depth_loss` — regularization_strategy.py:158–186.
  - Valid mask: `gt_depth > depth_tolerance` (default 0.1 m) — that is the *only* per-pixel gating for the default path.
  - EdgeAwareLogL1 branch passes `gt_img` (RGB) for the edge weighting; other losses just index `pred[valid], gt[valid]`.
  - **Weighting quirk (regularization_strategy.py:184):** `depth_loss += self.depth_lambda * depth_loss` → the depth term enters the total as **(1 + λ)·loss**, not λ·loss. With the recommended `--depth-lambda 0.2`, the depth loss has weight 1.2, comparable to the RGB loss — λ barely modulates it. Any "loss strength" ablation must account for this.
- `AGSMeshRegularization.get_depth_loss` — regularization_strategy.py:257–290: same EdgeAwareLogL1, weight is a clean `λ·loss`, and after step 7000 gt is zeroed where a precomputed **confidence map** is ≤ 0 (loaded via `--load-depth-confidence-masks`, produced by [scripts/depth_normal_consistency.py](../dn_splatter/scripts/depth_normal_consistency.py) from depth/normal consistency).
- `find_edges` — regularization_strategy.py:40–96: **a depth-discontinuity detector already in the repo**: Laplacian of *inverse* depth (`1/(d+1e-6)`), threshold 0.01, dilated (1 iter, 3×3). In AGS-Mesh it is applied to the gt normal map to mask the normal loss (regularization_strategy.py:301), not the depth loss. The 1-channel branch is exactly an inverse-depth edge band → good template/comparison for our band definition.
- `dilate_edge` — regularization_strategy.py:29 (conv-based dilation helper).

**Model hookup** — [dn_splatter/dn_model.py](../dn_splatter/dn_model.py)

- Config: dn_model.py:62–75 — `use_depth_loss` (default False), `depth_loss_type` (default **EdgeAwareLogL1**), `depth_tolerance=0.1`, `depth_lambda=0.0` (assert >0 when enabled; README recommends 0.2), `use_depth_smooth_loss` + TV/EdgeAwareTV.
- `get_loss_dict` — dn_model.py:624–729: rendered `outputs["depth"]` vs `depth_gt`; **if both sensor and mono depth are in the batch, mono wins** (dn_model.py:695–699: sensor assigned first, then overwritten by mono). Loss is computed at full training resolution (`num_downscales=0` by default → native image res).
- Depth loss operates on z-depth in **scene (metric·dataparser_scale) units** — see data flow below.

## 2. Depth prior ingestion + alignment

- Dataset: [dn_splatter/data/dn_dataset.py](../dn_splatter/data/dn_dataset.py) `get_metadata` (dn_dataset.py:119–194): loads `sensor_depth` and/or `mono_depth` per frame with `scale_factor = depth_unit_scale_factor(=1e-3) × dataparser_scale` (mm→m, then scene scaling). Optional euclidean→z conversion (dn_dataset.py:140–151). Confidence maps loaded unscaled.
- ScanNet++: [dn_splatter/data/scannetpp_dataparser.py](../dn_splatter/data/scannetpp_dataparser.py). `depth_mode` ∈ {mono, sensor (default), none, all} (line 50). iPhone mode: sensor depth from `<scene>/iphone/depth/*.png` (ARKit, mm). Mono mode: generated once into `<scene>/iphone/mono_depth/` by [scripts/depth_from_pretrain.py](../dn_splatter/scripts/depth_from_pretrain.py) with `return_mode="mono-aligned"`: **ZoeDepth estimates aligned per-frame to the sensor depth with a scalar scale+shift fitted by MSE grad descent** (`depth_align`, depth_from_pretrain.py:89–142). DSLR mode: [scripts/align_depth.py](../dn_splatter/scripts/align_depth.py) `ColmapToAlignedMonoDepths` — Zoe aligned per-frame to COLMAP SfM sparse depth, closed-form least squares (`compute_scale_and_shift`, align_depth.py:190, from MonoSDF; sparse mask 0.1 < d < 10).
- So the alignment the loss "sees" = per-frame global scale/shift on the mono depth; no local/edge-aware correction anywhere. T1.2 must reuse exactly these aligned `_aligned.npy` / mono_depth files.
- Train/eval split (iPhone): every 10th frame is eval (`skip_every_for_val_split=10`, scannetpp_dataparser.py:83,205), training frames further subsampled by `load_every=5` (line 81). DSLR mode uses ScanNet++ official `train_test_lists.json`.

## 3. Eval pipeline

- [dn_splatter/metrics.py](../dn_splatter/metrics.py): `RGBMetrics` (PSNR/SSIM/LPIPS, metrics.py:77), `DepthMetrics` (metrics.py:111–149: abs_rel, sq_rel, rmse, rmse_log, δ1–3, valid mask `gt > 0.1`), `NormalMetrics`, `PDMetrics` (point cloud).
- In-model metrics: dn_model.py:731–807 — depth metrics computed **against sensor depth only** (`"sensor_depth" in batch`).
- Full eval: [dn_splatter/dn_pipeline.py](../dn_splatter/dn_pipeline.py) `get_average_eval_image_metrics` (dn_pipeline.py:133) — renders eval views, whole-image metrics, optional point-cloud metrics; MuSHRoom gets a special faro-GT depth eval (dn_pipeline.py:628). Standalone eval utilities in [dn_splatter/eval/](../dn_splatter/eval/) (eval.py operates on saved render/gt image folders — convenient to extend with band masks).
- No masked/sliced metric exists anywhere → our edge-band-sliced metrics must be added (simplest: standalone script over saved renders + prior edge maps).

## 4. Existing discontinuity handling (grep: edge/gradient/discontinuity/confidence/uncertainty)

1. `EdgeAwareLogL1` — continuous **downweighting at RGB edges** (proxy for depth edges). This is the repo's "cheap leniency" control and our middle ablation arm.
2. `find_edges` inverse-depth Laplacian band — binary depth-edge mask, but only used to mask the AGS-Mesh **normal** loss.
3. AGS-Mesh confidence maps — per-pixel binary confidence from depth↔normal consistency, zeroes depth supervision after step 7000 (a *learned-free* down-masking, also relevant leniency baseline; needs `--load-depth-confidence-masks`).
4. `AdaptiveDepth`/`AdaptiveNormal` losses — step-scheduled versions of the above.
5. Nothing anywhere supervises against more than one depth hypothesis per pixel; nothing models flying pixels explicitly.

## 5. Data-flow summary (5 lines)

1. Dataparser (e.g. `scannetpp`) lists per-frame RGB + sensor depth (`depth/`, mm PNG) and/or mono depth (`mono_depth/*_aligned` files, ZoeDepth aligned per-frame scale+shift to sensor/SfM depth).
2. `GDataset.get_metadata` loads depths × `1e-3 × dataparser_scale` into batch as `sensor_depth`/`mono_depth` at camera resolution.
3. `DNSplatterModel.get_loss_dict` picks `depth_gt` (mono overrides sensor), renders depth via splatting, hands both to the regularization strategy.
4. `DNRegularization.get_depth_loss` masks `gt > 0.1`, computes EdgeAwareLogL1 (LogL1 × exp(−|∇RGB|)) and adds it to the RGB loss with effective weight (1+λ) [sic], plus normal + scale terms.
5. Eval: nerfstudio eval views (every 10th frame for ScanNet++ iPhone), whole-image PSNR/SSIM/LPIPS + depth abs_rel/RMSE/δ vs sensor depth — no edge-sliced metrics exist.

## Literature sanity check (bimodal/multi-hypothesis depth supervision)

Verdict: **no prior work does two-surface-hypothesis supervision of rendered depth at depth edges** — the idea appears open. Five closest:

1. **SCADE** (Uy et al., CVPR 2023, arXiv:2303.13582) — closest in mechanics: min-over-hypotheses ("space carving") loss over multiple cIMLE-sampled depth-map hypotheses for NeRF. But hypotheses are *global* generative samples of monodepth ambiguity (needs a cIMLE-trained depth net), applied at all pixels, supervising ray-termination samples — not local {d_fg, d_bg} from the prior's edge structure, not 3DGS. **Must cite/differentiate.**
2. **Modeling Depth Ambiguity / mixture-density flying-point-free depth** (arXiv:2606.02552, 2026) — closest in representation: fg/bg mixture NLL at boundary pixels to kill flying points, but a standalone depth *estimator* head, never used to supervise a radiance field.
3. **SMD-Nets** (Tosi et al., CVPR 2021) — bimodal Laplacian mixture over disparity at object boundaries for stereo; output parameterization, not a supervision loss for rendered depth.
4. **TWISE** (Imran et al., CVPR 2021) — twin fg/bg surface prediction at occlusion boundaries in depth completion; same intuition, different problem.
5. **DS-NeRF** (arXiv:2107.02791) + family (URF line-of-sight, EMD-based depth-guided NeRF arXiv:2403.13206) — distribution-level termination losses but a *single* mode per pixel; DET-GS (arXiv:2508.04099) masks regularization at edges = leniency, not two hypotheses.

Taxonomy: (a) termination-distribution losses — only SCADE is genuinely multi-hypothesis; (b) per-pixel uncertainty weighting (e.g. arXiv:2405.19657) — single target; (c) edge downweighting/masking — DN-Splatter EdgeAwareLogL1, DET-GS; (d) two-hypothesis supervision of rendered depth — **not found**.

## Implications for the kill tests

- Band definition: compute ours (‖∇ log D‖ threshold) **and** the repo's `find_edges` inverse-depth Laplacian for comparability; the loss operates at full image resolution, so bands should too.
- T1.2 "repo's own alignment" = the shipped `mono_depth/*_aligned` maps (Zoe + per-frame scale/shift) — do not refit.
- The default loss already downweights at RGB edges; exposure (T1.1) should therefore be reported both raw and weighted by the EdgeAware weight `exp(-|∇RGB|)` so we know how much residual supervision actually lands in the band.
- λ quirk means "unimodal depth loss on" ≈ weight 1.2 regardless of λ ∈ (0, ~0.5]; the no-depth-loss control is `use_depth_loss=False`, not λ→0.
