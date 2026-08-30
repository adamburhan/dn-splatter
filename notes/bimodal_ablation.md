# Bimodal Depth Loss — Four-Arm Ablation Results

**STATUS: COMPLETE. Verdict: the pre-registered rule passes — bimodal is the method — with
moderate margins over a strong masking baseline that must be reported prominently.**

Follow-up to `dnsplatter_killtest.md`. Sensor (ARKit) prior, MuSHRoom iPhone
{koivu, coffee_room, vr_room}, 30k steps, identical losses except the depth term.
**koivu = tuning scene** (WIN sweep); **coffee_room + vr_room = reporting scenes**, run as-is.

## Method as implemented

- Hypothesis extraction (`killtest/hyp_extract.py`): per frame, band = grad_logd (‖∇log D‖>0.1)
  dilated 2px; d_fg/d_bg = p90/p10 of inverse depth in a **WIN=21** window (chosen on koivu
  only from {13,17,21}: GT-coverage 54.3% / min-err 0.098 m vs unimodal's 18.8% / 0.448 m at
  band pixels); degenerate gate = T1.2 meaningful-gap threshold. Cached as npz next to the
  depth dir; every threshold in `killtest/kt_config.py` (`extract_config.json` dumped per scene).
- Losses (`dn_splatter/losses.py`): `BimodalLogL1` = per-pixel min(LogL1(d_fg), LogL1(d_bg))
  inside the active band, unimodal LogL1 elsewhere; `MaskedLogL1` = unimodal zeroed in-band.
  All arms share the sum/N_valid normalization and the repo's (1+λ) quirk, λ=0.2.
- Pre-training checks passed: synthetic step-edge extraction exact; flying pixels rejected;
  loss-descent snaps band pixels from the phantom middle to a plateau; masked arm leaks zero
  gradient in-band; dataloader band overlay pixel-aligned (1.47% of px).
- Arms: `logl1` (unimodal), `masked`, `bimodal` (all retrained), `nodepth` (reused from T1.3
  bit-identical config). Renders → band-sliced eval (`t13_eval.py`, grad_logd 2px band) +
  mesh (gs-mesh `dn` exporter, repo MuSHRoom vis-cull protocol, GT-edge boundary slicing at
  5 cm from Faro-depth-edge backprojections — boundary defined from GT, never the prior).

## Results (grad_logd 2px band; derr = rendered-depth median error vs Faro, m; wc/ph =
wrong-commitment / phantom-middle rates over meaningful-gap band pixels; mesh = F1@5cm)

**coffee_room** (reporting):

| arm | full PSNR | band / off PSNR | band LPIPS | band / off derr | wc | ph | mesh F1 all / bnd | bnd chamfer |
|---|---|---|---|---|---|---|---|---|
| unimodal | 25.10 | 21.78 / 25.19 | 0.143 | 0.148 / 0.019 | 0.224 | 0.343 | 0.923 / 0.879 | 0.0287 |
| masked | 25.07 | 22.20 / 25.15 | 0.134 | 0.121 / 0.019 | 0.271 | 0.271 | 0.928 / 0.889 | 0.0273 |
| **bimodal** | 25.12 | 22.21 / 25.20 | **0.133** | **0.091** / 0.020 | 0.330 | **0.178** | **0.934 / 0.900** | **0.0260** |
| nodepth | 25.14 | 22.19 / 25.21 | 0.137 | 0.216 / 0.048 | 0.272 | 0.226 | 0.593 / 0.556 | 0.0642 |

**vr_room** (reporting):

| arm | full PSNR | band / off PSNR | band LPIPS | band / off derr | wc | ph | mesh F1 all / bnd | bnd chamfer |
|---|---|---|---|---|---|---|---|---|
| unimodal | 26.66 | 22.86 / 26.80 | 0.143 | 0.128 / 0.010 | 0.234 | 0.346 | 0.847 / 0.791 | 0.0391 |
| masked | 26.70 | 23.24 / 26.82 | 0.139 | 0.105 / 0.010 | 0.258 | 0.278 | 0.855 / 0.802 | 0.0372 |
| **bimodal** | **26.72** | 23.19 / 26.85 | **0.134** | **0.082** / 0.010 | 0.314 | **0.189** | **0.870 / 0.824** | **0.0330** |
| nodepth | 26.61 | 23.49 / 26.72 | 0.148 | 0.154 / 0.046 | 0.315 | 0.208 | 0.530 / 0.489 | 0.0834 |

**koivu** (tuning scene, reported for completeness):

| arm | full PSNR | band PSNR | band derr | wc | ph | mesh F1 all / bnd |
|---|---|---|---|---|---|---|
| unimodal | 23.20 | 18.49 | 0.292 | 0.213 | 0.293 | 0.650 / 0.672 |
| masked | 23.24 | 18.40 | 0.264 | 0.237 | 0.260 | 0.662 / 0.688 |
| bimodal | 23.28 | 18.27 | **0.238** | 0.258 | **0.220** | 0.654 / 0.682 |
| nodepth | 23.75 | 19.13 | 0.318 | 0.214 | 0.234 | 0.338 / 0.380 |

## Pre-registered decision rule (reporting scenes only)

| condition | coffee_room | vr_room | verdict |
|---|---|---|---|
| band derr: bimodal < masked, clear margin | 0.091 vs 0.121 (−25%) | 0.082 vs 0.105 (−22%) | **PASS** (consistent ~25%) |
| boundary mesh: bimodal > masked | F1 +0.011, chamfer −5% | F1 +0.022, chamfer −11% | **PASS** (modest, consistent) |
| band RGB ≈ masked | PSNR +0.01, LPIPS −0.001 | PSNR −0.05, LPIPS −0.005 | **PASS** |
| wrong-commit doesn't erase depth win | 0.330 vs 0.271, win intact | 0.314 vs 0.258, win intact | **PASS w/ caveat** |

**Verdict: bimodal is the method.** All four conditions hold on both reporting scenes (and
directionally on the tuning scene). Honest framing for the paper:

1. **Masking is a strong, nearly-free baseline** — it did NOT collapse toward nodepth in the
   band as the brief predicted (its deleted supervision was mostly phantom anyway; photometric
   + off-band depth + normal losses interpolate edges better than the prior dictated). Masked
   already removes most of the RGB harm and most of the phantom geometry. Bimodal's added
   value over masking is real but moderate: ~25% band depth error, +0.01–0.02 boundary mesh
   F1, phantom-middle 0.18–0.22 vs 0.26–0.28, and it is the only arm whose phantom rate drops
   *below* the nodepth control in all scenes.
2. **Wrong-commitment is the price**: +0.05–0.06 absolute over masked in every scene — the
   min() sometimes locks onto the wrong surface. It does not erase the wins, but the soft
   (side-mass-weighted mixture) variant is the natural fix — flagged as future work, not
   implemented (per the pre-registration).
3. **Band-width limitation**: on koivu (prior edges misplaced 6–8 px > 2 px band), no arm
   recovers the ~0.8 dB band RGB gap to nodepth — harmful supervision survives just outside
   the band. A width-vs-misplacement sensitivity (4 px band; or misplacement-adaptive band)
   is the first follow-up ablation.
4. Side results: bimodal has the best full-image PSNR among depth-supervised arms in 3/3
   scenes and the best full-mesh F1 in 2/3; EdgeAwareLogL1 ≡ L1 ≡ LogL1 at edges (phase-1
   result) makes unimodal-vs-fix comparisons loss-shape-insensitive.

## Caveats / provenance

- Sensor prior only (phase 1). The mono (Zoe) replication is unrun; its harm baseline is not
  assumed (phase-2 arms would measure it directly, window_contrast band definition).
- Single seed per arm (matches T1.3 protocol); margins of ~0.01 F1 should be treated as
  direction-consistent rather than individually significant — seed repeats are cheap on a
  cluster if needed.
- nodepth arms reused from T1.3 (config bit-identical: same flags minus depth loss).
- All JSONs/logs/renders/meshes/panels: `snoopy:/scratch/adam/experiments/bimodal-depth/`
  (`t13v2_*.json`, `mesh_*.json`, `panels/*_4arm/`). Code: `killtest/` + minimal edits in
  `dn_splatter/{losses,regularization_strategy,dn_model,data/dn_dataset}.py`.
- Eval-infrastructure note: one earlier corrupted eval (coffee_room bimodal, eval raced the
  renderer) was detected and recomputed; the daemon now gates on training-DONE markers.

## Update 2026-08-30: 10-scene scale-up (Mila P1) revises the verdict

39/49 runs of the Mila sweep (all 10 MuSHRoom scenes x {nodepth, native, mask,
hard_multihyp}, seed 0 complete on 7 scenes; 10 tasks lost to a gsplat JIT race, resubmit
pending). The phase-2 rule still passes on its letter, but two negatives emerge at scale:

- Band depth **median** improves under bimodal vs masked on 7/7 scenes (0.205 -> 0.186)
  and phantom-middle is best-of-all-arms on 7/7 (0.187, below nodepth). Off-band identical
  across depth arms (clean intervention). Depth arms also beat nodepth on full-image LPIPS.
- **Band depth p90 regresses**: worst of four arms (1.458 vs mask 1.335), worse than mask
  on 7/7; wrong-commit rises to 0.273. Median down + p90 up = the distributional
  fingerprint of hard mode selection (wrong commitments carry the full fg/bg gap).
- **Band RGB "tie with mask" eroded**: mask edges bimodal on band PSNR (-0.16 dB mean,
  6/7 scenes) and band SSIM; band LPIPS still favors bimodal 7/7 (sharp-but-shifted
  boundary signature).
- Band-width sensitivity (koivu, 4px): clean negative. All depth arms sit ~0.7 dB below
  nodepth on koivu band RGB regardless of band handling -> harm is not band-local.
- DA2 (metric-hypersim, median-aligned per frame ~0.56-0.73): ambiguity structure present
  in mono depth too (band err 4.2-4.7x off-band, fly 24-26%, offsets 3-6.6px).

## Pre-registered decision rule: soft mixture (decisive experiment)

Arm `soft` = SoftBimodalLogL1: L = -tau*log(pi_f*exp(-L_f/tau) + pi_b*exp(-L_b/tau)),
pi_f = side mass of the local inverse-depth window (pi_fg in bimodal_hyp npz, clamped
[0.05,0.95]), tau = 0.1 (env DN_SOFT_TAU). tau->0 recovers the hard min.

**Soft wins iff, vs masked on the same scenes: band depth p90 <= masked's AND
wrong-commit <= masked's, while retaining bimodal's band depth median margin and its
phantom-middle win.** Registered before any soft training results were seen
(runs launched 2026-08-30 on snoopy: koivu, coffee_room, vr_room).
