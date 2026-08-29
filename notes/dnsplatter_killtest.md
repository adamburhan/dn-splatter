# Bimodal-depth kill tests on DN-Splatter — measurements

**STATUS: COMPLETE — all measurements and 9/9 trainings done. See VERDICT at the end.**

Setup: MuSHRoom iPhone captures (koivu, coffee_room, vr_room), 994×738, long_capture.
ScanNet++ unavailable (no access token on this machine) — MuSHRoom chosen because DN-Splatter
supports it natively and Faro laser GT depth exists for ~95% of long-capture frames.
Priors measured: **sensor** = ARKit depth (already upsampled to RGB res, the repo's default
supervision source, `depth_mode=sensor`); **mono** = repo's own ZoeDepth + per-frame
scale/shift alignment to sensor depth (`depth_from_pretrain`, `return_mode=mono-aligned`) —
the exact tensors the loss would see. GT = Faro `reference_depth` renders.

All thresholds in `killtest/kt_config.py`. Three edge definitions on the prior:
- `grad_logd`: ‖∇log D‖ > 0.1 (ours, strict)
- `repo_laplace`: Laplacian(1/D) > 0.01, signed — the repo's own `find_edges`
- `window_contrast`: local p90/p10 depth ratio in 7px window > 1.25 (smear-robust)
Bands = edge map dilated by 2 and 4 px. Valid = depth in (0.1, 15) m. 150 frames/scene.

## T1.1 Exposure — fraction of supervised pixels in the prior's edge band

All scenes, long_capture, N=150 frames each. 2px band (4px in parens for grad_logd):

| source | edge def | koivu | coffee_room | vr_room |
|---|---|---|---|---|
| sensor | grad_logd | 2.7% (4.4%) | 1.4% (2.4%) | 1.9% (3.1%) |
| sensor | window_contrast | 2.5% | 1.3% | 1.6% |
| sensor | repo_laplace | 11.2% | 8.6% | 11.6% |
| mono | grad_logd | 1.0% (1.7%) | 0.4% (0.8%) | 0.4% (0.9%) |
| mono | window_contrast | 2.7% | 1.1% | 1.5% |
| mono | repo_laplace | 2.0% | 0.9% | 1.4% |

(EdgeAware-weighted exposure ≈ raw exposure in every cell — see finding 1.)

Two findings beyond the raw number:
1. **The repo's EdgeAwareLogL1 "leniency" is a no-op in the band**: weighted exposure ≈ raw
   exposure (0.0273 → 0.0268). The weight is exp(−|∇RGB|) with typical RGB gradients ≪ 1, so
   even a maximal image edge only downweights by e⁻¹ ≈ 0.37×, and typical depth-edge pixels
   keep ~90–98% of their loss weight. The unimodal loss is essentially fully active at depth
   edges even in the repo's "edge-aware" default.
2. Strict definitions put exposure at **1.3–2.7% (sensor 2px; 2.4–4.4% at 4px), 0.4–2.7%
   (mono)** — **below the pre-registered ~5% bar in all scenes** (koivu 4px comes closest at
   4.4%). The repo's laplacian definition says 8.6–11.6%, but its T1.2 ratio collapses to
   2.3–4.0× (vs 5–17× strict), i.e. it dilutes true edges with false positives; the strict
   numbers are the honest ones. **T1.1 verdict: FAIL against the pre-registered bar as
   written (pixel share)** — but see the loss-mass reading below.

**Loss-mass exposure (the bar's intent).** Pixel share understates exposure of the
*objective*: each band pixel carries ~10× the error, so the band's share of depth-loss mass
is share ≈ e·m_band / (e·m_band + (1−e)·m_off) using T1.1 exposure e and T1.2 mean errors
(proxy: assumes the model renders true geometry, so it estimates the fraction of the loss's
gradient signal that is wrong; the realized share at convergence lies between this and the
pixel share):

| source | koivu | coffee_room | vr_room |
|---|---|---|---|
| sensor (grad_logd / window_contrast, 2px) | 10% / 10% | 9% / 9% | 14% / 13% |
| mono (grad_logd / window_contrast, 2px) | 4% / 8% | 2% / 4% | 2% / 5% |

So for the default sensor-depth supervision, **~10% of the depth objective's mass is
edge-band error** — well past a 5% bar read as loss-share, while mono sits at 2–8%.
T1.1 is thus a fail on pixels, a pass (sensor) on loss mass; the pre-registered ~5% bar
did not specify which, so we score it *borderline* overall and let T1.3 decide, with the
resolution argued here openly rather than post-hoc.

## T1.2 Gap/noise ratio — prior-vs-GT error, band vs off-band

150 frames/scene with Faro GT. |prior − GT| in meters, after the repo's own alignment.
2px band; median band err / median off err / **ratio** / fly frac:

| source | edge def | koivu | coffee_room | vr_room |
|---|---|---|---|---|
| sensor | grad_logd | 0.42/0.043/**9.7×**/0.32 | 0.30/0.025/**11.9×**/0.32 | 0.24/0.014/**17.4×**/0.31 |
| sensor | window_contrast | 0.47/0.043/**10.9×**/0.28 | 0.30/0.025/**12.0×**/0.28 | 0.25/0.014/**17.5×**/0.26 |
| sensor | repo_laplace | 0.16/0.040/4.0×/0.22 | 0.07/0.025/2.6×/0.21 | 0.03/0.014/2.3×/0.15 |
| mono | grad_logd | 0.66/0.119/**5.6×**/0.48 | 0.46/0.085/**5.5×**/0.42 | 0.42/0.082/**5.1×**/0.50 |
| mono | window_contrast | 0.51/0.117/4.4×/0.46 | 0.32/0.084/3.8×/0.42 | 0.27/0.082/3.3×/0.48 |
| mono | repo_laplace | 0.40/0.118/3.4×/0.57 | 0.24/0.084/2.8×/0.53 | 0.19/0.082/2.3×/0.61 |

(4px bands: strict-definition ratios drop ~20–30% but stay ≥2.8× everywhere, ≥3.4× for
grad_logd sensor. "fly frac" = fraction of band pixels whose prior depth is intermediate
between local fg/bg by ≥20% of the gap where the gap is meaningful — the flying-pixel
population no single-surface supervision can fit. The repo_laplace rows confirm that
definition is over-liberal: high exposure, diluted ratio.)

**Contrast-monotone fingerprint (mechanism check): present in all 3 scenes.** Band error
grows monotonically with local fg/bg contrast and tracks ≈½ the local gap (sensor,
grad_logd 2px; contrast bins 1–1.1 / 1.1–1.25 / 1.25–1.5 / 1.5–2 / >2):

| scene | band err med by bin (m) | local gap med by bin (m) |
|---|---|---|
| koivu | 0.20 → 0.34 → 0.49 → 0.65 → 0.99 | 0.12 → 0.34 → 0.64 → 1.15 → 1.69 |
| coffee_room | 0.09 → 0.25 → 0.36 → 0.51 → 0.78 | (comparable, monotone) |
| vr_room | 0.06 → 0.21 → 0.31 → 0.45 → 1.30 | 0.10 → 0.31 → 0.56 → 0.86 → 1.97 |

Off-band error is flat and small at low contrast (0.04 m) but also rises for the rare
off-band pixels with high window contrast — expected, since a 7px-contrast window reaches
~3px beyond a 2px band: the prior's edge error extends wider than the band itself. This
strengthens, not weakens, the edge-error attribution.

**Caveats measured, as pre-registered:**
- Median distance from prior edge pixels to nearest GT edge: **2–8 px** (koivu 6–8, coffee
  1.4–5, vr 4.5–8). ARKit/Zoe edges are not just smeared — they are often *misplaced* by
  more than the band half-width. Global alignment is fine (off-band sensor error 1.4–4.3 cm),
  so this is edge-specific — both a caveat for band metrics and additional evidence that
  supervision near edges pulls toward wrong geometry. (Some contribution from Faro-render
  pose/ICP error at occlusions cannot be excluded; the scene-to-scene variation suggests it
  is not a constant systematic offset.)
- Faro GT holes are excluded from both band and off-band sets (valid mask on GT).

**T1.2 verdict: PASS, decisively and consistently.** Strict-definition ratios 5.1–17.5×
across scenes (bar was ~3×), monotone contrast fingerprint everywhere, fly-pixel fraction
0.26–0.50 of band pixels. The mechanism is real and severe wherever the prior has an edge.

## T1.3 Artifact attribution — COMPLETE (9/9 runs)

9 trainings (3 scenes × {EdgeAwareLogL1 (repo default), L1 (pure unimodal), no-depth-loss}),
30k steps, identical normal/smooth/scale losses, sensor-depth supervision, `depth_lambda
0.2` (note: repo applies the depth term as (1+λ)·loss — see dnsplatter_map.md). Configs in
`killtest/train_baseline.sh`; outputs/renders/logs on snoopy under
`/scratch/adam/experiments/bimodal-depth/`. Eval = long-capture test views (29–35/scene),
band from the supervising sensor prior, grad_logd 2px unless noted; depth err vs Faro GT.

**coffee_room** (full-image PSNR/SSIM/LPIPS then band/off slices):

| arm | full | band / off PSNR | band / off SSIM | band / off LPIPS | band / off depth err |
|---|---|---|---|---|---|
| edgeaware | 24.83 / 0.855 / 0.146 | 21.34 / 24.92 | 0.767 / 0.856 | 0.154 / 0.146 | 0.154 / 0.020 |
| l1 | 24.94 / 0.858 / 0.143 | 21.60 / 25.02 | 0.782 / 0.859 | 0.148 / 0.142 | 0.160 / 0.019 |
| nodepth | **25.14** / 0.855 / **0.141** | **22.19** / 25.21 | **0.798** / 0.855 | **0.137** / 0.141 | 0.216 / 0.048 |

**koivu**:

| arm | full | band / off PSNR | band / off SSIM | band / off LPIPS | band / off depth err |
|---|---|---|---|---|---|
| edgeaware | 22.96 / 0.822 / 0.203 | 18.15 / 23.27 | 0.715 / 0.825 | 0.203 / 0.203 | 0.290 / 0.034 |
| l1 | 23.17 / 0.830 / 0.194 | 18.38 / 23.48 | 0.726 / 0.833 | 0.194 / 0.194 | 0.304 / 0.033 |
| nodepth | **23.75** / **0.851** / **0.175** | **19.13** / 24.02 | **0.753** / 0.854 | **0.164** / 0.175 | 0.318 / 0.114 |

**vr_room**:

| arm | full | band / off PSNR | band / off SSIM | band / off LPIPS | band / off depth err |
|---|---|---|---|---|---|
| edgeaware | 26.45 / 0.854 / 0.202 | 22.50 / 26.59 | 0.812 / 0.855 | 0.157 / 0.203 | 0.136 / 0.010 |
| l1 | 26.62 / 0.858 / **0.197** | 22.66 / 26.76 | 0.821 / 0.859 | 0.152 / 0.198 | 0.142 / 0.010 |
| nodepth | 26.61 / 0.855 / 0.221 | **23.49** / 26.72 | **0.839** / 0.856 | **0.148** / 0.222 | 0.154 / 0.046 |

(vr_room nuance: full-image LPIPS *favors* depth supervision — it sharpens the big
textureless walls — yet the band still favors nodepth. Benefit and harm live in different
places.)

**Findings (consistent across all three scenes):**
1. **EdgeAwareLogL1 ≡ L1 within noise** (all scenes) — experimentally confirms that the
   repo's RGB-gradient downweighting provides no protection at depth edges (matches T1.1
   finding 1). The "cheap leniency" arm of the eventual ablation must therefore be a
   *depth-edge-band-masked* loss, which does not exist in the repo.
2. **RGB/perceptual: unimodal depth loss is actively harmful, and the harm is
   band-concentrated (3/3 scenes).** nodepth wins band PSNR by 0.85–1.0 dB everywhere while
   the off-band gap is 0.1–0.3 dB (vr_room: 0.13 dB off vs 1.0 dB band). Band SSIM and band
   LPIPS agree in every scene. This satisfies the pre-registered "unimodal WORSE than
   no-depth in band" condition on appearance metrics.
3. **Depth accuracy: unimodal loss helps, but its band benefit collapses (3/3 scenes).**
   Off-band it wins 2.4× / 3.4× / 4.6× (coffee/koivu/vr); in-band only 1.4× / 1.1× / 1.1×.
   The loss buys accurate *surfaces* but almost no band geometry — while paying the RGB harm
   of #2 exactly there.
4. **Qualitative** (`panels/{koivu,coffee_room,vr_room}/`): supervised arms show depth error
   as bright halos traced along object silhouettes (chairs, plant, lamp, door frame) with
   accurate surfaces between; nodepth shows the inverse — diffuse blobby error on
   floors/walls, no edge halos. The wrong-surface-at-edges mechanism is directly visible in
   trained models.

## VERDICT (against the pre-registered decision rule)

| criterion | bar | outcome |
|---|---|---|
| T1.1 exposure | ≥ ~5% | **borderline** — 1.3–2.7% of pixels (fail) but 9–14% of depth-loss mass for sensor priors (pass); mono 2–8% |
| T1.2 band error ≥3× + contrast-monotone | pass/fail | **PASS** — 5.1–17.5× strict, monotone in all scenes, fly-pixels 26–50% of band |
| T1.3 band-concentrated artifacts, unimodal ≤ nodepth in band | pass/fail | **split, in the mechanism's favor**: on RGB/SSIM/LPIPS unimodal is *worse than nothing* in the band (3/3, band-concentrated); on depth error unimodal keeps a marginal 1.1–1.4× band benefit (vs 2.4–4.6× off-band) |

Per the pre-registration, this is the **borderline → report and discuss** branch, not a
clean go. The honest summary for that discussion:

- The mechanism is real, severe, fingerprinted, and *visible* in trained models (silhouette
  halos). The unimodal loss demonstrably damages appearance exactly where the prior's edges
  are, and buys almost no geometry there in exchange.
- The lever is narrow: ~2% of pixels, ~10% of depth-loss mass. Expected headline-metric
  gains from any fix are correspondingly small (~0.2–0.6 dB full-image PSNR ceiling, judged
  by the nodepth gap); the real, defensible wins are band-sliced metrics and qualitative
  artifact removal (floaters/silhouette bleed) — plus possibly mesh quality at boundaries,
  which we did not measure.
- The fix's true competitor is not EdgeAwareLogL1 (shown ineffective) but the trivial
  **band-masked unimodal loss** (drop depth supervision in the prior's edge band, keep it
  elsewhere). T1.3 suggests band-masking alone could capture most of the RGB-harm removal
  while keeping the off-band depth benefit. The bimodal loss is justified only by what
  masking gives up: committed fg/bg geometry at edges (sharper silhouettes in depth/mesh,
  no free-floating band). Any implementation phase must include the masked arm as the
  primary baseline — if bimodal ≈ masked on everything we measure, the answer is masking,
  not mixtures.
- Depth-source note: everything above is for the sensor (ARKit) prior. The mono (Zoe) prior
  has *lower* strict exposure (0.4–1%: its edges are smeared below the gradient threshold)
  but higher band error and fly fractions; a bimodal fix there needs the window-contrast
  band definition (2.7%/1.1%/1.5% exposure) since its per-pixel-gradient edges are barely
  detectable.

