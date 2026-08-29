"""Single source of truth for all kill-test band definitions and thresholds.

Every threshold used by T1.1/T1.2/T1.3 lives here so conclusions can be checked
at multiple band widths / edge definitions without hunting through scripts.
"""

CONFIG = {
    # --- edge definitions on the depth prior ---
    # ours: || grad log D || > thr  (forward differences)
    "grad_logd_threshold": 0.1,
    # repo (regularization_strategy.find_edges): laplacian(1/(D+eps)) > thr, signed
    "repo_laplacian_threshold": 0.01,
    # smear-robust: local window contrast d_bg/d_fg > thr (catches edges a smooth
    # mono prior spreads over many pixels, where per-pixel gradients stay small)
    "window_contrast_threshold": 1.25,
    # --- band widths (px dilation of the edge map); report all ---
    "band_dilations": [2, 4],
    # --- validity ---
    "depth_tolerance": 0.1,  # meters; matches DNRegularization depth_tolerance
    "max_depth": 15.0,  # meters; drop absurd values
    # --- local fg/bg structure (T1.2) ---
    "contrast_window": 7,  # px, window for local fg/bg depth extraction
    "fg_percentile": 10,  # d_fg = p10 of valid prior depths in window
    "bg_percentile": 90,  # d_bg = p90
    "contrast_bins": [1.0, 1.1, 1.25, 1.5, 2.0, 1e9],  # bins of d_bg/d_fg
    # flying pixel: prior depth strictly between local fg and bg by this margin
    "intermediate_margin": 0.2,  # fraction of (d_bg - d_fg)
    # only count flying pixels where the local gap is meaningful:
    "min_gap_abs": 0.1,  # meters
    "min_gap_rel": 0.05,  # fraction of d_fg
    # --- frame subsampling (statistics only, keeps runtime sane) ---
    "max_frames_per_scene": 150,
    # ================= phase 2: bimodal hypothesis extraction =================
    # band definition per prior type (mono's smeared edges are invisible to
    # grad_logd — diagnostic T1.1)
    "hyp_band_def": {"sensor": "grad_logd", "mono": "window_contrast"},
    "hyp_band_dilation": 2,  # px; 4 as sensitivity
    # two-surface hypotheses: percentiles of INVERSE depth in a WIN x WIN window
    # (robust to flying pixels; never raw min/max)
    "hyp_window": 21,  # chosen on koivu only (hyp_sweep.py: cov 54.3%, min_err 0.098 vs 47.3%/0.127 @13)
    "hyp_window_sweep": [13, 17, 21],
    "hyp_fg_percentile": 10,  # of inverse depth -> far side; see extraction code
    "hyp_bg_percentile": 90,
    # degenerate gate: fall back to unimodal where the local gap is not
    # meaningful (reuses min_gap_abs / min_gap_rel above)
    # --- output-side metrics (extended t13 eval) ---
    "commit_tolerance": 0.2,  # fraction of local GT gap
}
