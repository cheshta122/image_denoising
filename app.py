"""Streamlit demo for Adaptive Multi-Scale Edge-Aware CT Image Denoising."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ct_denoising.data import demo_phantom, normalize_image
from ct_denoising.pipeline import SingleImageConfig, run_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CT Denoising — Adaptive Multi-Scale",
    layout="wide",
)

st.title("Adaptive Multi-Scale Edge-Aware CT Image Denoising")
st.caption(
    "Combines multi-scale edge detection, Gaussian local variance estimation, "
    "texture-aware thresholding, and firm (semi-soft) threshold in the Wavelet/Shearlet domain."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Parameters")

    uploaded = st.file_uploader(
        "Upload a CT image (PNG / JPEG / TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    st.subheader("Noise Model")
    noise_type = st.selectbox(
        "Noise type",
        ["mixed", "poisson", "gaussian", "speckle"],
        index=0,
        help="Mixed = Poisson (quantum) + Gaussian (readout) — most realistic for CT",
    )
    poisson_peak = st.slider(
        "Poisson peak photons",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        help="Lower = noisier. Clinical low-dose CT ≈ 20–50",
    )
    gaussian_sigma = st.slider(
        "Gaussian σ (readout noise)",
        min_value=0.01,
        max_value=0.15,
        value=0.05,
        step=0.01,
    )

    st.subheader("Threshold Settings")
    threshold = st.slider(
        "Base threshold",
        min_value=0.01,
        max_value=0.20,
        value=0.06,
        step=0.005,
        help="Starting threshold before adaptive scaling",
    )
    threshold_mode = st.selectbox(
        "Threshold mode",
        ["firm", "soft", "hard"],
        index=0,
        help="Firm = semi-soft (recommended). Preserves signal better than pure soft.",
    )
    edge_method = st.selectbox(
        "Edge detection",
        ["multiscale", "sobel", "log"],
        index=0,
        help="Multiscale combines fine (Sobel) and coarse (LoG) edges",
    )

    st.subheader("Advanced")
    use_texture = st.checkbox(
        "Texture-aware thresholding",
        value=True,
        help="Apply different threshold multipliers to homogeneous/textured/edge regions",
    )
    prefer_shearlet = st.checkbox(
        "Prefer Shearlet (if available)",
        value=True,
        help="Falls back to Wavelet if PyShearLab is not installed",
    )
    seed = st.number_input("Random seed", value=42, min_value=0)

    run_btn = st.button("Run Denoising", type="primary", width="stretch")

# ── Main panel ───────────────────────────────────────────────────────────────
if run_btn:
    # Load image
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("L")
        image_array = normalize_image(np.array(pil_img))
    else:
        st.info("No image uploaded — using the enhanced CT phantom.")
        image_array = demo_phantom(256).image

    config = SingleImageConfig(
        noise_type=noise_type,
        gaussian_sigma=float(gaussian_sigma),
        poisson_peak=float(poisson_peak),
        threshold=float(threshold),
        threshold_mode=threshold_mode,
        edge_method=edge_method,
        seed=int(seed),
        prefer_shearlet=prefer_shearlet,
        use_texture=use_texture,
    )

    with st.spinner("Running pipeline…"):
        output = run_pipeline(image_array, config)

    backend = output["backend"]
    adaptive_name = output["adaptive_method"]

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Backend", backend.upper())
    col2.metric("PSNR (Proposed)", f"{output['psnr']:.2f} dB")
    col3.metric("SSIM (Proposed)", f"{output['ssim']:.3f}")
    col4.metric("RMSE (Proposed)", f"{output['rmse']:.4f}")

    # ── Result section: noisy image, denoised outputs, and transform comparison ──
    st.subheader("Results: Noisy Image vs Denoised Output")
    results = output["results"]

    noisy_metrics = output["metrics"].get("noisy", {})
    proposed_metrics = output["metrics"].get(adaptive_name, {})
    noisy_col, denoised_col = st.columns(2)
    noisy_col.image(
        output["noisy"],
        caption=(
            "Noisy CT Image\n"
            f"PSNR {noisy_metrics.get('psnr', 0):.2f} dB | "
            f"SSIM {noisy_metrics.get('ssim', 0):.3f}"
        ),
        clamp=True,
    )
    denoised_col.image(
        output["denoised"],
        caption=(
            f"Best Denoised Output ({adaptive_name.replace('_', ' ').title()})\n"
            f"PSNR {proposed_metrics.get('psnr', 0):.2f} dB | "
            f"SSIM {proposed_metrics.get('ssim', 0):.3f}"
        ),
        clamp=True,
    )

    noisy_vs_denoised = pd.DataFrame(
        [
            {
                "Image / Method": "Noisy CT Image",
                "PSNR (dB)": noisy_metrics.get("psnr"),
                "SSIM": noisy_metrics.get("ssim"),
                "RMSE": noisy_metrics.get("rmse"),
                "Observation": "Noise is present before denoising",
            },
            {
                "Image / Method": adaptive_name.replace("_", " ").title(),
                "PSNR (dB)": proposed_metrics.get("psnr"),
                "SSIM": proposed_metrics.get("ssim"),
                "RMSE": proposed_metrics.get("rmse"),
                "Observation": "Noise reduced while preserving structures",
            },
        ]
    )
    st.dataframe(noisy_vs_denoised, width="stretch", hide_index=True)

    st.subheader("Wavelet and Shearlet Results")
    available_backends = output.get("available_transform_backends", [])
    wavelet_tab, shearlet_tab = st.tabs(["Wavelet Result", "Shearlet Result"])

    def _method_metrics_table(method_names: list[str]) -> pd.DataFrame:
        rows = []
        for method in method_names:
            values = output["metrics"].get(method)
            if values is None:
                continue
            rows.append(
                {
                    "Method": method.replace("_", " ").title(),
                    "PSNR (dB)": values.get("psnr"),
                    "SSIM": values.get("ssim"),
                    "RMSE": values.get("rmse"),
                    "GradCorr": values.get("grad_corr"),
                    "NRR": values.get("nrr"),
                }
            )
        return pd.DataFrame(rows)

    with wavelet_tab:
        st.markdown("**Wavelet-domain denoising result**")
        wavelet_cols = st.columns(2)
        wavelet_cols[0].image(
            results["wavelet_fixed"],
            caption="Wavelet Fixed Threshold",
            clamp=True,
        )
        wavelet_cols[1].image(
            results["adaptive_wavelet"],
            caption="Adaptive Wavelet",
            clamp=True,
        )
        st.dataframe(
            _method_metrics_table(["wavelet_fixed", "adaptive_wavelet"]),
            width="stretch",
            hide_index=True,
        )

    with shearlet_tab:
        if "adaptive_shearlet" in results:
            if "shearlet-style fallback" in backend:
                st.markdown(
                    "**Shearlet-style denoising result**  \n"
                )
            else:
                st.markdown("**Shearlet-domain denoising result**")
            shearlet_cols = st.columns(2)
            shearlet_cols[0].image(
                results["shearlet_fixed"],
                caption="Shearlet Fixed Threshold" if "shearlet-style fallback" not in backend else "Shearlet-Style Fixed Threshold",
                clamp=True,
            )
            shearlet_cols[1].image(
                results["adaptive_shearlet"],
                caption="Adaptive Shearlet" if "shearlet-style fallback" not in backend else "Adaptive Shearlet-Style",
                clamp=True,
            )
            st.dataframe(
                _method_metrics_table(["shearlet_fixed", "adaptive_shearlet"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "Shearlet backend is not available in this environment. "
                "Install PyShearLab to generate Shearlet results; Wavelet results are shown above."
            )

    st.subheader("Wavelet vs Shearlet Comparison Table")
    comparison_methods = ["adaptive_wavelet"]
    if "adaptive_shearlet" in output["metrics"]:
        comparison_methods.append("adaptive_shearlet")
    transform_comparison = _method_metrics_table(comparison_methods)
    st.dataframe(transform_comparison, width="stretch", hide_index=True)

    # ── Full image grid ──
    st.subheader("Complete Visual Comparison")

    # Display order: original, noisy, baselines sorted by PSNR, proposed last
    method_order = sorted(
        [k for k in results if not k.startswith("adaptive_")],
        key=lambda k: output["metrics"].get(k, {}).get("psnr", 0.0),
    )
    adaptive_methods = [k for k in ["adaptive_wavelet", "adaptive_shearlet"] if k in results]
    display_order = ["noisy"] + method_order + adaptive_methods

    panels_per_row = 4
    all_panels = [("Original (clean)", image_array)] + [
        (k, output["results"][k] if k != "noisy" else output["noisy"])
        for k in display_order
    ]

    for row_start in range(0, len(all_panels), panels_per_row):
        cols = st.columns(panels_per_row)
        for col, (label, img) in zip(cols, all_panels[row_start: row_start + panels_per_row]):
            m = output["metrics"].get(label, {})
            caption = label.replace("_", " ").title()
            if m:
                caption += f"\nPSNR {m.get('psnr', 0):.2f} dB | SSIM {m.get('ssim', 0):.3f}"
            col.image(img, caption=caption, clamp=True)

    # ── Metrics table ──
    st.subheader("Full Metrics Table")
    st.code(output["metrics_table"], language=None)

    # ── Method explanation ──
    with st.expander("What makes this method different?"):
        st.markdown(f"""
**Backend:** `{backend}` transform domain

**Key innovations in this pipeline:**

1. **Multi-scale edge detection** — Combines fine (Sobel) and coarse (Laplacian-of-Gaussian) 
   edge maps, computed on a *pre-smoothed* image to prevent noise from corrupting the edge response.

2. **Gaussian-weighted local variance** — Unlike the standard box-filter variance, Gaussian 
   weighting produces spatially coherent estimates that better reflect tissue heterogeneity.

3. **Texture-aware threshold multipliers** — Classifies each pixel region into:
   - Homogeneous (air/uniform tissue) → aggressive denoising (×1.6)
   - Textured (lungs/trabecular bone) → gentle denoising (×0.75)  
   - Edge (structure boundaries) → minimal denoising (×0.35)

4. **Firm (semi-soft) threshold** — Interpolates between hard and soft threshold:
   - Below T_low → zero (like hard: kills definite noise)
   - Above T_high → kept as-is (no shrinkage bias on strong signals)
   - Between → linearly ramped (smooth, avoids Gibbs ringing)

5. **BayesShrink per-band noise estimation** — Estimates actual noise level from 
   the finest wavelet subband (MAD estimator), scales threshold accordingly.

6. **Per-level threshold scaling** — Coarser wavelet levels (more signal content) 
   get lower thresholds; finer levels (noisier) get higher thresholds.
        """)
else:
    st.info("Configure parameters in the sidebar and click **Run Denoising** to start.")
    st.image(
        demo_phantom(256).image,
        caption="Enhanced CT phantom (Shepp-Logan + tissue blobs + bone rings)",
        width=300,
    )
