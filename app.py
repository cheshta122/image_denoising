"""Streamlit demo for Adaptive Multi-Scale Edge-Aware CT Image Denoising."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from ct_denoising.data import demo_phantom, normalize_image
from ct_denoising.pipeline import SingleImageConfig, run_pipeline
from ct_denoising.metrics import metrics_summary_table

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CT Denoising — Adaptive Multi-Scale",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Adaptive Multi-Scale Edge-Aware CT Image Denoising")
st.caption(
    "Combines multi-scale edge detection, Gaussian local variance estimation, "
    "texture-aware thresholding, and firm (semi-soft) threshold in the Wavelet/Shearlet domain."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Parameters")

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

    run_btn = st.button("▶ Run Denoising", type="primary", use_container_width=True)

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

    # ── Image grid ──
    st.subheader("Visual Comparison")
    results = output["results"]

    # Display order: original, noisy, baselines sorted by PSNR, proposed last
    method_order = sorted(
        [k for k in results if not k.startswith("adaptive_")],
        key=lambda k: output["metrics"].get(k, {}).get("psnr", 0.0),
    )
    display_order = ["noisy"] + method_order + [adaptive_name]

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
    st.subheader("📊 Full Metrics Table")
    st.code(output["metrics_table"], language=None)

    # ── Method explanation ──
    with st.expander("📖 What makes this method different?"):
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
    st.info("👈 Configure parameters in the sidebar and click **Run Denoising** to start.")
    st.image(
        demo_phantom(256).image,
        caption="Enhanced CT phantom (Shepp-Logan + tissue blobs + bone rings)",
        width=300,
    )
