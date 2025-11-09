import os
import io
import json
import random
import zipfile
from datetime import datetime
from typing import List

import streamlit as st

from models import Storyboard, Panel
from parsing import parse_story
from prompting import build_prompts
from pipeline import generate_batch, get_backend_info
from export_utils import make_grid_image, make_pdf_bytes
from eval import compute_clip_alignment, compute_identity_similarity

st.set_page_config(page_title="🎬 AI Storyboarding Assistant", page_icon="🎨", layout="wide")

st.title("🎬 AI-Driven Storyboarding Assistant")
st.caption("Local, privacy-safe storyboard generation with consistency tools and designer-friendly exports.")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Inputs")
    script = st.text_area(
        "Script or scene description",
        "A person enters a studio, sees a smart lamp that adjusts light, and smiles.",
        height=140,
    )

    style = st.selectbox("Image Style 🎨", ["realistic", "anime", "comic", "cartoon", "art", "visual art"], index=0)
    num_panels = st.slider("Number of panels", 1, 6, 6)

    st.header("Generation")
    fixed_seed = st.checkbox("🔒 Keep visual consistency (fixed base seed)", value=True)
    img_size = st.select_slider("Image size (px)", options=[384, 448, 512, 576, 640], value=384)
    steps = st.slider("Inference steps", min_value=2, max_value=50, value=10)
    guidance = st.slider("Guidance scale", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
    color_boost = st.slider("Color boost", min_value=1.0, max_value=2.0, value=1.5, step=0.1)

    st.header("Consistency (training-free)")
    use_csa = st.checkbox(
        "CSA (StoryDiffusion) — share tokens across panels",
        value=False,
        help="Training-free attention tweak that samples tokens from other panels and concatenates into K/V; preserves text control.",
    )
    csa_rate = st.slider("CSA sampling rate", 0.1, 0.9, 0.5, 0.1)

    use_lpa = st.checkbox(
        "LPA-lite (Story2Board) — shared low-freq latents",
        value=False,
        help="Approximate latent anchoring by blending a shared base noise with per-panel noise.",
    )
    lpa_strength = st.slider("LPA-lite strength (shared noise)", 0.0, 1.0, 0.5, 0.1)

    generate_btn = st.button("🎨 Generate Storyboard", use_container_width=True)

# ---------- Session state ----------
if "base_seed" not in st.session_state:
    st.session_state.base_seed = random.randint(0, 10_000)
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "panel_prompts" not in st.session_state:
    st.session_state.panel_prompts = []
# fingerprint of (script, num_panels) to refresh text areas
if "desc_fp" not in st.session_state:
    st.session_state["desc_fp"] = (None, None)

# ---------- Tabs ----------
T1, T2, T3 = st.tabs(["Generation", "Evaluation", "Export"])

with T1:
    st.subheader("Storyboard Plan")
    storyboard = parse_story(script, num_panels)

    # --- IMPORTANT FIX: refresh per-panel text areas when script or panel count changes ---
    if st.session_state["desc_fp"] != (script, num_panels):
        for p in storyboard.panels:
            st.session_state[f"desc_{p.id}"] = p.description
        st.session_state["desc_fp"] = (script, num_panels)

    # Optional manual reset button
    if st.button("↺ Reset panel descriptions from script"):
        storyboard = parse_story(script, num_panels)
        for p in storyboard.panels:
            st.session_state[f"desc_{p.id}"] = p.description

    # Render editable text areas (backed by session_state so they persist)
    edited_descriptions: List[str] = []
    cols = st.columns(2)
    for i, p in enumerate(storyboard.panels):
        key = f"desc_{p.id}"
        with cols[i % 2]:
            edited = st.text_area(f"Panel {p.id} description", key=key)
            edited_descriptions.append(edited)

    # Build per-panel prompts from the edited descriptions + style
    panel_prompts = build_prompts(edited_descriptions, style)

    # Generate when clicked
    if generate_btn:
        st.session_state.panel_prompts = panel_prompts
        if fixed_seed:
            base_seed = st.session_state.base_seed
        else:
            base_seed = random.randint(0, 10_000)
            st.session_state.base_seed = base_seed

        seeds = [base_seed + i for i in range(num_panels)]
        st.info(
            f"Generating {num_panels} panels • size {img_size} • steps {steps} • guidance {guidance} • base_seed {base_seed}"
        )
        try:
            paths = generate_batch(
                prompts=panel_prompts,
                seeds=seeds,
                height=img_size,
                width=img_size,
                num_inference_steps=steps,
                guidance_scale=guidance,
                color_boost=color_boost,
                use_csa=use_csa,
                csa_rate=csa_rate,
                use_lpa=use_lpa,
                lpa_strength=lpa_strength,
            )
            st.session_state.image_paths = paths
            st.success("✅ Storyboard generation complete!")
        except Exception as e:
            st.error(f"Generation failed: {e}")

    # Preview grid
    if st.session_state.image_paths:
        st.subheader("Preview")
        grid = make_grid_image(st.session_state.image_paths, columns=3)
        st.image(grid, caption="2×3 grid preview (auto-layout)", use_container_width=True)

        # Per-panel quick actions
        st.divider()
        st.write("**Per-panel actions**")
        act_cols = st.columns(3)
        for i, pth in enumerate(st.session_state.image_paths):
            with act_cols[i % 3]:
                if st.button(f"♻️ Regenerate Panel {i+1}", key=f"regen_{i+1}"):
                    # single-panel regen with same seed (consistency)
                    seeds = [st.session_state.base_seed + i]
                    prompts = [st.session_state.panel_prompts[i]]
                    new_path = generate_batch(
                        prompts,
                        seeds,
                        height=img_size,
                        width=img_size,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        color_boost=color_boost,
                        use_csa=False,  # keep CSA off for single image
                        use_lpa=use_lpa,
                        lpa_strength=lpa_strength,
                    )[0]
                    st.session_state.image_paths[i] = new_path
                    st.experimental_rerun()

with T2:
    st.subheader("Evaluation (local)")
    if not st.session_state.image_paths:
        st.info("Generate panels first.")
    else:
        # CLIP alignment
        if st.button("Compute CLIP prompt alignment"):
            scores = compute_clip_alignment(st.session_state.image_paths, st.session_state.panel_prompts)
            st.write({f"Panel {i+1}": float(s) for i, s in enumerate(scores)})

        # Identity similarity (pairwise adjacent, as a proxy)
        if st.button("Compute identity similarity (adjacent panels)"):
            sims = compute_identity_similarity(st.session_state.image_paths)
            st.write({f"({i},{i+1})": float(v) for i, v in enumerate(sims, start=1)})

with T3:
    st.subheader("Exports")
    paths = st.session_state.image_paths
    prompts = st.session_state.panel_prompts
    if not paths:
        st.info("Generate panels first.")
    else:
        # ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, os.path.basename(p))
        zip_buffer.seek(0)
        st.download_button(
            "⬇️ Download All Panels (ZIP)",
            data=zip_buffer,
            file_name="storyboard_panels.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # 2×3 PDF
        pdf_bytes = make_pdf_bytes(paths, captions=[f"Panel {i+1}" for i in range(len(paths))])
        st.download_button(
            "📄 Download PDF (2×3 pages)",
            data=pdf_bytes,
            file_name="storyboard.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # Storyboard JSON
        sb_json = Storyboard(
            title="Storyboard",
            style=style,
            panels=[Panel(id=i + 1, description=prompts[i], seed=st.session_state.base_seed + i) for i in range(len(paths))],
        ).model_dump_json(indent=2)
        st.download_button(
            "📦 Export Storyboard JSON",
            data=sb_json,
            file_name="storyboard.json",
            mime="application/json",
            use_container_width=True,
        )

        # Meta log
        meta = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "backend": get_backend_info(),
            "style": style,
            "size": img_size,
            "steps": steps,
            "guidance": guidance,
            "color_boost": color_boost,
            "base_seed": st.session_state.base_seed,
            "prompts": prompts,
            "images": paths,
        }
        os.makedirs("outputs", exist_ok=True)
        log_path = os.path.join("outputs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        st.caption(f"Saved run meta → {log_path}")
