import streamlit as st
import os, sys, zipfile, io
from PIL import Image
from fpdf import FPDF
import random

# Import the pipeline
sys.path.append(os.path.dirname(__file__))
from pipeline import generate_image

st.set_page_config(page_title="🎬 AI Storyboarding Assistant", page_icon="🎨")

st.title("🎬 AI-Driven Storyboarding Assistant")
st.write("Generate colorful storyboard panels in different styles (2×3 layout).")

# --- Inputs ---
script = st.text_area("Enter your script or scene description:",
                      "A person enters a studio, sees a smart lamp that adjusts light, and smiles.")

style = st.selectbox(
    "Select Image Style 🎨",
    ["realistic", "anime", "comic", "cartoon", "art", "visual art"],
    index=0
)

num_panels = st.slider("Number of panels", 1, 6, 6)

# Seed control for visual consistency
fixed_seed = st.checkbox("🔒 Keep panels visually consistent (same seed for all panels)", value=False)

generate = st.button("🎨 Generate Storyboard")

# --- Generation ---
if generate:
    st.write(f"🧠 Generating {num_panels} {style}-style panels... Please wait ⏳")

    image_paths = []
    base_seed = random.randint(0, 9999)

    for i in range(num_panels):
        with st.spinner(f"Generating panel {i+1} of {num_panels}..."):
            seed = base_seed if fixed_seed else i * 42 + random.randint(0, 1000)
            img_path = generate_image(script, i+1, style, seed=seed)
            image_paths.append(img_path)

    st.success("✅ Storyboard generation complete!")

    # Display in a 2x3 grid
    cols = st.columns(3)
    for idx, img_path in enumerate(image_paths):
        row = idx // 3
        col = idx % 3
        with cols[col]:
            st.image(img_path, caption=f"Panel {idx+1}", use_container_width=True)
        if col == 2 and idx != len(image_paths) - 1:
            cols = st.columns(3)

    # --- ZIP Download ---
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))
    zip_buffer.seek(0)

    st.download_button(
        label="⬇️ Download All Panels (ZIP)",
        data=zip_buffer,
        file_name=f"storyboard_{style}_style.zip",
        mime="application/zip"
    )

    # --- PDF Export ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="🎬 AI Storyboarding Assistant Output", ln=True, align="C")

    for i, img_path in enumerate(image_paths):
        pdf.cell(200, 10, txt=f"Panel {i+1}", ln=True, align="L")
        pdf.image(img_path, x=30, w=150)
        pdf.ln(10)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    st.download_button(
        label="📄 Download as PDF",
        data=pdf_output,
        file_name=f"storyboard_{style}_style.pdf",
        mime="application/pdf"
    )

    st.write("All generated panels are saved in the **outputs/** folder.")
