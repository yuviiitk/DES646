import streamlit as st
import os
import sys

# Ensure correct path (so we can import pipeline)
sys.path.append(os.path.dirname(__file__))
from pipeline import generate_image

st.set_page_config(page_title="AI Storyboarding Assistant", page_icon="🎬")

st.title("🎬 AI-Driven Storyboarding Assistant")
st.write("Enter a short script and generate storyboard panels using AI.")


script = st.text_area("Enter your script or scene description:", height=150)
num_panels = st.slider("Number of panels", 1, 6, 3)

if st.button("Generate Storyboard"):
    if not script.strip():
        st.warning("Please enter a script before generating.")
    else:
        st.info("Generating panels... this might take a minute.")
        os.makedirs("../outputs", exist_ok=True)
        for i in range(num_panels):
            prompt = f"Storyboard panel {i+1}: {script}"
            path = generate_image(prompt, i+1)
            st.image(path, caption=f"Panel {i+1}")
        st.success("Storyboard generation complete!")
