DES646 - AI-Driven Storyboarding Assistant 🚀

Team Members 👥
- Anjali Maloth (210146)
- Jerry Surakshitha (210472)
- Lohit P Talavar (210564)
- Yuvraj Singh Kharte (211210)
- Priyanshi Meena (240804)

Overview 📌
- Streamlit app that turns a script/scene description into a storyboard.
- Parses text into panel descriptions, builds image prompts, generates images with Diffusers, and offers evaluation + export tools.
- Runs locally; models download once via Hugging Face.

Requirements ⚙️
- Python 3.10+
- Windows, macOS, or Linux
- Optional: NVIDIA GPU with CUDA for faster inference (CPU works but slower)

Quick Start ▶️
- Create and activate a virtual environment:
  - Windows PowerShell: `python -m venv .venv && .\\.venv\\Scripts\\activate`
  - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Launch the app: `streamlit run app/main_app.py`
- The first run downloads models (sd-turbo preferred, falls back to SD 1.5).

How It Works 🧠
- Parsing: splits your script into concise, panel-ready descriptions.
- Prompting: adds style-specific prompt fragments for image generation.
- Generation: uses a Stable Diffusion pipeline (DPMSolver scheduler).
- Consistency: optional training-free techniques to keep panels coherent.
- Evaluation: optional CLIP alignment and identity similarity (OpenCLIP).
- Export: ZIP of PNGs, PDF contact sheet, and a JSON storyboard dump.

End-to-End Pipeline 🔄
- Input (UI):
  - User enters script and selects style and generation settings in `app/main_app.py`.
- Parse script -> panels:
  - `app/parsing.py:parse_story` splits text into sentences/clauses, normalizes, deduplicates, and expands to N panel descriptions.
- Build prompts:
  - `app/prompting.py:build_prompts` maps style to a style fragment and appends it to each panel description.
- Seed strategy:
  - Base seed in session state; per-panel seeds are `base_seed + i` for reproducible, coherent ordering.
- Load diffusion pipeline:
  - `app/pipeline.py:get_pipeline` loads `stabilityai/sd-turbo` or falls back to SD 1.5, sets DPMSolver scheduler, enables slicing/offload, optional `torch.compile`.
- Optional consistency:
  - CSA: `app.consistency.install_csa` wraps self-attention processors to share a subset of K/V tokens across batch items.
  - LPA-lite: `app.consistency.build_lpa_latents` builds latents with shared low-frequency components blended via `alpha`.
- Inference:
  - `app/pipeline.py:generate_batch` creates per-sample `torch.Generator`s from seeds, optionally supplies shared latents, then calls the pipeline with steps/guidance and returns PIL images.
- Post-process & save:
  - Optional color boost via `PIL.ImageEnhance`.
  - Images saved under `outputs/` with seed and size in filenames.
- Preview & per-panel regen:
  - Grid preview via `app/export_utils.make_grid_image`.
  - Single panel regeneration uses the same per-panel seed and prompt to keep layout consistency.
- Evaluation (optional):
  - `app/eval.py:compute_clip_alignment` (OpenCLIP) for image-prompt similarity.
  - `app/eval.py:compute_identity_similarity` for adjacent panel feature similarity.
- Export:
  - ZIP of images, PDF contact sheet (`app/export_utils.make_pdf_bytes`), and storyboard JSON (`pydantic` model dump).

Using The App 🖱️
- Script: paste a short scene or paragraph in the sidebar.
- Style: choose from realistic, anime, comic, cartoon, art, visual art.
- Generation controls:
  - Image size: from 384 to 640 px
  - Steps: 2-50 (sd-turbo works best with very few steps)
  - Guidance: 0.0-12.0 (sd-turbo prefers 0.0)
  - Color boost: simple saturation/contrast punch after generation
- Consistency (training-free):
  - CSA (StoryDiffusion): shares a sample of K/V attention tokens across batch items to help maintain character/scene consistency.
  - LPA-lite (Story2Board): blends a shared low-frequency noise base with per-panel noise for softer consistency without training.
- Generate: click "Generate Storyboard" to produce all panels.
- Preview & edit: view a grid, and optionally regenerate any single panel.
- Evaluate (optional): compute CLIP prompt alignment and adjacent identity similarity (requires `open-clip-torch`).
- Export: download all panels (ZIP), a PDF contact sheet, or storyboard JSON.

Outputs 🖼️
- Image files: saved under `outputs/` as `panel_{idx}_seed{seed}_{HxW}.png`.
- Run metadata: `outputs/run_YYYYMMDD_HHMMSS.json` (settings, prompts, files).
- Exports via UI: ZIP, PDF, and JSON storyboard with panel prompts and seeds.

Project Structure 🗂️
- `app/main_app.py` - Streamlit UI and workflow (tabs: Generation, Evaluation, Export)
- `app/parsing.py` - Sentence/clause heuristics to segment script into panels
- `app/prompting.py` - Style mapping and prompt construction
- `app/pipeline.py` - Diffusers pipeline setup and batch image generation
- `app/consistency.py` - CSA attention wrapper and LPA-lite shared latents
- `app/export_utils.py` - Grid image creation and PDF export helpers
- `app/eval.py` - OpenCLIP prompt alignment and identity similarity
- `requirements.txt` - Python dependencies

Models & Performance ⚡
- The app tries `stabilityai/sd-turbo` first, then `runwayml/stable-diffusion-v1-5`.
- sd-turbo:
  - Use very low steps (<= 6) and guidance 0.0 for best speed/quality tradeoffs.
- GPU vs CPU:
  - GPU is recommended; CPU is supported but slower. Reduce image size/steps if you run out of memory.

Troubleshooting 🛠️
- Torch/CUDA installation:
  - If you need GPU support, install a CUDA-enabled PyTorch build from pytorch.org per your CUDA version.
- Hugging Face model access:
  - If SD 1.5 access is restricted, ensure your Hugging Face account accepted the license and is logged in (`huggingface-cli login`) or set `HUGGINGFACE_HUB_TOKEN`.
- Out of memory (OOM):
  - Lower image size, reduce steps, disable CSA/LPA-lite, or close other GPU apps.
- Slow generation:
  - Prefer sd-turbo, reduce size/steps, and ensure GPU is used if available.
- First run delays:
  - Model weights must download once; subsequent runs load from cache.

Development 🧪
- Code is organized by function; modify modules under `app/` as needed.
- Run the app locally with `streamlit run app/main_app.py`.
- Keep changes minimal and focused; avoid introducing heavy frameworks.

Credits 🙌
- Built with Streamlit, Hugging Face Diffusers/Transformers, PyTorch, and OpenCLIP.
