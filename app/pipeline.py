## app/pipeline.py

import os
import contextlib
from functools import lru_cache
from typing import List, Optional

import torch
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from consistency import install_csa, uninstall_csa, build_lpa_latents


@lru_cache(maxsize=1)
def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_pipeline() -> StableDiffusionPipeline:
    """Load a diffusion pipeline once and cache it. Prefers SD‑Turbo, then SD‑1.5."""
    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    candidates = [
        "stabilityai/sd-turbo",
        "runwayml/stable-diffusion-v1-5",
    ]

    last_err = None
    for name in candidates:
        try:
            print(f"⏳ Loading model: {name} on {device} ({dtype})")
            pipe = StableDiffusionPipeline.from_pretrained(name, torch_dtype=dtype, safety_checker=None)
            # Fast scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            if device.type == "cuda":
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe.to(device)
                try:
                    pipe.unet = torch.compile(pipe.unet)  # PyTorch 2.x
                except Exception:
                    pass
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
            else:
                pipe.to(device)

            print("🎨 Model ready")
            return pipe
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
            last_err = e
    raise RuntimeError(f"Could not load any model. Last error: {last_err}")


def _enhance_color(path: str, factor: float = 1.5) -> None:
    img = Image.open(path).convert("RGB")
    ImageEnhance.Color(img).enhance(factor).save(path, format="PNG")


def _is_turbo(pipe: StableDiffusionPipeline) -> bool:
    rep = getattr(pipe, "_internal_dict", {})
    name = rep.get("_name_or_path", "")
    return "sd-turbo" in str(name).lower()


def get_backend_info() -> str:
    pipe = get_pipeline()
    return getattr(pipe, "_internal_dict", {}).get("_name_or_path", "unknown")


def generate_batch(
    prompts: List[str],
    seeds: List[int],
    height: int = 384,
    width: int = 384,
    num_inference_steps: int = 10,
    guidance_scale: float = 5.0,
    color_boost: float = 1.5,
    use_csa: bool = False,
    csa_rate: float = 0.5,
    use_lpa: bool = False,
    lpa_strength: float = 0.5,
) -> List[str]:
    """Batch generation with optional CSA (token sharing) and LPA‑lite (shared low‑freq latents)."""
    assert len(prompts) == len(seeds), "prompts and seeds must match in length"
    n = len(prompts)

    pipe = get_pipeline()
    device = _get_device()
    dtype = next(pipe.unet.parameters()).dtype

    # Turbo prefers very low steps/guidance
    if _is_turbo(pipe):
        num_inference_steps = min(num_inference_steps, 6)
        guidance_scale = 0.0

    # Torch generators per sample
    gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

    # LPA‑lite latents (shared low‑freq noise)
    latents = None
    if use_lpa:
        latents = build_lpa_latents(batch=n, height=height, width=width, dtype=dtype,
                                    device=device, generators=gens, alpha=float(lpa_strength))

    # CSA attention patch
    patched = False
    if use_csa and n > 1:
        try:
            install_csa(pipe.unet, sample_rate=float(csa_rate))
            patched = True
        except Exception as e:
            print(f"[CSA] Failed to install CSA, continuing without it: {e}")

    os.makedirs("outputs", exist_ok=True)

    # Inference context
    autocast_ctx = torch.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
    with torch.inference_mode(), autocast_ctx:
        out_images = pipe(
            prompts,
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gens,
            latents=latents,
        ).images

    if patched:
        try:
            uninstall_csa(pipe.unet)
        except Exception:
            pass

    paths: List[str] = []
    for i, img in enumerate(out_images, start=1):
        out_path = os.path.join("outputs", f"panel_{i}_seed{seeds[i-1]}_{height}x{width}.png")
        img.save(out_path)
        if color_boost and color_boost != 1.0:
            _enhance_color(out_path, color_boost)
        paths.append(out_path)

    return paths
