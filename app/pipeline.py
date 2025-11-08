from diffusers import StableDiffusionPipeline
import torch, os
from PIL import Image, ImageEnhance

def load_model():
    """Load a lightweight, colorful Stable Diffusion model for Codespaces."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"✅ Using device: {device}")

    models = ["stabilityai/sd-turbo", "runwayml/stable-diffusion-v1-5"]
    for model_name in models:
        try:
            print(f"⏳ Loading model: {model_name}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None
            )
            pipe.to(device)
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            print(f"🎨 Successfully loaded model: {model_name}")
            return pipe
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {e}")
    raise RuntimeError("❌ Could not load any model from Hugging Face.")

# Load once globally
pipe = load_model()


def enhance_color(path, factor=1.5):
    """Boost image saturation after generation."""
    image = Image.open(path)
    enhancer = ImageEnhance.Color(image)
    colorful_image = enhancer.enhance(factor)
    colorful_image.save(path)


def generate_image(prompt, panel_no=1, style="realistic", seed=None):
    """Generate one storyboard panel image with a unique or provided seed."""
    os.makedirs("outputs", exist_ok=True)

    style_map = {
        "realistic": "realistic photography, cinematic lighting, ultra-detailed, vibrant",
        "anime": "anime style, detailed line art, vibrant tones, expressive lighting, digital anime art",
        "comic": "comic book illustration, bold outlines, halftone shading, action style",
        "cartoon": "cartoon style, playful, colorful, simple shapes, exaggerated expressions",
        "art": "oil painting style, artistic brush strokes, textured canvas look",
        "visual art": "concept art style, dramatic lighting, creative composition, painterly effect"
    }

    style_prompt = style_map.get(style.lower(), "realistic")
    final_prompt = f"{prompt}, {style_prompt}, vibrant colors, digital art style"

    # Handle seeding for reproducibility or variation
    if seed is None:
        seed = panel_no * 42
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    image = pipe(
        final_prompt,
        height=384,
        width=384,
        num_inference_steps=10,
        guidance_scale=5,
        generator=generator
    ).images[0]

    path = f"outputs/panel_{panel_no}.png"
    image.save(path)
    enhance_color(path)
    print(f"✅ Saved unique {style} panel: {path} (seed={seed})")
    return path
