from diffusers import StableDiffusionPipeline
import torch, os

def load_model():
    """Load the Stable Diffusion model once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe.to(device)
    return pipe

# Load model globally (only once)
pipe = load_model()

def generate_image(prompt, panel_no=1):
    """Generate one storyboard panel image."""
    os.makedirs("outputs", exist_ok=True)
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    path = f"outputs/panel_{panel_no}.png"
    image.save(path)
    print(f"✅ Saved: {path}")
    return path
