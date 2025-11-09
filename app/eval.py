## app/eval.py

from typing import List
import torch
from PIL import Image

# Lightweight OpenCLIP evaluation (install open-clip-torch)
try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False


def _load_openclip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer, preprocess, device


def compute_clip_alignment(image_paths: List[str], prompts: List[str]) -> List[float]:
    if not _HAS_OPENCLIP:
        raise RuntimeError("open-clip-torch not installed")
    model, tokenizer, preprocess, device = _load_openclip()
    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in image_paths]).to(device)
    texts = tokenizer(prompts).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(images)
        txt_feat = model.encode_text(texts)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat * txt_feat).sum(dim=-1)
    return sims.detach().float().cpu().tolist()


def compute_identity_similarity(image_paths: List[str]) -> List[float]:
    if not _HAS_OPENCLIP:
        raise RuntimeError("open-clip-torch not installed")
    model, _, preprocess, device = _load_openclip()
    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in image_paths]).to(device)
    with torch.no_grad():
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    sims = []
    for i in range(len(image_paths) - 1):
        sims.append((feats[i] * feats[i+1]).sum().item())
    return sims