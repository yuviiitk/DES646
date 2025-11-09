## app/prompting.py

from typing import List

_STYLE_MAP = {
    "realistic": "realistic photography, cinematic lighting, ultra-detailed, vibrant",
    "anime": "anime style, detailed line art, vibrant tones, expressive lighting, digital anime art",
    "comic": "comic book illustration, bold outlines, halftone shading, action style",
    "cartoon": "cartoon style, playful, colorful, simple shapes, exaggerated expressions",
    "art": "oil painting style, artistic brush strokes, textured canvas look",
    "visual art": "concept art style, dramatic lighting, creative composition, painterly effect",
}


def build_prompts(panel_descriptions: List[str], style: str) -> List[str]:
    style_prompt = _STYLE_MAP.get(style.lower(), _STYLE_MAP["realistic"])  # default
    prompts = [f"{desc}, {style_prompt}, vibrant colors, digital art style" for desc in panel_descriptions]
    return prompts