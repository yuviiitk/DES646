## app/consistency.py

from typing import List
import torch
import torch.nn.functional as F

# ---- CSA (StoryDiffusion, training‑free) ----
# We implement a lightweight AttentionProcessor wrapper that, for self‑attention blocks,
# concatenates a random subset of K/V tokens from other batch items. If any shape/API
# mismatch occurs (Diffusers versions differ), callers should catch and continue.

class _CSAProcessor(torch.nn.Module):
    def __init__(self, base_attn, sample_rate: float = 0.5):
        super().__init__()
        self.base_attn = base_attn
        self.sample_rate = sample_rate

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Only apply CSA to self‑attention (no encoder_hidden_states)
        if encoder_hidden_states is not None:
            return self.base_attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        # Call the underlying processor to get q,k,v or to compute attention if it encapsulates it.
        # Many Diffusers processors expect this signature and compute inside. We try to intercept
        # K/V tensors via kwargs if provided; otherwise we fallback to plain call.
        try:
            # Some processors pass through precomputed qkv via kwargs; this is best‑effort.
            if "key_states" in kwargs and "value_states" in kwargs:
                key_states = kwargs["key_states"]
                value_states = kwargs["value_states"]
            else:
                # Fallback: let the base processor handle it without CSA
                return self.base_attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

            bsz, seqlen, dim = hidden_states.shape
            # key/value expected shapes: (bsz, heads, seqlen, head_dim) in modern processors
            if key_states.dim() == 4:
                b, h, t, d = key_states.shape
                if b > 1:
                    # sample some tokens from other batch items
                    k_list, v_list = [key_states], [value_states]
                    # gather tokens from other items
                    other = key_states[torch.randperm(b)]
                    take = max(1, int(self.sample_rate * t))
                    other_tokens = other[:, :, :take, :]
                    k_list.append(other_tokens)
                    v_list.append(value_states[torch.randperm(b)][:, :, :take, :])
                    key_states = torch.cat(k_list, dim=2)
                    value_states = torch.cat(v_list, dim=2)
                    kwargs["key_states"], kwargs["value_states"] = key_states, value_states
        except Exception:
            # If anything goes wrong, fall back to base
            return self.base_attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        return self.base_attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)


def install_csa(unet, sample_rate: float = 0.5):
    """Wrap all attention processors with CSA on self‑attention blocks."""
    if not hasattr(unet, "attn_processors"):
        raise RuntimeError("UNet does not expose attn_processors in this Diffusers version.")

    new_procs = {}
    for name, proc in unet.attn_processors.items():
        if "attn1" in name:  # attn1 = self‑attention; attn2 = cross‑attention
            new_procs[name] = _CSAProcessor(proc, sample_rate=sample_rate)
        else:
            new_procs[name] = proc
    unet.set_attn_processor(new_procs)


def uninstall_csa(unet):
    """Remove CSA wrappers, restoring base processors."""
    # unwrap if needed
    new_procs = {}
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, _CSAProcessor):
            new_procs[name] = proc.base_attn
        else:
            new_procs[name] = proc
    unet.set_attn_processor(new_procs)


# ---- LPA‑lite (shared low‑frequency latents) ----
# Build initial latents where low‑frequency components are shared across the batch,
# and high‑frequency components are per‑sample. This approximates latent anchoring.

def _randn_latents(batch, channels, h8, w8, dtype, device, generators: List[torch.Generator]):
    latents = torch.stack([torch.randn((channels, h8, w8), generator=g, device=device, dtype=dtype) for g in generators])
    return latents


def _lowpass_fft(x: torch.Tensor, keep_ratio: float = 0.25) -> torch.Tensor:
    # x: (b, c, h, w)
    b, c, h, w = x.shape
    X = torch.fft.rfftn(x, dim=(2, 3))  # (b,c,h, w//2+1)
    mask = torch.zeros((h, w // 2 + 1), device=x.device, dtype=torch.bool)
    kh, kw = int(h * keep_ratio), int((w // 2 + 1) * keep_ratio)
    mask[:kh, :kw] = True
    mask = mask[None, None].expand(b, c, -1, -1)
    X_low = torch.zeros_like(X)
    X_low[mask] = X[mask]
    x_low = torch.fft.irfftn(X_low, s=(h, w), dim=(2, 3)).real
    return x_low


def build_lpa_latents(
    batch: int,
    height: int,
    width: int,
    dtype,
    device,
    generators: List[torch.Generator],
    alpha: float = 0.5,
) -> torch.Tensor:
    """Return latents shaped (batch, C, H/8, W/8) with shared low‑freq.
    alpha=1.0 ⇒ fully shared; alpha=0.0 ⇒ fully independent.
    """
    # UNet expects latents size (b, 4, H/8, W/8) for SD‑1.5
    h8, w8 = height // 8, width // 8
    base_gen = generators[0]

    # base noise shared prototype
    base = torch.randn((1, 4, h8, w8), generator=base_gen, device=device, dtype=dtype)
    base_low = _lowpass_fft(base, keep_ratio=0.35)

    indiv = _randn_latents(batch, 4, h8, w8, dtype, device, generators)
    indiv_low = _lowpass_fft(indiv, keep_ratio=0.35)

    shared = base_low.expand(batch, -1, -1, -1)
    low_mix = alpha * shared + (1 - alpha) * indiv_low
    hi_mix = indiv - indiv_low

    latents = low_mix + hi_mix
    return latents