"""LoRA block weights: formats, presets, and which block a model weight belongs to.

A block-weight list starts with BASE (the text encoder) followed by the UNet
blocks that hold attention layers, the same layouts other block-weight tools
use, so values written in a prompt are portable:

  SDXL, 12 values:  BASE, IN04, IN05, IN07, IN08, M00, OUT00..OUT05
  SD1.x, 17 values: BASE, IN01, IN02, IN04, IN05, IN07, IN08, M00, OUT03..OUT11

Full layouts are accepted too (SDXL 20: BASE, IN00..IN08, M00, OUT00..OUT08;
SD1.x 26: BASE, IN00..IN11, M00, OUT00..OUT11). A block a short layout does not
list (they hold only convolutions, used by LoCon / LyCORIS) takes the weight of
the nearest listed block on the same side, so "output blocks only" really means
all of them. No WebUI imports, so this can be tested on its own.
"""

from __future__ import annotations

import re

LAYOUTS = {
    "sdxl": ["BASE", "IN04", "IN05", "IN07", "IN08", "M00",
             "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"],
    "sd1": ["BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "M00",
            "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"],
}
FULL = {
    "sdxl": ["BASE"] + [f"IN{i:02d}" for i in range(9)] + ["M00"] + [f"OUT{i:02d}" for i in range(9)],
    "sd1": ["BASE"] + [f"IN{i:02d}" for i in range(12)] + ["M00"] + [f"OUT{i:02d}" for i in range(12)],
}
BY_LENGTH = {len(v): (k, v) for k, v in {**{f"{k}": v for k, v in LAYOUTS.items()},
                                          **{f"{k}_full": v for k, v in FULL.items()}}.items()}


def _vec(family, on):
    names = LAYOUTS[family]
    return [1.0 if n in on else 0.0 for n in names]


def _all_but(family, off):
    names = LAYOUTS[family]
    return [0.0 if n in off else 1.0 for n in names]


_IN = {"sdxl": {"IN04", "IN05", "IN07", "IN08"}, "sd1": {"IN01", "IN02", "IN04", "IN05", "IN07", "IN08"}}
_OUT = {"sdxl": {f"OUT{i:02d}" for i in range(6)}, "sd1": {f"OUT{i:02d}" for i in range(3, 12)}}

# (name, description, {family: values}); a family missing from a preset means
# the preset does not exist for it.
PRESETS = [
    ("All", "Every block at full weight (the same as no block weights).",
     {f: [1.0] * len(LAYOUTS[f]) for f in LAYOUTS}),
    ("No text encoder", "The LoRA changes the image model only; your prompt words keep their usual meaning.",
     {f: _all_but(f, {"BASE"}) for f in LAYOUTS}),
    ("Composition (in + mid)", "Input and middle blocks: layout, pose and shapes; leaves the LoRA's rendering style out.",
     {f: _vec(f, {"BASE", "M00"} | _IN[f]) for f in LAYOUTS}),
    ("Detail & style (out)", "Output blocks: textures, colour and rendering; keeps your own composition.",
     {f: _vec(f, {"BASE"} | _OUT[f]) for f in LAYOUTS}),
    ("Style only (OUT01)", "SDXL block where B-LoRA found style to live. Strongest with LoRAs trained for it; "
     "an approximation with ordinary LoRAs.",
     {"sdxl": _vec("sdxl", {"OUT01"})}),
    ("Content only (OUT00)", "SDXL block where B-LoRA found the subject's content to live; the counterpart of "
     "Style only.",
     {"sdxl": _vec("sdxl", {"OUT00"})}),
    ("Style + content", "OUT00 and OUT01 only: the subject and its look, without the rest of the LoRA.",
     {"sdxl": _vec("sdxl", {"OUT00", "OUT01"})}),
]
PRESET_BY_NAME = {p[0].lower(): p for p in PRESETS}


def family_of(values):
    """('sdxl' | 'sd1', names) for a list of values, or (None, None)."""
    hit = BY_LENGTH.get(len(values))
    if hit is None:
        return None, None
    key, names = hit
    return key.replace("_full", ""), names


def parse(text):
    """'1,0,0.5,...' or a preset name -> list of floats, or None."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        try:
            vals = [float(v) for v in text]
        except (TypeError, ValueError):
            return None
        return vals if family_of(vals)[0] else None
    s = str(text).strip()
    if not s:
        return None
    if s.lower() in PRESET_BY_NAME:
        return s  # resolved later, when the model family is known
    try:
        vals = [float(v) for v in re.split(r"[,\s]+", s) if v != ""]
    except ValueError:
        return None
    return vals if family_of(vals)[0] else None


def resolve(spec, family):
    """A parsed spec (list or preset name) -> {block: weight} for `family`, or None."""
    if spec is None:
        return None
    if isinstance(spec, str):
        preset = PRESET_BY_NAME.get(spec.lower())
        if preset is None or family not in preset[2]:
            return None
        spec = preset[2][family]
    fam, names = family_of(spec)
    if fam != family:
        return None
    weights = dict(zip(names, spec))
    for block in FULL[family]:
        if block in weights:
            continue
        side, n = block[:-2], int(block[-2:])
        listed = [(abs(int(b[-2:]) - n), int(b[-2:]), w) for b, w in weights.items()
                  if b[:-2] == side and b != "BASE"]
        if listed:
            weights[block] = min(listed)[2]
    return weights


def is_neutral(values):
    return values is None or all(abs(float(v) - 1.0) < 1e-9 for v in values)


def fmt_list(values):
    from .tags import fmt
    return ",".join(fmt(v) for v in values)


# ------------------------------------------------------------ model weights

_LDM = re.compile(r"(?:^|\.)(input_blocks|output_blocks)\.(\d+)\.|(?:^|\.)(middle_block)\.")
_DIFFUSERS = re.compile(r"(?:^|\.)(down_blocks|up_blocks)\.(\d+)\.(attentions|resnets|downsamplers|upsamplers)\.(\d+)\."
                        r"|(?:^|\.)(mid_block)\.")


def block_of(key):
    """The block a model weight key belongs to: 'BASE', 'IN04', 'M00', 'OUT01' ...,
    or None for UNet weights outside the blocks (time embedding, in/out convs)."""
    if isinstance(key, (tuple, list)):
        key = key[0]
    key = str(key)
    m = _LDM.search(key)
    if m:
        if m.group(3):
            return "M00"
        prefix = "IN" if m.group(1) == "input_blocks" else "OUT"
        return f"{prefix}{int(m.group(2)):02d}"
    m = _DIFFUSERS.search(key)
    if m:
        if m.group(5):
            return "M00"
        level, kind, j = int(m.group(2)), m.group(3), int(m.group(4))
        if m.group(1) == "down_blocks":
            n = 3 * (level + 1) if kind == "downsamplers" else 3 * level + j + 1
            return f"IN{n:02d}"
        n = 3 * level + 2 if kind == "upsamplers" else 3 * level + j
        return f"OUT{n:02d}"
    if "diffusion_model" in key or key.startswith("unet."):
        return None          # UNet weights outside the blocks
    return "BASE"            # text encoder


def split_by_weight(keys, weights):
    """Group patch keys by their block weight: {weight: [keys]} (weight-1 blocks
    and unknown blocks under 1.0)."""
    groups = {}
    for k in keys:
        b = block_of(k)
        w = float(weights.get(b, 1.0)) if b is not None else 1.0
        groups.setdefault(w, []).append(k)
    return groups


# ------------------------------------------------------------ LoRA files

_UNET_SD = re.compile(r"(input_blocks|output_blocks|middle_block|down_blocks|up_blocks|mid_block)")
_DIT = re.compile(r"(double_blocks|single_blocks|transformer_blocks|single_transformer_blocks"
                  r"|(?:^|[._])blocks[._]\d|(?:^|[._])layers[._]\d|noise_refiner|context_refiner)")


def lora_arch(keys):
    """'sd' for an SD 1.x / 2.x / SDXL LoRA (UNet blocks), 'dit' for a transformer
    model (Flux, Qwen, Anima, ...), None if it cannot be told (e.g. text encoder only)."""
    sd = dit = False
    for k in keys:
        if k == "__metadata__":
            continue
        if _UNET_SD.search(k):
            sd = True
            break
        if not dit and _DIT.search(k) and "te_" not in k[:12] and "text" not in k[:20]:
            dit = True
    return "sd" if sd else ("dit" if dit else None)


def lora_arch_of_file(path):
    """lora_arch() from a .safetensors header only (no tensors are read)."""
    import json
    import struct
    if not str(path).lower().endswith(".safetensors"):
        return None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 100_000_000:
                return None
            header = json.loads(f.read(n))
    except Exception:
        return None
    return lora_arch(header.keys())
