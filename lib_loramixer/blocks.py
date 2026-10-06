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
all of them.

Transformer (DiT) models have no fixed layout: the loaded model's blocks are
counted at run time and registered (register()), named by their container:

  Flux / Chroma: BASE, D00..D18 (double_blocks), S00..S37 (single_blocks)
  Qwen-Image:    BASE, T00..T59 (transformer_blocks)
  Anima, Wan:    BASE, B00..   (blocks)
  Z-Image:       BASE, CR00.., NR00.. (context / noise refiner), L00.. (layers)

Besides a list of values, a spec can name blocks, ranges and groups with
NAME*VALUE entries, so it stays short and works on any model that has them:
  lbw=TE*0,S07-S20*0.5,D*0.8     TE / BASE, one block, a range, a whole group
  lbw=EARLY*1,MIDDLE*0.5,LATE*0  thirds of the model, first to last
Blocks not named keep 1. ("=" cannot be used inside: the WebUI splits on it.)
No WebUI imports, so this can be tested on its own.
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
# ------------------------------------------------- transformer (DiT) layouts

# (container in the model, block name prefix, label), in the order they run.
DIT_GROUPS = [
    ("context_refiner", "CR", "Context refiner"),
    ("noise_refiner", "NR", "Noise refiner"),
    ("layerwise_blocks", "LW", "Text blocks"),
    ("refiner_blocks", "RF", "Text refiner"),
    ("double_blocks", "D", "Double blocks"),
    ("joint_blocks", "J", "Joint blocks"),
    ("transformer_blocks", "T", "Transformer blocks"),
    ("blocks", "B", "Blocks"),
    ("layers", "L", "Layers"),
    ("single_blocks", "S", "Single blocks"),
]
_DIT_KEY = re.compile(r"(?:^|\.)(" + "|".join(g[0] for g in DIT_GROUPS) + r")\.(\d+)\.")
_DIT_PREFIX = {g[0]: g[1] for g in DIT_GROUPS}
GROUP_LABELS = {g[1]: g[2] for g in DIT_GROUPS}
GROUP_LABELS.update({"IN": "Input blocks", "M": "Middle block", "OUT": "Output blocks"})

_DYNAMIC = {}        # family -> {"label": str, "names": [...]}, registered from the loaded model
LABELS = {"sdxl": "SDXL", "sd1": "SD 1.x"}


def register(family, label, groups):
    """Add a transformer model's layout: groups is [(prefix, count)] in run order."""
    names = ["BASE"] + [f"{p}{i:02d}" for p, n in groups for i in range(n)]
    _DYNAMIC[family] = {"label": label, "names": names}
    return names


def layouts():
    """{family: editor names}: SDXL, SD 1.x and every registered transformer model."""
    out = dict(LAYOUTS)
    out.update({k: v["names"] for k, v in _DYNAMIC.items()})
    return out


def labels():
    out = dict(LABELS)
    out.update({k: v["label"] for k, v in _DYNAMIC.items()})
    return out


def names_of(family):
    return layouts().get(family)


def full_of(family):
    return FULL.get(family) or names_of(family)


def prefix_of(name):
    return "BASE" if name == "BASE" else re.sub(r"\d+$", "", name)


def _thirds(names):
    body = [n for n in names if n != "BASE"]
    k = len(body)
    return body[: round(k / 3)], body[round(k / 3): round(2 * k / 3)], body[round(2 * k / 3):]


def _dit_presets(family, names):
    """(name, about, values) for a transformer layout."""
    def only(keep, te=True):
        return [1.0 if (n == "BASE" and te) or n in keep else 0.0 for n in names]
    early, middle, late = _thirds(names)
    out = [
        ("All", "Every block at full weight (the same as no block weights).", [1.0] * len(names)),
        ("No text encoder", "The LoRA changes the image model only; your prompt words keep their usual meaning.",
         [0.0] + [1.0] * (len(names) - 1)),
        ("Early blocks (first third)", "Experimental. The first third of the transformer, where layout and "
         "shapes tend to form.", only(set(early))),
        ("Middle blocks (middle third)", "Experimental. The middle third of the transformer.", only(set(middle))),
        ("Late blocks (last third)", "Experimental. The last third of the transformer, where detail, texture "
         "and rendering tend to be decided.", only(set(late))),
        ("Skip late blocks", "Experimental. Everything but the last third: the LoRA's subject and layout "
         "with less of its rendering.", only(set(early) | set(middle))),
    ]
    prefixes = {prefix_of(n) for n in names}
    if {"D", "S"} <= prefixes:
        out += [
            ("Double blocks only", "Experimental. Flux / Chroma double-stream blocks, where text and image "
             "are still separate streams.", only({n for n in names if n.startswith("D")})),
            ("Single blocks only", "Experimental. Flux / Chroma single-stream blocks, the later part of the "
             "model.", only({n for n in names if n.startswith("S")})),
        ]
    return out


def presets():
    """[(name, about, {family: values})] for SD 1.x, SDXL and the registered transformer models."""
    merged = {name: (name, about, dict(values)) for name, about, values in PRESETS}
    order = [p[0] for p in PRESETS]
    for fam, info in _DYNAMIC.items():
        for name, about, values in _dit_presets(fam, info["names"]):
            if name not in merged:
                merged[name] = (name, about, {})
                order.append(name)
            merged[name][2][fam] = values
    return [merged[n] for n in order]


def preset_by_name(name):
    for p in presets():
        if p[0].lower() == str(name).strip().lower():
            return p
    return None


def family_of(values):
    """(family, names) for a list of values, or (None, None)."""
    hit = BY_LENGTH.get(len(values))
    if hit is not None:
        key, names = hit
        return key.replace("_full", ""), names
    for fam, info in _DYNAMIC.items():
        if len(info["names"]) == len(values):
            return fam, info["names"]
    return None, None


class Named(tuple):
    """A spec of (target, value) entries: TE*0, S07-S20*0.5, D*0.8, LATE*0."""


def _parse_named(s):
    entries = []
    for part in re.split(r"[,;]", s):
        part = part.strip()
        if not part:
            continue
        if "*" not in part:
            return None
        target, value = part.rsplit("*", 1)
        try:
            entries.append((target.strip().upper(), float(value)))
        except ValueError:
            return None
    return Named(entries) if entries else None


def parse(text):
    """'1,0,0.5,...', 'TE*0,S*0.5' or a preset name -> list of floats, Named, the
    preset name, or None. A list of any length is accepted here; resolve() checks
    it fits the loaded model."""
    if text is None:
        return None
    if isinstance(text, Named):
        return text
    if isinstance(text, (list, tuple)):
        try:
            vals = [float(v) for v in text]
        except (TypeError, ValueError):
            return None
        return vals or None
    s = str(text).strip()
    if not s:
        return None
    if preset_by_name(s) is not None:
        return s  # resolved later, when the model family is known
    if "*" in s:
        return _parse_named(s)
    try:
        vals = [float(v) for v in re.split(r"[,\s]+", s) if v != ""]
    except ValueError:
        return None
    return vals or None


def targets(target, family):
    """The blocks of `family` a Named entry's target means."""
    full = full_of(family) or []
    body = [n for n in full if n != "BASE"]
    t = target.upper()
    if t in ("TE", "BASE"):
        return ["BASE"]
    if t in ("ALL", "UNET", "DIT", "MODEL"):
        return body
    if t in ("EARLY", "MIDDLE", "LATE"):
        return list(_thirds(full)[("EARLY", "MIDDLE", "LATE").index(t)])
    if t == "MID" and "M00" in full:
        return ["M00"]
    if t in full:
        return [t]
    m = re.fullmatch(r"([A-Z]+)(\d+)-([A-Z]*)(\d+)", t)
    if m and (not m.group(3) or m.group(3) == m.group(1)):
        lo, hi = sorted((int(m.group(2)), int(m.group(4))))
        return [n for n in body if prefix_of(n) == m.group(1) and lo <= int(n[len(m.group(1)):]) <= hi]
    return [n for n in body if prefix_of(n) == t]


def resolve(spec, family):
    """A parsed spec (list, Named or preset name) -> {block: weight} for `family`, or None."""
    if spec is None or not family or names_of(family) is None:
        return None
    if isinstance(spec, str):
        preset = preset_by_name(spec)
        if preset is None or family not in preset[2]:
            return None
        spec = preset[2][family]
    if isinstance(spec, Named):
        weights = {n: 1.0 for n in full_of(family)}
        for target, value in spec:
            for n in targets(target, family):
                weights[n] = value
        return weights
    fam, names = family_of(spec)
    if fam != family:
        return None
    weights = dict(zip(names, spec))
    for block in full_of(family):
        if block in weights:
            continue
        side, n = block[:-2], int(block[-2:])
        listed = [(abs(int(b[-2:]) - n), int(b[-2:]), w) for b, w in weights.items()
                  if b[:-2] == side and b != "BASE"]
        if listed:
            weights[block] = min(listed)[2]
    return weights


def values_for(spec, family):
    """A spec as the editor's list of values for `family`, or None."""
    w = resolve(spec, family)
    names = names_of(family)
    return [float(w.get(n, 1.0)) for n in names] if w and names else None


def fmt_spec(values):
    """The lbw= text for a list of values: as it is for SD 1.x / SDXL, and as short
    NAME*VALUE entries for a transformer layout (58 numbers for Flux otherwise)."""
    fam, names = family_of(values)
    if fam is None or fam in LAYOUTS:
        return fmt_list(values)
    from .tags import fmt
    entries = []
    if abs(values[0] - 1.0) > 1e-9:
        entries.append(f"TE*{fmt(values[0])}")
    groups = {}
    for n, v in zip(names[1:], values[1:]):
        groups.setdefault(prefix_of(n), []).append((n, float(v)))
    for prefix, items in groups.items():
        if all(abs(v - items[0][1]) < 1e-9 for _, v in items):
            if abs(items[0][1] - 1.0) > 1e-9:
                entries.append(f"{prefix}*{fmt(items[0][1])}")
            continue
        i = 0
        while i < len(items):
            j = i
            while j + 1 < len(items) and abs(items[j + 1][1] - items[i][1]) < 1e-9:
                j += 1
            if abs(items[i][1] - 1.0) > 1e-9:
                target = items[i][0] if i == j else f"{items[i][0]}-{items[j][0]}"
                entries.append(f"{target}*{fmt(items[i][1])}")
            i = j + 1
    return ",".join(entries) if entries else fmt_list(values)


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
    if "diffusion_model." in key:
        m = _DIT_KEY.search(key)
        if m:
            return f"{_DIT_PREFIX[m.group(1)]}{int(m.group(2)):02d}"
    if "diffusion_model" in key or key.startswith("unet."):
        return None          # model weights outside the blocks (embedders, final layer)
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


def tensor_names(path):
    """Tensor names from a .safetensors or .gguf header (no tensors are read), or []."""
    import json
    import struct
    p = str(path).lower()
    try:
        with open(path, "rb") as f:
            if p.endswith(".safetensors"):
                n = struct.unpack("<Q", f.read(8))[0]
                if n <= 0 or n > 100_000_000:
                    return []
                return [k for k in json.loads(f.read(n)) if k != "__metadata__"]
            if p.endswith(".gguf"):
                return _gguf_names(f)
    except Exception:
        return []
    return []


def _gguf_names(f):
    import struct

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        return f.read(u64()).decode("utf8", "replace")

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(kind):
        if kind in sizes:
            f.seek(sizes[kind], 1)
        elif kind == 8:
            string()
        elif kind == 9:
            sub, count = u32(), u64()
            if sub in sizes:
                f.seek(sizes[sub] * count, 1)
            else:
                for _ in range(count):
                    skip(sub)
        else:
            raise ValueError(f"unknown GGUF value type {kind}")

    if f.read(4) != b"GGUF":
        return []
    u32()                                   # version
    n_tensors, n_kv = u64(), u64()
    for _ in range(n_kv):
        string()
        skip(u32())
    names = []
    for _ in range(n_tensors):
        names.append(string())
        dims = u32()
        f.seek(8 * dims + 4 + 8, 1)         # dims, type, offset
    return names


# Weights of the text encoders / VAE inside an all-in-one checkpoint.
_NOT_MODEL = re.compile(r"(text_model|text_encoders|conditioner|cond_stage_model|first_stage_model|"
                        r"(?:^|\.)vae\.|llm_adapter|qwen3|t5xxl|clip_[lg]\.)")


def model_groups(names):
    """[(prefix, count)] of a transformer's blocks from its weight names, in run
    order, or 'sdxl' / 'sd1' for a UNet, or None."""
    names = [n for n in names if not _NOT_MODEL.search(n)]
    ins = {int(m.group(1)) for n in names for m in [re.search(r"(?:^|\.)input_blocks\.(\d+)\.", n)] if m}
    if ins:
        return {9: "sdxl", 12: "sd1"}.get(max(ins) + 1)
    found = {}
    for n in names:
        m = _DIT_KEY.search(n)
        if m:
            found.setdefault(m.group(1), set()).add(int(m.group(2)))
    return [(_DIT_PREFIX[c], max(found[c]) + 1) for c, _p, _l in DIT_GROUPS if c in found] or None


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
