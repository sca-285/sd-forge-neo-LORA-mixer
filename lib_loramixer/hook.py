"""Block weights inside the WebUI's own LoRA loader.

Forge, reForge and Neo all apply a LoRA the same way: the Lora extension's
activate() reads the <lora:...> tags, load_networks() calls
load_lora_for_models() once per LoRA file, and that calls the model patcher's
add_patches(patches, strength) for the UNet and for the text encoder. Three
thin wrappers add block weights to that path without replacing any of it:

  activate()              reads lbw=... from each tag, keyed by LoRA file,
                          and makes the loader reapply LoRAs when only the
                          block weights changed (its cache ignores them);
  load_lora_for_models()  notes which LoRA file is being applied;
  add_patches()           splits that file's patches by block and adds each
                          group at strength x block weight (0 = left out).

Nothing changes for a LoRA without lbw=. The same path serves SD 1.x, SDXL and
transformer models (Flux, Anima, ...): only the block names differ (blocks.py).
"""

from __future__ import annotations

import re
import sys
import threading

from . import blocks

_state = threading.local()
_specs = {}            # LoRA filename -> parsed spec (list or preset name)
_last_signature = None
_warned = set()


def _note(key, text):
    if key not in _warned:
        _warned.add(key)
        print(f"[LoRA Mixer] {text}")


# Transformer models by class (Forge Neo's names), for a readable family id.
_DIT_NAMES = {
    "integratedfluxtransformer2dmodel": ("flux", "Flux"),
    "integratedchromatransformer2dmodel": ("chroma", "Chroma"),
    "qwenimagetransformer2dmodel": ("qwen", "Qwen-Image"),
    "anima": ("anima", "Anima"),
    "nextdit": ("zimage", "Z-Image / Lumina 2"),
    "wanmodel": ("wan", "Wan"),
    "singlestreamdit": ("krea", "Krea"),
}


def _diffusion_model():
    try:
        # model_data, not shared.sd_model: on reForge reading that from the UI
        # can start loading a checkpoint.
        from modules import sd_models
        m = sd_models.model_data.sd_model
    except Exception:
        return None, None
    if m is None:
        return None, None
    try:
        unet = m.forge_objects.unet.model.diffusion_model
    except Exception:
        unet = getattr(getattr(m, "model", None), "diffusion_model", None)
    return m, unet


def _is_nunchaku(unet):
    return any("nunchaku" in c.__name__.lower() or c.__name__.lower().startswith("svdq")
               for c in type(unet).__mro__)


def _register(fam, label, groups):
    """Register a transformer layout under `fam`, or a variant name when `fam`
    already has another block count; returns the family id."""
    names = ["BASE"] + [f"{p}{i:02d}" for p, n in groups for i in range(n)]
    known = blocks.names_of(fam)
    if known is not None and known != names:        # a variant with another block count
        fam, label = f"{fam}{len(names) - 1}", f"{label} ({len(names) - 1} blocks)"
    if blocks.names_of(fam) is None:
        blocks.register(fam, label, groups)
    return fam


_file_families = {}      # (path, mtime) -> family or None


def _family_of_checkpoint(path):
    """The family of a checkpoint file that is chosen but not loaded yet, told
    from its weight names (safetensors / GGUF header only)."""
    import os
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key in _file_families:
        fam = _file_families[key]
        return fam
    names = blocks.tensor_names(path)
    found = blocks.model_groups(names)
    fam = None
    if isinstance(found, str):
        fam = found
    elif found:
        prefixes = {p for p, _ in found}
        text = " ".join(names[:4000])
        if {"D", "S"} <= prefixes:
            base = ("chroma", "Chroma") if "distilled_guidance_layer" in text else ("flux", "Flux")
        elif prefixes == {"T"}:
            base = ("qwen", "Qwen-Image")
        elif "L" in prefixes and ({"NR", "CR"} & prefixes):
            base = ("zimage", "Z-Image / Lumina 2")
        elif "B" in prefixes and "patch_embedding" in text:
            base = ("wan", "Wan")
        elif "B" in prefixes and ("refiner_blocks" in text or "layerwise_blocks" in text):
            base = ("krea", "Krea")
        elif "B" in prefixes:
            base = ("anima", "Anima")
        else:
            base = ("dit", "Transformer")
        fam = _register(base[0], base[1], found)
    _file_families[key] = fam
    return fam


def _pending_checkpoint():
    """Forge / Neo load the chosen checkpoint only at the next generation; until
    then the old model stays in memory. The chosen file while that is so, else None."""
    try:
        from modules import sd_models
        md = sd_models.model_data
        params = getattr(md, "forge_loading_parameters", None)
        if not params or getattr(md, "forge_hash", None) == str(params):
            return None
        return getattr(params.get("checkpoint_info"), "filename", None)
    except Exception:
        return None


def _dit_family(unet):
    """Register the transformer's block layout and return its family id, or None."""
    if unet is None or _is_nunchaku(unet):
        return None
    groups = []
    for container, prefix, _label in blocks.DIT_GROUPS:
        mods = getattr(unet, container, None)
        try:
            n = len(mods) if mods is not None else 0
        except TypeError:
            n = 0
        if n:
            groups.append((prefix, n))
    if not groups:
        return None
    cls = type(unet).__name__
    fam, label = _DIT_NAMES.get(cls.lower(), (re.sub(r"[^a-z0-9]+", "", cls.lower()) or "dit", cls))
    return _register(fam, label, groups)


def _family():
    """'sdxl', 'sd1' or a transformer family ('flux', 'anima', ...) for the loaded
    model, None for anything else.

    The model's structure first: a transformer's layout is read from the blocks
    it has, and a UNet's input blocks tell SDXL (9) from SD 1.x / 2.x (12). The
    flags come last: reForge marks every model that is not SDXL / SD2 / SD3 as
    is_sd1, Flux and Chroma included.
    """
    m, unet = _diffusion_model()
    if m is None:
        return None
    blocks_in = getattr(unet, "input_blocks", None)
    if blocks_in is not None and getattr(unet, "output_blocks", None) is not None:
        n = len(blocks_in)
        if n == 9:
            return "sdxl"
        if n == 12:
            return "sd1"
    fam = _dit_family(unet)
    if fam is not None:
        return fam
    if unet is not None and (_is_nunchaku(unet) or blocks_in is None):
        return None              # a transformer we cannot weight; not an SD model either
    if getattr(m, "is_sdxl", False):
        return "sdxl"
    if getattr(m, "is_sd1", False) or getattr(m, "is_sd2", False):
        return "sd1"
    return None


def current_family():
    """The family block weights are edited and applied for: the chosen checkpoint
    when it is not loaded yet (Forge / Neo load at the next generation), else the
    loaded model."""
    pending = _pending_checkpoint()
    if pending:
        fam = _family_of_checkpoint(pending)
        if fam:
            return fam
    return _family() or "other"


def unsupported_note():
    """Why block weights cannot act on the loaded model, or ''."""
    _m, unet = _diffusion_model()
    if unet is not None and _is_nunchaku(unet):
        return ("This is a Nunchaku (SVDQuant) model: it applies LoRAs its own way, so block weights "
                "have no effect on it. GGUF and fp8 models work.")
    return ""


# ------------------------------------------------------------------ wrappers

def _wrap_add_patches(cls):
    if getattr(cls.add_patches, "_lora_mixer", False):
        return
    original = cls.add_patches

    def add_patches(self, *args, **kwargs):
        weights = getattr(_state, "weights", None)
        if not weights:
            return original(self, *args, **kwargs)
        args = list(args)
        if "patches" in kwargs:
            patches = kwargs["patches"]
        elif args:
            patches = args[0]
        else:
            return original(self, *args, **kwargs)
        if "strength_patch" in kwargs:
            strength = float(kwargs["strength_patch"])
        elif len(args) > 1:
            strength = float(args[1])
        else:
            strength = 1.0

        loaded = None
        zeroed = []
        for w, keys in blocks.split_by_weight(list(patches), weights).items():
            if w == 0.0:
                zeroed = keys
                continue
            sub = {k: patches[k] for k in keys}
            a, kw = list(args), dict(kwargs)
            if "patches" in kw:
                kw["patches"] = sub
            else:
                a[0] = sub
            if "strength_patch" in kw or len(a) < 2:
                kw["strength_patch"] = strength * w
            else:
                a[1] = strength * w
            got = original(self, *a, **kw)
            if loaded is None:
                loaded = got
            elif isinstance(loaded, set):
                loaded |= set(got)
            else:
                loaded = list(loaded) + [k for k in got if k not in loaded]
        # Report blocks weighted 0 as loaded: they were handled, and Forge / Neo
        # discard a LoRA whose "skipped" share looks like a model mismatch.
        if zeroed:
            try:
                model_keys = set(self.model.state_dict().keys())
            except Exception:
                model_keys = None
            present = [k for k in zeroed
                       if model_keys is None or (k if isinstance(k, str) else k[0]) in model_keys]
            if loaded is None:
                loaded = set(present) if not isinstance(patches, list) else list(present)
            elif isinstance(loaded, set):
                loaded |= set(present)
            else:
                loaded = list(loaded) + [k for k in present if k not in loaded]
        return loaded if loaded is not None else []

    add_patches._lora_mixer = True
    cls.add_patches = add_patches


def _patcher_classes(sd_model):
    out = []
    fo = getattr(sd_model, "forge_objects", None)
    for obj in (getattr(fo, "unet", None), getattr(getattr(fo, "clip", None), "patcher", None)):
        if obj is None:
            continue
        for cls in type(obj).__mro__:
            if "add_patches" in cls.__dict__:
                out.append(cls)
                break
    return out


def _wrap_loader(networks):
    fn = getattr(networks, "load_lora_for_models", None)
    if fn is None or getattr(fn, "_lora_mixer", False):
        return fn is not None

    def load_lora_for_models(*args, **kwargs):
        filename = kwargs.get("filename")
        spec = _specs.get(filename) if filename else None
        weights = None
        if spec is not None:
            fam = _family()
            weights = blocks.resolve(spec, fam) if fam else None
            if weights is None:
                _note(("fam", filename, fam), f"block weights for {filename} do not fit this model "
                      f"({fam or 'blocks not recognised'}); applied without them.")
        _state.weights = weights
        try:
            return fn(*args, **kwargs)
        finally:
            _state.weights = None

    load_lora_for_models._lora_mixer = True
    networks.load_lora_for_models = load_lora_for_models
    return True


def _filename_for(networks, name):
    for table in ("available_network_aliases", "available_networks"):
        on_disk = getattr(networks, table, {}).get(name)
        if on_disk is not None:
            return on_disk.filename
    return None


def _wrap_activate(lora_network, networks):
    if getattr(lora_network.activate, "_lora_mixer", False):
        return
    original = lora_network.activate

    def activate(p, params_list, *args, **kwargs):
        global _last_signature
        _specs.clear()
        for params in params_list or []:
            raw = getattr(params, "named", {}).get("lbw")
            if raw is None or not getattr(params, "positional", None):
                continue
            spec = blocks.parse(raw)
            if spec is None:
                _note(("bad", raw), f"could not read lbw={raw}; ignored.")
                continue
            filename = _filename_for(networks, params.positional[0])
            if filename:
                _specs[filename] = spec
        signature = repr(sorted((k, repr(v)) for k, v in _specs.items()))
        if signature != _last_signature:
            _last_signature = signature
            try:
                from modules import shared
                shared.sd_model.current_lora_hash = None   # the loader's cache ignores lbw
            except Exception:
                pass
        try:
            from modules import shared
            for cls in _patcher_classes(shared.sd_model):
                _wrap_add_patches(cls)
        except Exception as e:
            _note("patcher", f"block weights unavailable: {e}")
        return original(p, params_list, *args, **kwargs)

    activate._lora_mixer = True
    lora_network.activate = activate


def install():
    """Wrap the Lora extension in place. Safe to call repeatedly."""
    try:
        from modules import extra_networks
    except Exception:
        return False
    lora_network = extra_networks.extra_network_registry.get("lora")
    if lora_network is None:
        return False
    networks = getattr(sys.modules.get(type(lora_network).__module__), "networks", None)
    if networks is None:
        _note("networks", "could not find the LoRA loader; block weights are unavailable.")
        return False
    if not _wrap_loader(networks):
        _note("loader", "this WebUI's LoRA loader has no load_lora_for_models; block weights are unavailable.")
        return False
    _wrap_activate(lora_network, networks)
    return True
