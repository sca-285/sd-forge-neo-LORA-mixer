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

Nothing changes for a LoRA without lbw=.
"""

from __future__ import annotations

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


def _family():
    """'sdxl' or 'sd1' for the loaded model, None for anything else.

    The model's own flags first; failing those, the UNet's shape (SDXL has 9
    input blocks, SD 1.x / 2.x have 12), so a WebUI that names its flags
    differently still gets block weights.
    """
    try:
        from modules import shared
        m = shared.sd_model
    except Exception:
        return None
    if m is None:
        return None
    if getattr(m, "is_sdxl", False):
        return "sdxl"
    if getattr(m, "is_sd1", False) or getattr(m, "is_sd2", False):
        return "sd1"
    try:
        unet = m.forge_objects.unet.model.diffusion_model
    except Exception:
        unet = getattr(getattr(m, "model", None), "diffusion_model", None)
    blocks_in = getattr(unet, "input_blocks", None)
    if blocks_in is not None and getattr(unet, "output_blocks", None) is not None:
        n = len(blocks_in)
        if n == 9:
            return "sdxl"
        if n == 12:
            return "sd1"
    return None


def current_family():
    return _family() or "other"


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
                      f"({fam or 'not SD1/SDXL'}); applied without them.")
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
