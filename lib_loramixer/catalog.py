"""The LoRA list the widget picks from, read through the WebUI's own LoRA page.

Going through the Extra Networks page keeps names, aliases, previews and user
metadata exactly as each WebUI (Forge, reForge, Neo) shows them in its sidebar.
"""

from __future__ import annotations

import json
import os
import re
import threading

from . import blocks

_lock = threading.Lock()
_cache = None

_JS_STRING = re.compile(r'^"((?:[^"\\]|\\.)*)"')


def _unquote_js(s):
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _tag_name(item):
    """The name the sidebar writes into <lora:NAME:...> for this item."""
    prompt = item.get("prompt") or ""
    m = _JS_STRING.match(prompt)
    if m:
        text = _unquote_js(m.group(1))
        m2 = re.match(r"<lora:(.*?):", text)
        if m2:
            return m2.group(1)
    return item.get("name")


def _split_words(text):
    return [w.strip() for w in re.split(r"[,\n]", text or "") if w.strip()]


def _civitai_words(path):
    """trainedWords from a Civitai Helper / civitai info file next to the LoRA."""
    base = os.path.splitext(path)[0]
    for ext in (".civitai.info", ".info"):
        f = base + ext
        if os.path.isfile(f):
            try:
                with open(f, "r", encoding="utf8") as fh:
                    data = json.load(fh)
                words = data.get("trainedWords") or []
                out = []
                for w in words:
                    out += _split_words(w) if isinstance(w, str) else []
                return out
            except Exception:
                return []
    return []


def _lora_page():
    from modules import ui_extra_networks
    for page in ui_extra_networks.extra_pages:
        if getattr(page, "extra_networks_tabname", "") == "lora" or getattr(page, "name", "") == "lora":
            return page
    return None


def _root_dirs():
    from modules import shared
    dirs = [getattr(shared.cmd_opts, "lora_dir", None)]
    dirs += list(getattr(shared.cmd_opts, "lora_dirs", None) or [])
    return [os.path.abspath(d) for d in dirs if d]


def _folder(path, roots):
    ap = os.path.abspath(path)
    for r in roots:
        if ap.startswith(r + os.sep):
            return os.path.dirname(os.path.relpath(ap, r)).replace(os.sep, "/")
    return ""


def build(refresh=False):
    """[{name, tag, folder, preview, weight, triggers, sd, hash}], cached."""
    global _cache
    with _lock:
        if _cache is not None and not refresh:
            return _cache
        page = _lora_page()
        if page is None:
            return []
        if refresh:
            try:
                page.refresh()
            except Exception:
                pass
        from modules import shared
        default_w = float(getattr(shared.opts, "extra_networks_default_multiplier", 1.0) or 1.0)
        roots = _root_dirs()
        # The page's own module has the host's `networks` imported.
        import sys
        networks = getattr(sys.modules.get(type(page).__module__), "networks", None)
        if networks is not None:
            names = list(networks.available_networks)
        else:
            names = [it["name"] for it in page.list_items()]
        out = []
        for index, name in enumerate(names):
            try:
                item = page.create_item(name, index, enable_filter=False)
            except TypeError:
                item = page.create_item(name, index)
            except Exception:
                item = None
            if not item:
                continue
            um = item.get("user_metadata") or {}
            path = item.get("filename") or ""
            triggers = _split_words(um.get("activation text", ""))
            for w in _civitai_words(path):
                if w.lower() not in {t.lower() for t in triggers}:
                    triggers.append(w)
            try:
                weight = float(um.get("preferred weight") or 0) or default_w
            except (TypeError, ValueError):
                weight = default_w
            out.append({
                "name": item.get("name"),
                "tag": _tag_name(item),
                "folder": _folder(path, roots),
                "preview": item.get("preview") or "",
                "weight": weight,
                "triggers": triggers,
                "sd": um.get("sd version") or "",
                "hash": item.get("shorthash") or "",
                "arch": blocks.lora_arch_of_file(path),
            })
        out.sort(key=lambda x: (x["folder"].lower(), x["name"].lower()))
        _cache = out
        return out
