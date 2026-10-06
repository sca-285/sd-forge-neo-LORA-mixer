"""The LoRA list the widget picks from, read through the WebUI's own LoRA page.

Going through the Extra Networks page keeps names, aliases, previews and user
metadata exactly as each WebUI (Forge, reForge, Neo) shows them in its sidebar.

With thousands of LoRAs the slow parts are reading every file's header (for
its architecture) and its .civitai.info. Those results are kept on disk, keyed
by file size and mtime, so only new or changed files are read again, and the
reads that remain run in parallel. Previews are served as small cached
thumbnails instead of the full-size images.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

from . import blocks

_lock = threading.Lock()
_cache = None
_thumb_paths = {}            # key -> preview image path; the only files /thumb serves

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
META_FILE = os.path.join(CACHE_DIR, "catalog_meta.json")
THUMB_DIR = os.path.join(CACHE_DIR, "thumbs")
THUMB_SIZE = 192             # px, the long side; sharp in the list (40px) and the grid view
META_VERSION = 1

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


def _civitai_file(path):
    base = os.path.splitext(path)[0]
    for ext in (".civitai.info", ".info"):
        f = base + ext
        if os.path.isfile(f):
            return f
    return None


def _civitai_words(info_file):
    """trainedWords from a Civitai Helper / civitai info file next to the LoRA."""
    if not info_file:
        return []
    try:
        with open(info_file, "r", encoding="utf8") as fh:
            data = json.load(fh)
        out = []
        for w in data.get("trainedWords") or []:
            out += _split_words(w) if isinstance(w, str) else []
        return out
    except Exception:
        return []


def _stat(path):
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return 0, 0


def _load_meta():
    try:
        with open(META_FILE, "r", encoding="utf8") as fh:
            data = json.load(fh)
        if data.get("v") == META_VERSION:
            return data.get("files") or {}
    except Exception:
        pass
    return {}


def _save_meta(files):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = META_FILE + ".tmp"
        with open(tmp, "w", encoding="utf8") as fh:
            json.dump({"v": META_VERSION, "files": files}, fh, ensure_ascii=False)
        os.replace(tmp, META_FILE)
    except Exception as e:
        print(f"[LoRA Mixer] could not write {META_FILE}: {e}")


def _file_meta(path, old):
    """{m, s, arch, ci, cim, words} for one LoRA file, reusing `old` when nothing changed."""
    m, s = _stat(path)
    ci = _civitai_file(path)
    cim = _stat(ci)[0] if ci else 0
    if old and old.get("m") == m and old.get("s") == s and old.get("ci") == ci and old.get("cim") == cim:
        return old
    arch = old.get("arch") if old and old.get("m") == m and old.get("s") == s else blocks.lora_arch_of_file(path)
    return {"m": m, "s": s, "arch": arch, "ci": ci, "cim": cim, "words": _civitai_words(ci)}


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


def _preview_file(preview_url):
    """The image file behind the sidebar's ./sd_extra_networks/thumb?filename=... link."""
    if not preview_url or preview_url.startswith("data:"):
        return None
    try:
        q = parse_qs(urlparse(preview_url).query)
        f = (q.get("filename") or [None])[0]
    except Exception:
        return None
    return f if f and os.path.isfile(f) else None


def _thumb_key(path):
    return hashlib.sha1(os.path.abspath(path).encode("utf8", "surrogateescape")).hexdigest()[:20]


def _items(page):
    import sys
    # The page's own module has the host's `networks` imported.
    networks = getattr(sys.modules.get(type(page).__module__), "networks", None)
    if networks is not None:
        names = list(networks.available_networks)
    else:
        names = [it["name"] for it in page.list_items()]
    items = []
    for index, name in enumerate(names):
        try:
            item = page.create_item(name, index, enable_filter=False)
        except TypeError:
            item = page.create_item(name, index)
        except Exception:
            item = None
        if item:
            items.append(item)
    return items


def build(refresh=False):
    """[{name, tag, folder, preview, weight, triggers, sd, hash, arch, mt}], cached."""
    global _cache, _thumb_paths
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
        items = _items(page)

        old_meta = _load_meta()
        paths = [it.get("filename") or "" for it in items]
        with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as pool:
            metas = list(pool.map(lambda p: _file_meta(p, old_meta.get(p)) if p else {}, paths))
        new_meta = {p: m for p, m in zip(paths, metas) if p}
        if new_meta != old_meta:
            _save_meta(new_meta)

        out, thumbs = [], {}
        for item, path, meta in zip(items, paths, metas):
            um = item.get("user_metadata") or {}
            triggers = _split_words(um.get("activation text", ""))
            seen = {t.lower() for t in triggers}
            for w in meta.get("words") or []:
                if w.lower() not in seen:
                    seen.add(w.lower())
                    triggers.append(w)
            try:
                weight = float(um.get("preferred weight") or 0) or default_w
            except (TypeError, ValueError):
                weight = default_w
            preview = item.get("preview") or ""
            pfile = _preview_file(preview)
            if pfile:
                key = _thumb_key(pfile)
                thumbs[key] = pfile
                preview = f"./lora-mixer/thumb?k={key}&v={_stat(pfile)[0]}"
            sd = um.get("sd version") or ""
            out.append({
                "name": item.get("name"),
                "tag": _tag_name(item),
                "folder": _folder(path, roots),
                "preview": preview,
                "weight": weight,
                "triggers": triggers,
                "sd": "" if sd == "Unknown" else sd,
                "hash": item.get("shorthash") or "",
                "arch": meta.get("arch"),
                "mt": int((meta.get("m") or 0) // 1_000_000_000),
            })
        out.sort(key=lambda x: (x["folder"].lower(), x["name"].lower()))
        _thumb_paths = thumbs
        _cache = out
        return out


def warm_up():
    """Build the list in the background so the first Add LoRA opens at once."""
    def run():
        try:
            build()
        except Exception as e:
            print(f"[LoRA Mixer] could not read the LoRA list: {e}")
    threading.Thread(target=run, name="lora-mixer-catalog", daemon=True).start()


def thumbnail(key):
    """(bytes, media type) of a small preview for a key from build(), or None."""
    src = _thumb_paths.get(key)
    if not src or not os.path.isfile(src):
        return None
    out = os.path.join(THUMB_DIR, key + ".webp")
    if os.path.isfile(out) and _stat(out)[0] >= _stat(src)[0]:
        with open(out, "rb") as fh:
            return fh.read(), "image/webp"
    try:
        from PIL import Image
        with Image.open(src) as im:
            im.seek(0)                      # first frame of a GIF / animated WebP
            im.thumbnail((THUMB_SIZE, THUMB_SIZE))
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=82, method=4)
        data = buf.getvalue()
    except Exception:
        # Not an image PIL can read (or no PIL): hand over the original.
        with open(src, "rb") as fh:
            data = fh.read()
        ext = os.path.splitext(src)[1].lower().lstrip(".")
        return data, {"jpg": "image/jpeg"}.get(ext, f"image/{ext or 'png'}")
    try:
        os.makedirs(THUMB_DIR, exist_ok=True)
        tmp = f"{out}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, out)
    except Exception:
        pass
    return data, "image/webp"
