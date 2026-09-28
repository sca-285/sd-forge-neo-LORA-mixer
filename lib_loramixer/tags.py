"""<lora:...> tags: building them from the mixer's rows, finding them in a prompt.

No WebUI imports, so this can be tested on its own.
"""

from __future__ import annotations

import json
import re

from . import blocks

# <lora:name>, <lora:name:0.8>, <lora:name:0.5:1.0>, <lora:name:te=0.5:unet=1>.
# The name may contain spaces and dots, not ':' or '>'.
TAG_RE = re.compile(r"<lora:([^:>]+)((?::[^:>]*)*)>", re.IGNORECASE)


def fmt(v) -> str:
    """0.8 -> '0.8', 1.0 -> '1', -0.25 -> '-0.25'."""
    return f"{float(v):.3f}".rstrip("0").rstrip(".") if float(v) != 0 else "0"


def parse_state(text) -> dict:
    """The widget's JSON state, tolerant of anything malformed."""
    try:
        state = json.loads(text) if text else {}
    except (TypeError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    items = []
    for it in state.get("items", []) or []:
        if not isinstance(it, dict) or not str(it.get("name", "")).strip():
            continue
        try:
            w = float(it.get("w", 1.0))
            te = float(it.get("te", w))
            unet = float(it.get("unet", w))
        except (TypeError, ValueError):
            continue
        lbw = blocks.parse(it.get("lbw")) if it.get("lbw") is not None else None
        if isinstance(lbw, str) or blocks.is_neutral(lbw):
            lbw = None
        items.append({"name": str(it["name"]).strip(), "on": bool(it.get("on", True)),
                      "w": w, "te": te, "unet": unet, "lbw": lbw, "out": bool(it.get("out", False))})
    return {"split": bool(state.get("split", False)), "items": items}


def tag_for(item, split) -> str:
    lbw = item.get("lbw")
    extra = f":lbw={blocks.fmt_list(lbw)}" if lbw and not blocks.is_neutral(lbw) else ""
    if split and abs(item["te"] - item["unet"]) > 1e-9:
        # Positional form <lora:name:te:unet>, read the same way by all three WebUIs.
        return f"<lora:{item['name']}:{fmt(item['te'])}:{fmt(item['unet'])}{extra}>"
    w = item["unet"] if split else item["w"]
    return f"<lora:{item['name']}:{fmt(w)}{extra}>"


def names_in(prompt) -> set:
    return {m.group(1).strip().lower() for m in TAG_RE.finditer(prompt or "")}


def tags_for(state, prompt="") -> tuple[list, list]:
    """(tags to add, names skipped because the prompt already has them)."""
    present = names_in(prompt)
    tags, skipped = [], []
    for it in state["items"]:
        if not it["on"] or it.get("out"):
            continue
        if it["name"].lower() in present:
            skipped.append(it["name"])
            continue
        tags.append(tag_for(it, state["split"]))
    return tags, skipped


def append_tags(prompt, tags) -> str:
    if not tags:
        return prompt
    prompt = (prompt or "").rstrip()
    sep = "" if not prompt or prompt.endswith(",") else ","
    return f"{prompt}{sep} {' '.join(tags)}".strip()


def extract(prompt, last=None, family=None):
    """Remove <lora:...> tags from a prompt: all of them, or only the last `last`
    (the ones the stack appended).

    Returns (clean prompt, items, split) where items are stack entries in
    prompt order and split says whether any tag had separate TE / UNet weights.
    """
    items, split = [], False
    matches = list(TAG_RE.finditer(prompt or ""))
    if last is not None:
        matches = matches[-last:] if last > 0 else []
    taken = set()
    for m in matches:
        name = m.group(1).strip()
        args = [a for a in m.group(2).split(":") if a != ""]
        pos, named = [], {}
        for a in args:
            if "=" in a:
                k, v = a.split("=", 1)
                named[k.strip()] = v.strip()
            else:
                pos.append(a.strip())
        try:
            te = float(named.get("te", pos[0] if pos else 1.0))
            unet = float(named.get("unet", pos[1] if len(pos) > 1 else te))
        except ValueError:
            continue
        if abs(te - unet) > 1e-9:
            split = True
        lbw = blocks.parse(named.get("lbw")) if "lbw" in named else None
        if isinstance(lbw, str):            # a preset name: store its values for this model
            w = blocks.resolve(lbw, family) if family else None
            names = blocks.LAYOUTS.get(family, [])
            lbw = [w[n] for n in names] if w else None
        if blocks.is_neutral(lbw):
            lbw = None
        items.append({"name": name, "on": True, "w": te, "te": te, "unet": unet, "lbw": lbw})
        taken.add(m.span())
    clean = TAG_RE.sub(lambda m: "" if m.span() in taken else m.group(0), prompt or "")
    clean = re.sub(r"[ \t]+,", ",", clean)            # " ," left by a removed tag
    clean = re.sub(r",\s*,+", ",", clean)             # ", ," runs
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"(^[\s,]+|[\s,]+$)", "", clean)
    clean = re.sub(r",(\S)", r", \1", clean) if ", " in (prompt or "") else clean
    return clean, items, split


def absorb(state, prompt, family=None):
    """Move every <lora:...> tag from `prompt` into the mixer state.

    A LoRA already in the list takes the tag's weights and is switched on, in
    its old place; a new one is appended. Rows that release() sent to the
    prompt and that are no longer there were deleted by hand, so they go.
    Returns (clean prompt, new state).
    """
    clean, items, split = extract(prompt, None, family)
    rows = [dict(r) for r in state["items"]]
    index = {r["name"].lower(): r for r in rows}
    found = set()
    for it in items:
        key = it["name"].lower()
        found.add(key)
        row = index.get(key)
        if row is None:
            row = dict(it, out=False)
            rows.append(row)
            index[key] = row
        else:
            row.update({k: it[k] for k in ("w", "te", "unet")}, on=True, out=False)
            row["lbw"] = it.get("lbw")
    rows = [r for r in rows if not (r.get("out") and r["name"].lower() not in found)]
    if not items and len(rows) == len(state["items"]):
        return prompt, state
    return clean, {"split": state["split"] or split, "items": rows}


def release(state, prompt):
    """Write the rows that are on back into `prompt` as tags. They stay in the
    list, marked as living in the prompt ("out"), so turning the mixer on again
    puts them back in the same order. Returns (new prompt, new state)."""
    on = [r for r in state["items"] if r["on"] and not r.get("out")]
    if not on:
        return prompt, state
    added = [tag_for(r, state["split"]) for r in on]
    rows = [dict(r, out=True) if (r["on"] and not r.get("out")) else dict(r) for r in state["items"]]
    return append_tags(prompt, added), {"split": state["split"], "items": rows}
