"""LoRA Mixer: pick LoRAs in a list, with optional per-block weights.

The widget (javascript/lora_mixer.js) keeps its state as JSON in a hidden
textbox. At generation the enabled rows are appended to the prompt as
<lora:name:weight> tags (plus lbw=... when a row has block weights), before the
WebUI reads extra networks from it, so PNG info, "Lora hashes" and sites such
as Civitai see exactly what they would see had the tags been typed.

Block weights are applied inside the WebUI's own LoRA loader (lib_loramixer/
hook.py), so lbw=... also works in a tag typed by hand.
"""

import json
import os
import sys

import gradio as gr
from modules import script_callbacks, scripts, shared
from modules.ui_components import InputAccordion

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from lib_loramixer import blocks, catalog, hook, tags  # noqa: E402

INFOTEXT_KEY = "LoRA Mixer"           # written with the image: how many tags the mixer added


def on_ui_settings():
    section = ("lora_mixer", "LoRA Mixer")
    shared.opts.add_option("lora_mixer_absorb", shared.OptionInfo(
        True, "While LoRA Mixer is on, move <lora> tags that appear in the prompt into it",
        gr.Checkbox, section=section).info(
        "Pasted parameters, Send to, sidebar clicks and finished typed tags. Turning the mixer on or off "
        "always moves the LoRAs between the list and the prompt."))


def on_app_started(_demo, app):
    from fastapi import Body, Query, Response

    def lora_list(refresh: bool = Query(False)):
        return {"items": catalog.build(refresh=refresh)}

    def thumb(k: str = Query(...)):
        got = catalog.thumbnail(k)
        if got is None:
            return Response(status_code=404)
        data, media = got
        # The URL carries the preview's mtime, so the browser may keep it for good.
        return Response(content=data, media_type=media, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    def block_info():
        """The loaded model's family and the block layouts and presets for the editor."""
        fam = hook.current_family()       # first: registers a transformer model's layout
        return {
            "family": fam,
            "layouts": blocks.layouts(),
            "labels": blocks.labels(),
            "groups": blocks.GROUP_LABELS,
            "presets": [{"name": n, "about": a, "values": v} for n, a, v in blocks.presets()],
            "note": hook.unsupported_note(),
        }

    def family():
        return {"family": hook.current_family()}

    def convert(body: dict = Body(...)):
        """Move LoRAs between the prompt text and the mixer's list.

        direction "in": every <lora> tag in the prompt joins the list;
        direction "out": the rows that are on go back into the prompt as tags.
        """
        state = tags.parse_state(json.dumps(body.get("state") or {}))
        prompt = str(body.get("prompt") or "")
        if body.get("direction") == "out":
            prompt, state = tags.release(state, prompt)
        else:
            fam = hook.current_family()
            prompt, state = tags.absorb(state, prompt, fam if fam != "other" else None)
        return {"prompt": prompt, "state": state}

    app.add_api_route("/lora-mixer/list", lora_list, methods=["GET"])
    app.add_api_route("/lora-mixer/thumb", thumb, methods=["GET"])
    app.add_api_route("/lora-mixer/convert", convert, methods=["POST"])
    app.add_api_route("/lora-mixer/blocks", block_info, methods=["GET"])
    app.add_api_route("/lora-mixer/family", family, methods=["GET"])
    hook.install()
    catalog.warm_up()


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)


class Script(scripts.Script):
    sorting_priority = 5

    def title(self):
        return "LoRA Mixer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with InputAccordion(False, label="LoRA Mixer", elem_id=f"lora_mixer_{tab}") as enabled:
            floating = gr.Checkbox(label="Floating window", value=False, elem_id=f"lora_mixer_float_{tab}",
                                   elem_classes=["lora-mixer-float-toggle"])
            gr.HTML(f'<div class="lora-mixer-mount" data-tab="{tab}"></div>', elem_id=f"lora_mixer_mount_{tab}")
            state = gr.Textbox(value='{"split": false, "items": []}', elem_id=f"lora_mixer_state_{tab}",
                               elem_classes=["lora-mixer-state"], show_label=False, lines=1)
        state.do_not_save_to_config = True

        # Pasting never switches the mixer on or fills it by itself: the tags stay
        # in the prompt, and the browser side moves them in only if the mixer is
        # on (javascript/lora_mixer.js).
        return [enabled, floating, state]

    def process(self, p, enabled, floating, state_text):
        hook.install()     # again here: Reload UI rebuilds the Lora extension's objects
        if not enabled:
            return
        state = tags.parse_state(state_text)
        fam = hook.current_family()       # also registers a transformer model's layout for tags_for
        base = getattr(p, "all_prompts", None) or [p.prompt]
        added, skipped = tags.tags_for(state, base[0] if base else "")
        if skipped:
            print(f"[LoRA Mixer] already in the prompt, not added again: {', '.join(skipped)}")
        if any(it["lbw"] for it in state["items"] if it["on"]) and fam == "other":
            print("[LoRA Mixer] block weights: this model's blocks were not recognised; the LoRAs are used without them.")
        if not added:
            return

        def extend(prompts):
            return [tags.append_tags(x, tags.tags_for(state, x)[0]) for x in prompts]

        p.all_prompts = extend(p.all_prompts)
        if getattr(p, "all_hr_prompts", None):
            p.all_hr_prompts = extend(p.all_hr_prompts)
        if getattr(p, "main_prompt", None) is not None:
            p.main_prompt = p.all_prompts[0]
        p.extra_generation_params[INFOTEXT_KEY] = str(len(added))
