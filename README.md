# Stable Diffusion LoRA Mixer for Forge Classic (Neo), Forge, reForge

A LoRA list for Forge, reForge and Forge Classic (Neo), in the spirit of
rgthree's **Power Lora Loader** for ComfyUI, with **block weights** built in:
each LoRA is a row you can switch on and off, weight, reorder or remove, and
limit to parts of the model (for example only its style, or leave the text
encoder alone), without editing `<lora:...>` tags by hand.

At generation the rows that are on become ordinary `<lora:name:weight>` tags at
the end of the prompt, before the WebUI reads extra networks from it. The image
therefore carries exactly what a typed tag would give: the tags in the prompt,
`Lora hashes` in PNG info, and so Civitai and other sites recognise the LoRAs.
Generating never edits your prompt box.

![asset1](https://iili.io/n5y8eJj.png)
![asset2](https://iili.io/n5y8k5x.png)

## Install

WebUI → Extensions → Install from URL → this repo → Apply and restart.

```bash
git clone https://github.com/sca-285/sd-forge-neo-LORA-mixer.git
```

## Using it

```
[x] LoRA Mixer
    [ ] Floating window
    [+ Add LoRA]  2 of 3 on                     [split] [all on/off]
    ⋮⋮ (o) [thumb] detail_tweaker    [-] 0.8 [+]   [blocks] [tag] [x]
    ⋮⋮ (o) [thumb] Film Look v2      [-] 0.6 [+]   [blocks] [tag] [x]
         film grain · cinematic · analog · Add all
```

- **Add LoRA** opens a searchable list with thumbnails, built for thousands of
  LoRAs:
  - Search matches the name, the folder and the trigger words; every word must
    match, `_ - .` count as spaces, and matches in the name come first.
  - Filter by **folder** (subfolders included), show only **★ favorites**
    (click ☆ on a LoRA) or **Recent** ones; order by folder, name or newest.
  - **List / grid** view (grid shows bigger previews).
  - Keyboard: **↑ ↓ PgUp PgDn** move, **Enter** adds, **Shift+Enter** (or
    Shift/Ctrl+click) adds and keeps the list open to add several, **Esc** closes.
  - The refresh button rescans the LoRA folders.

  Only the rows in view are drawn and previews are served as small cached
  thumbnails, so the list stays fast. The list is read in the background at
  start-up; file headers and `.civitai.info` files are cached in `cache/` and
  read again only for new or changed files.
- Click a LoRA's **name** to swap it for another; drag the **⋮⋮** handle to
  reorder; the switch turns a row off without removing it.
- The weight starts at the LoRA's *preferred weight* (set in the sidebar's
  metadata editor) or the WebUI default. **−/+** step by 0.05; type any value.
- **Split** (the sliders icon) gives each row separate **TE** (text encoder)
  and **UNet** weights, written as `<lora:name:te:unet>`.
- The **tag** button shows the trigger words, from the LoRA's *activation text*
  or a Civitai `.civitai.info` file next to it. Click one to add it to the
  prompt, or **Add all**. Words already in the prompt are dimmed.
- The **blocks** button opens the block weights of that LoRA (see below). It
  is highlighted when the LoRA has block weights.
- **Floating window** moves the list into a window you can drag and resize,
  kept per tab (txt2img, img2img) and shown only on its own tab. Its switch is
  the same as the accordion's checkbox; the dock button puts it back. Position
  and size are remembered in the browser.

A LoRA that is also typed in the prompt is not added a second time (the console
says so).

## Block weights

A LoRA touches the text encoder and every block of the UNet. Block weights
scale each part separately: 0 leaves it out, 1 is the normal strength, and the
row's own weight still multiplies everything.

| Layout | Values, in order |
|---|---|
| SDXL (12) | TE, IN04, IN05, IN07, IN08, MID, OUT00, OUT01, OUT02, OUT03, OUT04, OUT05 |
| SD 1.x (17) | TE, IN01, IN02, IN04, IN05, IN07, IN08, MID, OUT03 … OUT11 |

These are the blocks with attention layers, the usual layouts of block-weight
tools, so values written in a prompt stay portable. The full layouts (SDXL 20,
SD 1.x 26) are accepted too. Blocks a short layout leaves out hold only
convolutions (used by LoCon / LyCORIS); they follow the nearest listed block
on the same side, so "output blocks only" means all of them.

The editor opens on the loaded model's layout and offers presets:

| Preset | What it keeps |
|---|---|
| All | everything (the same as no block weights) |
| No text encoder | the image model only; prompt words keep their usual meaning |
| Composition (in + mid) | input and middle blocks: layout, pose, shapes |
| Detail & style (out) | output blocks: texture, colour, rendering |
| Style only (OUT01) | SDXL: the block where the B-LoRA paper found style |
| Content only (OUT00) | SDXL: the block where B-LoRA found the subject's content |
| Style + content | SDXL: OUT00 and OUT01 |

The last three come from research on LoRAs trained for that split
([B-LoRA](https://b-lora.github.io/B-LoRA/), Frenkel et al., 2024); with an
ordinary LoRA they are an approximation worth trying, not a guarantee.

The **blocks** button always works. When block weights may not act, the editor
says so: for a LoRA that looks like one for a transformer model (Flux, Qwen,
Anima, …, told from the key names in the file), or when the loaded model is not
recognised as SD 1.x / SDXL (by its flags, or else by its UNet's shape).

In the prompt a block-weighted LoRA is written as
`<lora:name:0.8:lbw=1,0,0,0,0,0,0,1,0,0,0,0>`. Block weights are applied inside
the WebUI's own LoRA loader, so this also works when typed by hand, and a
preset name works there too (`lbw=No text encoder`). They apply to SD 1.x and
SDXL models; with other models (Flux, Qwen, …) the LoRA is used without them
and the console says so.

## On and off mechanism

The accordion's checkbox decides where your LoRAs live.

- **On**: the mixer holds them. Any `<lora:...>` tag that shows up in the
  prompt moves into the list: pasted parameters, **Send to txt2img / img2img**,
  a click on a LoRA in the sidebar, or a tag you finish typing (taken when you
  leave the prompt box, never while you are typing in it). A LoRA already in the
  list takes the tag's weights and is switched on in its old place.
- **Off**: the prompt holds them. Switching the mixer off writes the rows that
  are on back into the prompt as ordinary tags (with split and block weights);
  rows that were off stay in the list. Pasted parameters and sidebar clicks
  stay as text, and the mixer never switches itself on.
- Switching it **on again** takes the tags back out of the prompt into the list,
  in the same order, with any weight you edited in the text meanwhile. A LoRA
  you deleted from the prompt is dropped from the list too.

`Settings > LoRA Mixer`: untick *move <lora> tags that appear in the prompt*
to stop the automatic pick-up while the mixer is on; switching it on or off
still moves the LoRAs.

Images made with the mixer carry `LoRA Mixer: N` in PNG info, the number of
tags it added.

## Files

```
scripts/lora_mixer.py      UI, prompt injection, /lora-mixer/* routes (list, blocks, convert)
lib_loramixer/tags.py      building and reading <lora:...> tags
lib_loramixer/blocks.py    block layouts, presets, which block a weight belongs to
lib_loramixer/hook.py      block weights inside the WebUI's LoRA loader
lib_loramixer/catalog.py   the LoRA list, read through the WebUI's own LoRA page
javascript/lora_mixer.js   the list, picker, block editor and floating window
style.css
```

## Credits

- Idea and workflow from **Power Lora Loader** in
  [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) by rgthree. No code
  is shared; this is a separate implementation for the WebUI.
- Block weights follow the idea and the `lbw=` layouts of
  [LoRA Block Weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)
  by hako-mikan. No code is shared.
- Style / content blocks: *Implicit Style-Content Separation using B-LoRA*,
  Yarden Frenkel, Yael Vinker, Ariel Shamir, Daniel Cohen-Or (ECCV 2024).

Thanks also to **Claude**, for help building this
extension.

## License

MIT, see `LICENSE`.
