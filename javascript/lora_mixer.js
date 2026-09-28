// LoRA Mixer: a list of LoRAs with on/off, weight, block weights, order and
// trigger words. State lives as JSON in the hidden textbox
// #lora_mixer_state_{tab}; the server turns enabled rows into <lora:...> tags
// at generation.

(function () {
    "use strict";

    const TABS = ["txt2img", "img2img"];
    const STEP = 0.05;
    const ICON = {
        add: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
        minus: '<svg viewBox="0 0 24 24"><path d="M5 12h14"/></svg>',
        plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
        remove: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
        grip: '<svg viewBox="0 0 24 24"><circle cx="9" cy="6" r="1.4"/><circle cx="15" cy="6" r="1.4"/><circle cx="9" cy="12" r="1.4"/><circle cx="15" cy="12" r="1.4"/><circle cx="9" cy="18" r="1.4"/><circle cx="15" cy="18" r="1.4"/></svg>',
        tag: '<svg viewBox="0 0 24 24"><path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/></svg>',
        refresh: '<svg viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/></svg>',
        split: '<svg viewBox="0 0 24 24"><path d="M4 7h10M4 17h16M17 4v6M8 14v6"/></svg>',
        dock: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 15h18"/></svg>',
        blocks: '<svg viewBox="0 0 24 24"><path d="M4 20V10M9 20V4M14 20v-8M19 20V7"/></svg>',
        all: '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="4"/><path d="M8 12l3 3 5-6"/></svg>',
    };

    let catalog = null;          // [{name, tag, folder, preview, weight, triggers, sd, hash}]
    let blockInfo = null;        // {family, layouts: {sdxl: [...], sd1: [...]}, presets: [{name, about, values}]}
    let catalogPromise = null;
    const stacks = {};           // tab -> Stack

    const app = () => (typeof gradioApp === "function" ? gradioApp() : document);
    const $ = (sel, root) => (root || app()).querySelector(sel);

    function el(tag, cls, html) {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (html !== undefined) e.innerHTML = html;
        return e;
    }

    function iconButton(icon, title, cls) {
        const b = el("button", "ls-btn" + (cls ? " " + cls : ""), ICON[icon]);
        b.type = "button";
        b.title = title;
        b.setAttribute("aria-label", title);
        return b;
    }

    function fire(input) {
        if (typeof updateInput === "function") updateInput(input);
        else input.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function round(v) {
        return Math.round(v * 1000) / 1000;
    }

    function loadBlockInfo() {
        return fetch("./lora-mixer/blocks")
            .then((r) => r.json())
            .then((d) => (blockInfo = d))
            .catch(() => (blockInfo = { family: "other", layouts: {}, presets: [] }));
    }

    function familyOf(values) {
        if (!values || !blockInfo) return null;
        for (const [fam, names] of Object.entries(blockInfo.layouts)) if (names.length === values.length) return fam;
        return null;
    }

    function presetFor(values) {
        if (!values || !blockInfo) return null;
        const fam = familyOf(values);
        const p = blockInfo.presets.find((x) => x.values[fam] && x.values[fam].every((v, i) => Math.abs(v - values[i]) < 1e-9));
        return p ? p.name : null;
    }

    // A hint shown inside the block editor when block weights may have no effect.
    // It never stops you from editing them.
    function blocksHint(info) {
        if (!blockInfo) return "";
        if (info && info.arch === "dit") {
            return "This LoRA looks like one for a transformer model (Flux, Qwen, Anima…). Block weights only act on SD 1.x / SDXL blocks, so they may have no effect.";
        }
        if (!blockInfo.layouts || !blockInfo.layouts[blockInfo.family]) {
            return "The loaded model was not recognised as SD 1.x or SDXL. If it is one, block weights still apply; otherwise the LoRA is used without them.";
        }
        return "";
    }

    // The loaded model can change at any time (checkpoint dropdown); follow it.
    function watchFamily() {
        setInterval(() => {
            if (document.hidden || !blockInfo) return;
            fetch("./lora-mixer/family").then((r) => r.json()).then((d) => {
                if (d && d.family && d.family !== blockInfo.family) {
                    blockInfo.family = d.family;
                    Object.values(stacks).forEach((s) => s.render());
                }
            }).catch(() => {});
        }, 4000);
    }

    function isNeutral(values) {
        return !values || values.every((v) => Math.abs(v - 1) < 1e-9);
    }

    function loadCatalog(refresh) {
        if (catalog && !refresh) return Promise.resolve(catalog);
        if (catalogPromise && !refresh) return catalogPromise;
        catalogPromise = fetch("./lora-mixer/list" + (refresh ? "?refresh=true" : ""))
            .then((r) => r.json())
            .then((d) => (catalog = d.items || []))
            .catch(() => (catalog = []));
        return catalogPromise;
    }

    function findLora(tag) {
        if (!catalog) return null;
        const t = String(tag).toLowerCase();
        return catalog.find((c) => String(c.tag).toLowerCase() === t) ||
            catalog.find((c) => String(c.name).toLowerCase() === t) || null;
    }

    // ------------------------------------------------------------- picker

    class Picker {
        constructor(stack) {
            this.stack = stack;
            this.node = el("div", "ls-picker");
            this.node.hidden = true;
            this.search = el("input", "ls-search");
            this.search.type = "search";
            this.search.placeholder = "Search LoRAs (name, folder, trigger word)";
            const refresh = iconButton("refresh", "Rescan the LoRA folders");
            refresh.addEventListener("click", () => {
                this.list.innerHTML = '<div class="ls-empty">Scanning…</div>';
                loadCatalog(true).then(() => { this.render(); this.stack.render(); });
            });
            const top = el("div", "ls-picker-top");
            top.append(this.search, refresh);
            this.list = el("div", "ls-picker-list");
            this.node.append(top, this.list);
            this.search.addEventListener("input", () => this.render());
            this.search.addEventListener("keydown", (e) => {
                if (e.key === "Escape") this.close();
                if (e.key === "Enter") {
                    const first = this.list.querySelector(".ls-pick");
                    if (first) first.click();
                }
            });
            document.addEventListener("mousedown", (e) => {
                if (!this.node.hidden && !this.node.contains(e.target) && !e.target.closest(".ls-open-picker")) this.close();
            });
            this.onPick = null;
        }

        open(onPick) {
            this.onPick = onPick;
            this.node.hidden = false;
            this.search.value = "";
            this.list.innerHTML = '<div class="ls-empty">Loading…</div>';
            loadCatalog().then(() => this.render());
            setTimeout(() => this.search.focus(), 0);
        }

        close() {
            this.node.hidden = true;
            this.onPick = null;
        }

        render() {
            const q = this.search.value.trim().toLowerCase();
            const words = q.split(/\s+/).filter(Boolean);
            const used = new Set(this.stack.items.map((i) => i.name.toLowerCase()));
            const hits = (catalog || []).filter((c) => {
                const hay = (c.name + " " + c.tag + " " + c.folder + " " + c.triggers.join(" ")).toLowerCase();
                return words.every((w) => hay.includes(w));
            });
            this.list.innerHTML = "";
            if (!hits.length) {
                this.list.append(el("div", "ls-empty", catalog && catalog.length ? "No match." : "No LoRAs found."));
                return;
            }
            for (const c of hits.slice(0, 300)) {
                const row = el("button", "ls-pick" + (used.has(String(c.tag).toLowerCase()) ? " ls-used" : ""));
                row.type = "button";
                row.title = c.tag + (c.hash ? "  ·  " + c.hash : "");
                const thumb = el("span", "ls-thumb");
                if (c.preview) {
                    const img = el("img");
                    img.loading = "lazy";
                    img.src = c.preview;
                    thumb.append(img);
                }
                const text = el("span", "ls-pick-text");
                text.append(el("span", "ls-pick-name", escapeHtml(c.name)));
                const sub = [c.folder, c.sd].filter(Boolean).join("  ·  ");
                if (sub) text.append(el("span", "ls-pick-sub", escapeHtml(sub)));
                row.append(thumb, text);
                row.addEventListener("click", () => {
                    const cb = this.onPick;
                    this.close();
                    if (cb) cb(c);
                });
                this.list.append(row);
            }
            if (hits.length > 300) this.list.append(el("div", "ls-empty", `${hits.length - 300} more: refine the search.`));
        }
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }

    // -------------------------------------------------------------- stack

    class Stack {
        constructor(tab) {
            this.tab = tab;
            this.items = [];
            this.split = false;
            this.openTriggers = new Set();
            this.openBlocks = new Set();
            this.lastWritten = null;
            this.root = el("div", "ls-root");
            this.build();
        }

        get textarea() {
            return $(`#lora_mixer_state_${this.tab} textarea`);
        }

        // --- state <-> textbox
        readState() {
            const ta = this.textarea;
            if (!ta || ta.value === this.lastWritten) return false;
            let s = {};
            try { s = JSON.parse(ta.value || "{}"); } catch (e) { s = {}; }
            this.split = !!s.split;
            this.items = (s.items || []).map((i) => ({
                name: String(i.name), on: i.on !== false,
                w: Number(i.w ?? 1), te: Number(i.te ?? i.w ?? 1), unet: Number(i.unet ?? i.w ?? 1),
                lbw: Array.isArray(i.lbw) ? i.lbw.map(Number) : null,
                out: !!i.out,
            }));
            this.lastWritten = ta.value;
            this.render();
            return true;
        }

        save() {
            const ta = this.textarea;
            if (!ta) return;
            const value = JSON.stringify({ split: this.split, items: this.items }, (k, v) => (k.startsWith("_") ? undefined : v));
            this.lastWritten = value;
            if (ta.value !== value) {
                ta.value = value;
                fire(ta);
            }
        }

        change() {
            this.save();
            this.render();
        }

        // --- UI
        build() {
            const bar = el("div", "ls-bar");
            this.addBtn = el("button", "ls-btn ls-add ls-open-picker", ICON.add + "<span>Add LoRA</span>");
            this.addBtn.type = "button";
            this.addBtn.addEventListener("click", () => this.picker.open((c) => this.add(c)));
            this.allBtn = iconButton("all", "Turn every LoRA on / off", "ls-all");
            this.allBtn.addEventListener("click", () => {
                const on = !this.items.every((i) => i.on);
                this.items.forEach((i) => (i.on = on));
                this.change();
            });
            this.splitBtn = iconButton("split", "Separate weights for the UNet and the text encoder", "ls-split");
            this.splitBtn.addEventListener("click", () => {
                this.split = !this.split;
                this.items.forEach((i) => {
                    if (this.split) { i.te = i.w; i.unet = i.w; } else { i.w = i.unet; }
                });
                this.change();
            });
            this.count = el("span", "ls-count");
            bar.append(this.addBtn, this.count, el("span", "ls-spacer"), this.splitBtn, this.allBtn);
            this.list = el("div", "ls-list");
            this.picker = new Picker(this);
            this.root.append(bar, this.picker.node, this.list);
        }

        add(c) {
            if (this.items.some((i) => i.name.toLowerCase() === String(c.tag).toLowerCase())) return;
            const w = round(Number(c.weight) || 1);
            this.items.push({ name: c.tag, on: true, w, te: w, unet: w, lbw: null, out: false });
            this.change();
        }

        weightControl(item, key, label) {
            const box = el("span", "ls-weight");
            if (label) box.append(el("span", "ls-wlabel", label));
            const minus = iconButton("minus", `Lower ${label || "weight"} by ${STEP}`);
            const plus = iconButton("plus", `Raise ${label || "weight"} by ${STEP}`);
            const input = el("input", "ls-num");
            input.type = "number";
            input.step = String(STEP);
            input.value = String(item[key]);
            const set = (v) => {
                if (!Number.isFinite(v)) return;
                item[key] = round(v);
                this.change();
            };
            minus.addEventListener("click", () => set(item[key] - STEP));
            plus.addEventListener("click", () => set(item[key] + STEP));
            input.addEventListener("change", () => set(parseFloat(input.value)));
            input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); });
            box.append(minus, input, plus);
            return box;
        }

        render() {
            const shown = this.items.filter((i) => !i.out);
            const out = this.items.length - shown.length;
            this.count.textContent = shown.length ? `${shown.filter((i) => i.on).length} of ${shown.length} on` : "";
            this.splitBtn.classList.toggle("ls-active", this.split);
            this.allBtn.classList.toggle("ls-active", shown.length > 0 && shown.every((i) => i.on));
            this.list.innerHTML = "";
            if (out) {
                this.list.append(el("div", "ls-note ls-out-note",
                    `${out} LoRA${out > 1 ? "s are" : " is"} in the prompt while LoRA Mixer is off. Turn it on to bring ${out > 1 ? "them" : "it"} back here.`));
            }
            if (!shown.length) {
                if (!out) this.list.append(el("div", "ls-empty", "No LoRAs yet. Use Add LoRA, click a LoRA in the sidebar, or paste parameters from an image while LoRA Mixer is on."));
                return;
            }
            this.items.forEach((item, idx) => { if (!item.out) this.list.append(this.row(item, idx)); });
        }

        // ---- LoRAs between the list and the prompt
        enabled() {
            const real = enabledToggle(this.tab);
            return !!(real && real.checked);
        }

        convert(direction) {
            if (this.busy) return Promise.resolve();
            const box = this.promptBox();
            if (!box) return Promise.resolve();
            this.busy = true;
            const sent = box.value;
            return fetch("./lora-mixer/convert", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ direction, prompt: sent, state: { split: this.split, items: this.items } }),
            }).then((r) => r.json()).then((d) => {
                if (!d || !d.state) return;
                if (box.value !== sent) return;      // the prompt changed meanwhile; try again next time
                this.settledPrompt = d.prompt;       // nothing more to take from this text
                this.split = !!d.state.split;
                this.items = d.state.items.map((i) => ({ ...i, out: !!i.out }));
                if (d.prompt !== sent) {
                    box.value = d.prompt;
                    fire(box);
                }
                this.change();
            }).catch(() => {}).finally(() => { this.busy = false; });
        }

        // While on, <lora> tags that show up in the prompt (paste, Send to, sidebar
        // clicks, a finished typed tag) move into the list; never while typing in it.
        absorbFromPrompt() {
            if (!this.enabled() || this.busy) return;
            if (typeof opts !== "undefined" && opts && opts.lora_mixer_absorb === false) return;
            const box = this.promptBox();
            if (!box || document.activeElement === box) return;
            if (box.value === this.settledPrompt && !this.items.some((i) => i.out)) return;
            // Full generation parameters waiting for the paste button: leave them to it.
            if (/\n\s*(Negative prompt:|Steps: )/.test(box.value)) return;
            const hasTag = /<lora:[^:>]+(?::[^>]*)?>/i.test(box.value);
            const hasOut = this.items.some((i) => i.out);
            if (hasTag || hasOut) this.convert("in");
        }

        row(item, idx) {
            const info = findLora(item.name);
            const row = el("div", "ls-row" + (item.on ? "" : " ls-off"));
            row.draggable = false;
            row.dataset.index = idx;

            const grip = el("span", "ls-grip", ICON.grip);
            grip.title = "Drag to reorder";
            grip.addEventListener("mousedown", () => (row.draggable = true));
            row.addEventListener("dragstart", (e) => {
                e.dataTransfer.setData("text/plain", String(idx));
                row.classList.add("ls-dragging");
            });
            row.addEventListener("dragend", () => { row.draggable = false; row.classList.remove("ls-dragging"); });
            row.addEventListener("dragover", (e) => { e.preventDefault(); row.classList.add("ls-dropzone"); });
            row.addEventListener("dragleave", () => row.classList.remove("ls-dropzone"));
            row.addEventListener("drop", (e) => {
                e.preventDefault();
                row.classList.remove("ls-dropzone");
                const from = parseInt(e.dataTransfer.getData("text/plain"), 10);
                if (!Number.isInteger(from) || from === idx) return;
                const [moved] = this.items.splice(from, 1);
                this.items.splice(idx, 0, moved);
                this.change();
            });

            const toggle = el("label", "ls-switch");
            const cb = el("input");
            cb.type = "checkbox";
            cb.checked = item.on;
            cb.title = item.on ? "On" : "Off";
            cb.addEventListener("change", () => { item.on = cb.checked; this.change(); });
            toggle.append(cb, el("span", "ls-slider"));

            const thumb = el("span", "ls-thumb");
            if (info && info.preview) {
                const img = el("img");
                img.loading = "lazy";
                img.src = info.preview;
                thumb.append(img);
            }

            const name = el("button", "ls-name ls-open-picker");
            name.type = "button";
            name.textContent = info ? info.name : item.name;
            name.title = (info ? "" : "Not found in the LoRA folders. ") + "Click to choose another LoRA";
            if (catalog && !info) name.classList.add("ls-missing");
            name.addEventListener("click", () => this.picker.open((c) => {
                item.name = c.tag;
                this.change();
            }));

            const weights = el("span", "ls-weights");
            if (this.split) {
                weights.append(this.weightControl(item, "te", "TE"), this.weightControl(item, "unet", "UNet"));
            } else {
                weights.append(this.weightControl(item, "w", ""));
            }

            const actions = el("span", "ls-actions");
            const bb = iconButton("blocks", "Block weights", "ls-blocks-btn");
            const preset = presetFor(item.lbw);
            const hasLbw = !!item.lbw && !isNeutral(item.lbw);
            bb.title = hasLbw ? `Block weights: ${preset || item.lbw.join(", ")}` : "Block weights (off)";
            bb.addEventListener("click", () => {
                if (this.openBlocks.has(item.name)) this.openBlocks.delete(item.name);
                else this.openBlocks.add(item.name);
                (blockInfo ? Promise.resolve() : loadBlockInfo()).then(() => this.render());
            });
            bb.classList.toggle("ls-active", hasLbw);
            bb.classList.toggle("ls-open", this.openBlocks.has(item.name));
            actions.append(bb);
            const triggers = info ? info.triggers : [];
            if (triggers.length) {
                const tb = iconButton("tag", "Trigger words", "ls-trigger-btn");
                tb.classList.toggle("ls-active", this.openTriggers.has(item.name));
                tb.addEventListener("click", () => {
                    if (this.openTriggers.has(item.name)) this.openTriggers.delete(item.name);
                    else this.openTriggers.add(item.name);
                    this.render();
                });
                actions.append(tb);
            }
            const rm = iconButton("remove", "Remove", "ls-remove");
            rm.addEventListener("click", () => {
                this.items.splice(idx, 1);
                this.change();
            });
            actions.append(rm);

            const main = el("div", "ls-row-main");
            main.append(grip, toggle, thumb, name, weights, actions);
            row.append(main);

            if (triggers.length && this.openTriggers.has(item.name)) {
                const chips = el("div", "ls-chips");
                const inPrompt = this.promptText().toLowerCase();
                for (const w of triggers) {
                    const chip = el("button", "ls-chip" + (inPrompt.includes(w.toLowerCase()) ? " ls-chip-used" : ""));
                    chip.type = "button";
                    chip.textContent = w;
                    chip.title = "Add to the prompt";
                    chip.addEventListener("click", () => { this.insertWords([w]); this.render(); });
                    chips.append(chip);
                }
                if (triggers.length > 1) {
                    const allChip = el("button", "ls-chip ls-chip-all", "Add all");
                    allChip.type = "button";
                    allChip.addEventListener("click", () => { this.insertWords(triggers); this.render(); });
                    chips.append(allChip);
                }
                row.append(chips);
            }
            if (this.openBlocks.has(item.name) && blockInfo) row.append(this.blockEditor(item, info));
            return row;
        }

        blockEditor(item, info) {
            const box = el("div", "ls-blocks");
            const modelFam = blockInfo.family;
            const fams = Object.keys(blockInfo.layouts);
            // The layout: the one the values are in, else the one last chosen, else the model's.
            let fam = familyOf(item.lbw) || item._fam || (blockInfo.layouts[modelFam] ? modelFam : "sdxl");
            if (!blockInfo.layouts[fam]) fam = fams[0];
            const names = blockInfo.layouts[fam] || [];
            const vals = item.lbw && familyOf(item.lbw) === fam ? item.lbw.slice() : names.map(() => 1);

            const head = el("div", "ls-blocks-head");
            const famSel = el("span", "ls-seg");
            for (const f of fams) {
                const b = el("button", "ls-seg-btn" + (f === fam ? " ls-active" : ""), f === "sdxl" ? "SDXL" : "SD 1.x");
                b.type = "button";
                b.title = f === modelFam ? "The loaded model's layout" : "Layout for another model family";
                b.addEventListener("click", () => {
                    if (f === fam) return;
                    item.lbw = null;
                    item._fam = f;
                    this.change();
                });
                famSel.append(b);
            }
            const reset = el("button", "ls-btn ls-reset", "Reset");
            reset.type = "button";
            reset.title = "Back to full weight in every block";
            reset.addEventListener("click", () => { item.lbw = null; this.change(); });
            head.append(el("span", "ls-blocks-title", "Block weights"), famSel, el("span", "ls-spacer"), reset);
            box.append(head);
            const hint = blocksHint(info);
            if (hint) box.append(el("div", "ls-note", hint));

            const presets = el("div", "ls-chips ls-presets");
            const current = presetFor(item.lbw);
            for (const p of blockInfo.presets) {
                if (!p.values[fam]) continue;
                const on = current === p.name || (!item.lbw && p.name === "All");
                const chip = el("button", "ls-chip" + (on ? " ls-chip-on" : ""), escapeHtml(p.name));
                chip.type = "button";
                chip.title = p.about;
                chip.addEventListener("click", () => {
                    item.lbw = isNeutral(p.values[fam]) ? null : p.values[fam].slice();
                    item._fam = fam;
                    this.change();
                });
                presets.append(chip);
            }
            box.append(presets);

            const grid = el("div", "ls-grid");
            names.forEach((n, i) => {
                const cell = el("label", "ls-cell" + (Math.abs(vals[i] - 1) > 1e-9 ? " ls-cell-changed" : ""));
                cell.append(el("span", "ls-cell-name", n === "M00" ? "MID" : n === "BASE" ? "TE" : n));
                cell.title = n === "BASE" ? "BASE: the text encoder" : n;
                const input = el("input", "ls-cell-num");
                input.type = "number";
                input.step = "0.1";
                input.value = String(vals[i]);
                input.addEventListener("change", () => {
                    const v = parseFloat(input.value);
                    if (!Number.isFinite(v)) return;
                    const next = vals.slice();
                    next[i] = round(v);
                    item.lbw = isNeutral(next) ? null : next;
                    item._fam = fam;
                    this.change();
                });
                cell.append(input);
                grid.append(cell);
            });
            box.append(grid);
            return box;
        }

        promptBox() {
            return $(`#${this.tab}_prompt textarea`);
        }

        promptText() {
            const box = this.promptBox();
            return box ? box.value : "";
        }

        insertWords(words) {
            const box = this.promptBox();
            if (!box) return;
            let text = box.value;
            const lower = text.toLowerCase();
            const add = words.filter((w) => !lower.includes(w.toLowerCase()));
            if (!add.length) return;
            text = text.replace(/[\s,]+$/, "");
            box.value = (text ? text + ", " : "") + add.join(", ");
            fire(box);
        }
    }

    // ----------------------------------------------------------- floating

    const POS_KEY = (tab) => `lora-mixer-float-${tab}`;

    function floatToggle(tab) {
        return $(`#lora_mixer_float_${tab} input[type=checkbox]`);
    }

    function enabledToggle(tab) {
        return $(`#lora_mixer_${tab}-visible-checkbox`) || $(`#lora_mixer_${tab} input[type=checkbox]`);
    }

    function currentTab() {
        try {
            const c = get_uiCurrentTabContent();
            if (c && c.id) return c.id.replace(/^tab_/, "");
        } catch (e) { /* fall through */ }
        return "txt2img";
    }

    class FloatWindow {
        constructor(stack) {
            this.stack = stack;
            this.tab = stack.tab;
            this.node = el("div", "ls-float");
            this.node.hidden = true;
            const head = el("div", "ls-float-head");
            this.title = el("span", "ls-float-title", `LoRA Mixer · ${this.tab}`);
            this.power = el("label", "ls-switch ls-power");
            this.powerBox = el("input");
            this.powerBox.type = "checkbox";
            this.powerBox.title = "LoRA Mixer on / off (same as the accordion checkbox)";
            this.power.append(this.powerBox, el("span", "ls-slider"));
            this.powerBox.addEventListener("change", () => {
                const real = enabledToggle(this.tab);
                if (real && real.checked !== this.powerBox.checked) real.click();
            });
            const dock = iconButton("dock", "Back into the panel");
            dock.addEventListener("click", () => {
                const t = floatToggle(this.tab);
                if (t && t.checked) t.click();
            });
            head.append(this.power, this.title, el("span", "ls-spacer"), dock);
            this.body = el("div", "ls-float-body");
            this.node.append(head, this.body);
            // Inside the Gradio container, so theme variables and fonts apply.
            (app().querySelector(".gradio-container") || document.body).append(this.node);
            this.restore();
            this.drag(head);
            new ResizeObserver(() => this.remember()).observe(this.node);
        }

        restore() {
            let pos = null;
            try { pos = JSON.parse(localStorage.getItem(POS_KEY(this.tab)) || "null"); } catch (e) { pos = null; }
            const w = Math.min((pos && pos.w) || 480, window.innerWidth - 16);
            const h = Math.min((pos && pos.h) || 460, window.innerHeight - 16);
            const x = Math.min(Math.max((pos && pos.x) ?? window.innerWidth - w - 24, 0), window.innerWidth - 80);
            const y = Math.min(Math.max((pos && pos.y) ?? 96, 0), window.innerHeight - 48);
            Object.assign(this.node.style, { left: x + "px", top: y + "px", width: w + "px", height: h + "px" });
        }

        remember() {
            if (this.node.hidden) return;
            const r = this.node.getBoundingClientRect();
            try {
                localStorage.setItem(POS_KEY(this.tab), JSON.stringify({ x: r.left, y: r.top, w: r.width, h: r.height }));
            } catch (e) { /* storage unavailable */ }
        }

        drag(handle) {
            let sx, sy, ox, oy;
            const move = (e) => {
                const x = Math.min(Math.max(ox + e.clientX - sx, 0), window.innerWidth - 80);
                const y = Math.min(Math.max(oy + e.clientY - sy, 0), window.innerHeight - 40);
                this.node.style.left = x + "px";
                this.node.style.top = y + "px";
            };
            const up = () => {
                document.removeEventListener("pointermove", move);
                document.removeEventListener("pointerup", up);
                this.remember();
            };
            handle.addEventListener("pointerdown", (e) => {
                if (e.target.closest("button, input, label")) return;
                sx = e.clientX; sy = e.clientY;
                const r = this.node.getBoundingClientRect();
                ox = r.left; oy = r.top;
                document.addEventListener("pointermove", move);
                document.addEventListener("pointerup", up);
                e.preventDefault();
            });
        }

        sync() {
            const floating = !!(floatToggle(this.tab) || {}).checked;
            const mount = $(`#lora_mixer_mount_${this.tab} .lora-mixer-mount`);
            const want = floating ? this.body : mount;
            if (want && this.stack.root.parentNode !== want) want.append(this.stack.root);
            if (mount) mount.classList.toggle("ls-mount-away", floating);
            this.node.hidden = !(floating && currentTab() === this.tab);
            const real = enabledToggle(this.tab);
            if (real) {
                this.powerBox.checked = real.checked;
                this.node.classList.toggle("ls-float-disabled", !real.checked);
            }
        }
    }

    // --------------------------------------------------------------- boot

    function setup() {
        for (const tab of TABS) {
            if (stacks[tab]) continue;
            const mount = $(`#lora_mixer_mount_${tab} .lora-mixer-mount`);
            if (!mount || !$(`#lora_mixer_state_${tab} textarea`)) continue;
            const stack = new Stack(tab);
            mount.append(stack.root);
            stack.readState();
            stack.render();
            stack.float = new FloatWindow(stack);
            stacks[tab] = stack;
            // Turning the mixer off hands its LoRAs to the prompt; turning it on takes them back.
            const real = enabledToggle(tab);
            if (real) real.addEventListener("change", () => stack.convert(real.checked ? "in" : "out"));
            const box = stack.promptBox();
            if (box) box.addEventListener("blur", () => setTimeout(() => stack.absorbFromPrompt(), 50));
            const ft = floatToggle(tab);
            if (ft) ft.addEventListener("change", () => stack.float.sync());
        }
    }

    function tick() {
        for (const tab of Object.keys(stacks)) {
            const s = stacks[tab];
            s.readState();          // UI defaults, other scripts
            s.float.sync();
            s.absorbFromPrompt();
        }
    }

    onUiLoaded(() => {
        setup();
        Promise.all([loadCatalog(), loadBlockInfo()]).then(() => {
            Object.values(stacks).forEach((s) => s.render());
            watchFamily();
        });
        setInterval(tick, 400);
    });
})();
