"""Local, non-network dialogs for text regions and preferred translations."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image

from .preview import ImagePreview
from .shortcuts import bind_editor_keys
from .theme import BG, ACCENT, PROTECTED, center_dialog

from .errors import EnhancerError
from .translation import MAX_REGIONS, boxes_overlap, parse_replacements, validate_text


class RegionEditor(tk.Toplevel):
    def __init__(self, parent, image: Image.Image, regions=(), protected_regions=()):
        super().__init__(parent)
        self.title("Mark Translation Areas")
        self.configure(background=BG)
        self.transient(parent)
        self.geometry(f"{min(1150, self.winfo_screenwidth()-50)}x{min(780, self.winfo_screenheight()-90)}")
        self.minsize(720, 480)
        self.image = image
        self.regions = list(regions)
        self.protected_regions = list(protected_regions)
        self._history = [("Translate", b) for b in regions] + [("Protect", b) for b in protected_regions]
        self.result = self.protected_result = None
        self._start = None
        self._preview_size = image.size
        self.zoom = tk.StringVar(value="Fit")
        self.mode = tk.StringVar(value="Translate")
        self.info = tk.StringVar()
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Text areas", style="Section.TLabel").pack(side="left")
        ttk.Label(header, text="Protected areas take priority", style="Small.TLabel").pack(side="right")
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(controls, text="Translate", variable=self.mode, value="Translate",
                        style="Segment.TRadiobutton").pack(side="left")
        ttk.Radiobutton(controls, text="Protect", variable=self.mode, value="Protect",
                        style="Protect.Segment.TRadiobutton").pack(side="left", padx=(0, 16))
        ttk.Button(controls, text="Fit", command=lambda: self.change_zoom("Fit")).pack(side="left")
        ttk.Button(controls, text="100%", command=lambda: self.change_zoom("100%")).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear All", command=self.clear).pack(side="right")
        ttk.Button(controls, text="Undo", command=self.undo).pack(side="right", padx=4)
        ttk.Label(outer, text="Left-drag to mark  ·  Scroll to zoom  ·  Right-drag to pan",
                  style="Small.TLabel").pack(anchor="w", pady=(0, 8))
        bottom = ttk.Frame(outer)
        bottom.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(bottom, textvariable=self.info, style="Small.TLabel").pack(side="left")
        ttk.Button(bottom, text="Apply Areas", style="Accent.TButton", command=self.accept).pack(side="right")
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right", padx=6)
        frame = ttk.Frame(outer)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas = ImagePreview(frame, centered=False, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sx = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.scroll_x)
        sy = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.scroll_y)
        sx.grid(row=1, column=0, sticky="ew")
        sy.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.unbind("<Double-Button-1>")
        self.canvas.on_view_change = self._view_changed
        self.canvas.set_image(image)
        bind_editor_keys(self, self.accept, undo=self.undo)
        self.update_idletasks()
        self.render()
        center_dialog(self, parent)
        self.grab_set()

    def change_zoom(self, value):
        self.zoom.set(value)
        self.render()

    def _view_changed(self):
        self._start = None
        scale = self.canvas.scale
        self._preview_size = (round(self.image.width*scale), round(self.image.height*scale))
        self.zoom.set("Fit" if self.canvas.fit_mode else f"{scale*100:g}%")
        self.info.set(f"{len(self.regions)} translate · {len(self.protected_regions)} protected · {scale*100:.0f}%")

    def render(self):
        if self.zoom.get() == "Fit":
            self.canvas.fit()
        else:
            self.canvas.set_zoom(float(self.zoom.get().removesuffix("%"))/100)
        self.draw_regions()

    def draw_regions(self):
        self.canvas.overlays = tuple([(b, ACCENT, f"T{i}") for i, b in enumerate(self.regions, 1)] +
                                     [(b, PROTECTED, f"P{i}") for i, b in enumerate(self.protected_regions, 1)])
        self.canvas.render()

    def _point(self, event, *, clamp=True):
        return self.canvas.image_point(event.x, event.y, clamp=clamp)

    def start_drag(self, event):
        if not self.canvas._panning:
            self._start = self._point(event, clamp=False)

    def drag(self, event):
        self.canvas.delete("drag")
        if self._start is not None:
            x, y = self._point(event)
            self.canvas.create_rectangle(*self.canvas.screen_box((*self._start, x, y)),
                                         outline=ACCENT if self.mode.get() == "Translate" else PROTECTED,
                                         width=2, dash=(5, 3), tags="drag")

    def end_drag(self, event):
        start, self._start = self._start, None
        self.canvas.delete("drag")
        if start is None:
            return
        end = self._point(event)
        if abs(start[0]-end[0]) < 1 or abs(start[1]-end[1]) < 1:
            return
        box = (math.floor(min(start[0], end[0])), math.floor(min(start[1], end[1])),
               math.ceil(max(start[0], end[0])), math.ceil(max(start[1], end[1])))
        target = self.regions if self.mode.get() == "Translate" else self.protected_regions
        if len(target) >= MAX_REGIONS:
            messagebox.showwarning("Area Limit", "Select at most 32 areas of each kind.", parent=self)
        elif any(boxes_overlap(box, existing) for existing in target):
            messagebox.showwarning("Overlapping Areas", "Use separate boxes of the same kind. Protected boxes can overlap translation boxes.", parent=self)
        else:
            target.append(box)
            self._history.append((self.mode.get(), box))
        self.draw_regions()

    def undo(self):
        if self._history:
            kind, box = self._history.pop()
            (self.regions if kind == "Translate" else self.protected_regions).remove(box)
        self.draw_regions()

    def clear(self):
        self.regions.clear()
        self.protected_regions.clear()
        self._history.clear()
        self.draw_regions()

    def accept(self):
        self.result, self.protected_result = tuple(self.regions), tuple(self.protected_regions)
        self.destroy()


class WordingEditor(tk.Toplevel):
    def __init__(self, parent, replacements: str, protected_terms: str):
        super().__init__(parent)
        self.title("Translation Wording")
        self.transient(parent)
        self.result = None
        self.geometry("600x450")
        self.minsize(500, 380)
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", side="bottom", pady=(14, 0))
        ttk.Button(buttons, text="Apply", style="Accent.TButton", command=self.accept).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=6)
        ttk.Label(frame, text="Exact wording", style="Section.TLabel").pack(anchor="w")
        ttk.Label(frame, text="One pair per line: PLAY => SPIELEN", style="Small.TLabel").pack(anchor="w", pady=(4, 8))
        self.replacements = tk.Text(frame, height=6, wrap="word", undo=True)
        self.replacements.pack(fill="both", expand=True)
        self.replacements.insert("1.0", replacements)
        ttk.Label(frame, text="Keep unchanged", style="Section.TLabel").pack(anchor="w", pady=(14, 0))
        ttk.Label(frame, text="One name or phrase per line. Takes priority over exact wording.",
                  style="Small.TLabel", wraplength=460).pack(anchor="w", pady=(4, 8))
        self.protected_terms = tk.Text(frame, height=3, wrap="word", undo=True)
        self.protected_terms.pack(fill="both", expand=True)
        self.protected_terms.insert("1.0", protected_terms)
        bind_editor_keys(self, self.accept)
        center_dialog(self, parent)
        self.grab_set()
        self.replacements.focus_set()

    def accept(self):
        replacements = self.replacements.get("1.0", "end-1c")
        protected_terms = self.protected_terms.get("1.0", "end-1c")
        try:
            parse_replacements(replacements)
            validate_text(protected_terms, "Do not translate", 2000)
        except EnhancerError as exc:
            messagebox.showwarning("Check Translation Wording", str(exc), parent=self)
            return
        self.result = (replacements, protected_terms)
        self.destroy()
