"""Edit the instructions and optional label area for one queued image."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk

from .annotations import ImageAnnotation
from .core import Texture
from .errors import EnhancerError
from .preview import ImagePreview
from .shortcuts import bind_editor_keys, primary_label
from .theme import ACCENT, center_dialog


class AnnotationEditor(tk.Toplevel):
    def __init__(self, parent, texture: Texture, annotation: ImageAnnotation | None, remembered: bool):
        super().__init__(parent)
        self.title("Annotate image")
        self.transient(parent)
        self.result = None  # Cancel; a saved result is (annotation or None, remember).
        self.digest = texture.sha256
        changed = bool(annotation and annotation.source_sha256 != self.digest)
        self.area = annotation.area if annotation and not changed else None
        self.start = None
        self.remember = tk.BooleanVar(self, remembered)
        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=texture.path.name, style="Section.TLabel", wraplength=600).pack(anchor="w")
        if changed:
            ttk.Label(body, text="The source has changed. Review the wording and mark the area again.",
                      style="Small.TLabel").pack(anchor="w", pady=(4, 0))

        form = ttk.Frame(body)
        form.pack(fill="both", expand=True, pady=(10, 8))
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        form.rowconfigure(0, weight=1)
        art = ttk.Frame(form)
        art.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.preview = ImagePreview(art, cursor="crosshair", width=270, height=220)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<ButtonPress-1>", self._start_area)
        self.preview.bind("<B1-Motion>", self._drag_area)
        self.preview.bind("<ButtonRelease-1>", self._finish_area)
        self.preview.unbind("<Double-Button-1>")
        self.preview.set_image(texture.image)
        self.area_label = tk.StringVar(self)
        controls = ttk.Frame(art)
        controls.pack(fill="x", pady=(5, 0))
        ttk.Label(controls, textvariable=self.area_label, style="Small.TLabel").pack(side="left")
        self.clear_area = ttk.Button(controls, text="Clear area", command=lambda: self._set_area(None))
        self.clear_area.pack(side="right")
        self._set_area(self.area)

        fields = ttk.Frame(form)
        fields.grid(row=0, column=1, sticky="nsew")
        ttk.Label(fields, text="Exact text").pack(anchor="w")
        self.exact = tk.Text(fields, height=2, width=30, wrap="word", undo=True)
        self.exact.pack(fill="x", pady=(3, 10))
        ttk.Label(fields, text="Visual note (optional)").pack(anchor="w")
        self.note = tk.Text(fields, height=3, width=30, wrap="word", undo=True)
        self.note.pack(fill="x", pady=(3, 8))
        if annotation:
            self.exact.insert("1.0", annotation.exact_text)
            self.note.insert("1.0", annotation.note)
        ttk.Label(fields, text="Drag a box around the label and its accents to keep the surrounding artwork locally upscaled.",
                  wraplength=280, style="Small.TLabel").pack(anchor="w")
        ttk.Checkbutton(fields, text="Remember for this file", variable=self.remember).pack(anchor="w", pady=(6, 0))
        ttk.Label(body, text="Annotated images use OpenAI. Run Queue shows the paid request count before uploading.",
                  wraplength=600, style="Small.TLabel").pack(anchor="w", pady=(3, 8))

        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        if annotation:
            ttk.Button(buttons, text="Remove annotation", command=self.remove).pack(side="left")
        ttk.Button(buttons, text=f"Save ({primary_label(self)}+Enter)", style="Accent.TButton",
                   command=self.accept).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=6)
        bind_editor_keys(self, self.accept)
        self.geometry("670x430")
        self.minsize(550, 380)
        center_dialog(self, parent)
        self.grab_set()
        self.exact.focus_set()

    def _start_area(self, event):
        self.start = self.preview.image_point(event.x, event.y)

    def _drag_area(self, event):
        self.preview.delete("drag")
        if self.start is not None:
            end = self.preview.image_point(event.x, event.y, clamp=True)
            self.preview.create_rectangle(*self.preview.screen_box((*self.start, *end)), outline=ACCENT,
                                          width=2, dash=(5, 3), tags="drag")

    def _finish_area(self, event):
        start, self.start = self.start, None
        self.preview.delete("drag")
        if start is None:
            return
        end = self.preview.image_point(event.x, event.y, clamp=True)
        if abs(end[0] - start[0]) >= 1 and abs(end[1] - start[1]) >= 1:
            self._set_area((math.floor(min(start[0], end[0])), math.floor(min(start[1], end[1])),
                            math.ceil(max(start[0], end[0])), math.ceil(max(start[1], end[1]))))

    def _set_area(self, area):
        self.area = area
        self.preview.overlays = ((area, ACCENT, ""),) if area else ()
        self.preview.render()
        self.area_label.set("Label area selected" if area else "Whole image")
        self.clear_area.configure(state="normal" if area else "disabled")

    def accept(self):
        wording = self.exact.get("1.0", "end-1c")
        annotation = ImageAnnotation(self.digest, wording if wording.strip() else "",
                                     self.note.get("1.0", "end-1c").strip(), self.area)
        try:
            annotation.validate()
            annotation.check_bounds(self.preview.source.size)
        except EnhancerError as exc:
            messagebox.showwarning("Check Annotation", str(exc), parent=self)
            return
        self.result = (annotation, self.remember.get())
        self.destroy()

    def remove(self):
        self.result = (None, False)
        self.destroy()
