"""Small desktop UI; workers communicate exclusively through a queue."""
from __future__ import annotations

import os
import json
from dataclasses import replace
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from PIL import Image, ImageDraw, ImageTk

from . import APP_NAME, __version__
from .annotation_gui import AnnotationEditor
from .annotations import AnnotationStore, ImageAnnotation, finish_annotated_area
from .api import Result, default_recovery_directory, private_recovery_directory, redact, run_job, validate_key
from .batch import BatchEvent, BatchSummary, SavedExport, collect_files, prepare_batch, run_batch, request_count
from .core import EnhancerError, MAX_OUTPUT_EDGE, MODEL, QUALITIES, Settings, Texture, load_texture, plan_local, plan_settings, save_image_atomic, needs_api, source_coverage, is_up_to_date
from .formats.common import MAX_OUTPUT_FILE_BYTES, open_output_image
from .formats import EXTENSIONS, SavedImage, encoded_size, orient_for_export, output_name, output_profile, read_saved_export
from .storage import default_data_directory, save_original_atomic, write_atomic
from .translation import LANGUAGES, composite_translation, parse_replacements, map_box
from .translation_gui import RegionEditor, WordingEditor
from .theme import apply_theme, center_dialog, BG, PANEL, MUTED, ACCENT, PROTECTED
from .preview import ImagePreview
from .shortcuts import install_workspace_keys, show_shortcuts


EDGE_PRESETS = {"2K (2048)": 2048, "4K (4096)": 4096}
SCALE_PRESETS = {f"{factor}× original": factor for factor in range(2, 9)}
CUSTOM_EDGE = "Custom longest edge"
CUSTOM_SCALE = "Custom scale factor"
NEURAL_METHOD = "Neural artwork (free)"
SIZE_CHOICES = (*EDGE_PRESETS, CUSTOM_EDGE, "Original size", *SCALE_PRESETS, CUSTOM_SCALE)
FORMAT_CHOICES = {"Retain original": "original", "JPG / JPEG": "JPEG", "PNG": "PNG", "TGA": "TGA", "DDS": "DDS"}
IMAGE_FILTERS = [("Supported images", " ".join("*" + e for ext in EXTENSIONS for e in (ext, ext.upper()))),
                 ("JPEG images", "*.jpg *.jpeg *.JPG *.JPEG"), ("All files", "*.*")]


def output_target(selection: str, edge_text: str, factor_text: str) -> tuple[int | float, int | None]:
    if selection in SCALE_PRESETS:
        return SCALE_PRESETS[selection], None
    if selection == "Original size":
        return 1, None
    if selection in EDGE_PRESETS:
        return 2, EDGE_PRESETS[selection]
    if selection == CUSTOM_EDGE:
        try:
            return 2, int(edge_text)
        except ValueError as exc:
            raise EnhancerError(f"Enter a whole number from 1 to {MAX_OUTPUT_EDGE} for the longest edge.") from exc
    if selection == CUSTOM_SCALE:
        value = factor_text.strip().lower()
        if value.endswith(("x", "×")):
            value = value[:-1].strip()
        try:
            return float(value.replace(",", ".")), None
        except ValueError as exc:
            raise EnhancerError("Enter a scale factor such as 1.5, 10 or 12.") from exc
    raise EnhancerError("Choose an output size.")


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class App(tk.Tk):
    def __init__(self, *, runner: Callable = run_job, recovery_directory: Path | None = None,
                 prompt_for_key: bool = True):
        super().__init__(className="ImageEnhancer")
        apply_theme(self)
        self.title(f"{APP_NAME} {__version__}")
        width, height = min(1280, self.winfo_screenwidth() - 40), min(800, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(min(width, 1000), min(height, 640))
        self._runner = runner
        self._queue: queue.Queue[tuple[Result | None, str | None]] = queue.Queue()
        self._busy = False
        self._started_at: float | None = None
        self._file_started_at: float | None = None
        self._active_paid = False
        self._batch_active = False
        self._batch_stop = threading.Event()
        self._batch_messages = queue.Queue()
        self._batch_paths: list[Path] = []
        self._annotations: dict[Path, ImageAnnotation] = {}
        self._remembered: set[Path] = set()
        self._annotation_store = AnnotationStore(default_data_directory() / "annotations")
        self._batch_details: dict[int, str] = {}
        self._batch_outputs: dict[int, Path] = {}
        self._saved_exports: dict[Path, SavedExport] = {}
        self._preview_path: Path | None = None
        self._saved_view: SavedImage | None = None
        self._batch_done: set[int] = set()
        self._batch_rows: tuple[int, ...] = ()
        self._batch_settings: Settings | None = None
        self._batch_key = ""
        self._batch_rerun = False
        self._rerun_snapshot = {}
        self._closed = False
        self._startup_id: str | None = None
        self._poll_id: str | None = None
        self._preview_id: str | None = None
        self._photos: list[ImageTk.PhotoImage | None] = [None, None]
        self._editable: list[tuple[tk.Widget, str]] = []
        self.texture: Texture | None = None
        self.result: Result | None = None
        self.recovery_directory = recovery_directory or default_recovery_directory()
        self._preferences_path = None if recovery_directory else default_recovery_directory().parent / "preferences.json"
        if self._preferences_path:
            try:
                if self._preferences_path.stat().st_size <= 8192:
                    preferences = json.loads(self._preferences_path.read_text(encoding="utf-8"))
                    location = preferences.get("output_directory") if isinstance(preferences, dict) else None
                    if isinstance(location, str) and location and Path(location).is_absolute():
                        self.recovery_directory = Path(location)
            except (OSError, ValueError):
                pass
        self.output_location = tk.StringVar(value=str(self.recovery_directory))
        self.output_format = tk.StringVar(value="Retain original")
        self._last_output_directory = self.recovery_directory
        self.key = tk.StringVar(value=os.environ.get("OPENAI_API_KEY", ""))
        self.task = tk.StringVar(value="Enhance artwork")
        self.source_language = tk.StringVar(value="Auto-detect")
        self.target_language = tk.StringVar(value="German")
        self.translation_scope = tk.StringVar(value="Selected text areas")
        self.keep_names = tk.BooleanVar(value=True)
        self.text_regions: tuple[tuple[int, int, int, int], ...] = ()
        self.protected_regions = ()
        self.optimize_translation = tk.BooleanVar(value=True)
        self.reuse_identical = tk.BooleanVar(value=True)
        self.keep_intermediates = tk.BooleanVar(value=False)
        self.request_hint = tk.StringVar(value="")
        self.replacements = ""
        self.protected_terms = ""
        self.translation_summary = tk.StringVar(value="Select the text areas to translate.")
        self._previous_task = "Enhance artwork"
        self._enhance_size = "2K (2048)"
        self._translation_size = "Original size"
        self._enhance_method = "OpenAI enhancement"
        self.output_size = tk.StringVar(value="2K (2048)")
        self.custom_edge = tk.StringVar(value="2048")
        self.custom_scale = tk.StringVar(value="10")
        self.quality = tk.StringVar(value="max")
        self.method = tk.StringVar(value="OpenAI enhancement")
        self.protection = tk.StringVar(value="Detail recovery")
        self.instruction = tk.StringVar()
        self.preserve_alpha = tk.BooleanVar(value=True)
        self.filename = tk.StringVar(value="Add images to get started")
        self.dimensions = tk.StringVar(value="JPG / JPEG · PNG · TGA · DDS")
        self.elapsed = tk.StringVar(value="Elapsed: 00:00:00")
        self.size_hint = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self._build()
        self._setup_drop()
        install_workspace_keys(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.task.trace_add("write", lambda *_: self._task_changed())
        self.translation_scope.trace_add("write", lambda *_: self._update_translation_summary())
        self.source_language.trace_add("write", lambda *_: self._update_plan())
        self.target_language.trace_add("write", lambda *_: self._update_plan())
        self.output_size.trace_add("write", lambda *_: self._update_plan())
        self.output_format.trace_add("write", lambda *_: self._update_plan())
        self.custom_edge.trace_add("write", lambda *_: self._update_plan())
        self.custom_scale.trace_add("write", lambda *_: self._update_plan())
        self.method.trace_add("write", lambda *_: self._update_plan())
        self.quality.trace_add("write", lambda *_: self._update_plan())
        self._task_changed()
        self._poll_id = self.after(100, self._poll)
        if prompt_for_key:
            self._startup_id = self.after(100, self._startup_setup)

    def _edit(self, widget: tk.Widget, state: str = "normal") -> tk.Widget:
        self._editable.append((widget, state))
        return widget

    def _build(self) -> None:
        outer = ttk.Frame(self)
        self.drop_surface = outer
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer, padding=(16, 10))
        header.pack(fill="x")
        mark = tk.Canvas(header, width=26, height=28, background=BG, highlightthickness=0)
        mark.pack(side="left", padx=(0, 9))
        for y, color in ((17, "#4b4669"), (12, "#71668f"), (7, ACCENT)):
            mark.create_polygon(2, y, 13, y-5, 24, y, 13, y+5, fill="", outline=color, width=2)
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left", padx=(0, 24))
        self.open_button = self._edit(ttk.Button(header, text="Add Images", command=self.add_files))
        self.open_button.pack(side="left", padx=(0, 6))
        self._edit(ttk.Button(header, text="Add Folder", style="Quiet.TButton", command=self.add_folder)).pack(side="left")
        ttk.Button(header, text="Shortcuts", style="Quiet.TButton", command=lambda: show_shortcuts(self)).pack(side="right")
        self._edit(ttk.Button(header, text="API Settings", style="Quiet.TButton", command=self.api_settings)).pack(side="right")
        ttk.Separator(header, orient="vertical").pack(side="right", fill="y", padx=10, pady=5)
        ttk.Button(header, text="Open Output Folder", style="Quiet.TButton", command=self.open_output_folder).pack(side="right")
        ttk.Separator(outer).pack(fill="x")

        bottom = ttk.Frame(outer, padding=(16, 7))
        bottom.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=90)
        self.elapsed_label = ttk.Label(bottom, textvariable=self.elapsed, style="Small.TLabel")
        self.elapsed_label.pack(side="right")
        self.status_label = ttk.Label(bottom, textvariable=self.status, style="Small.TLabel", wraplength=600)
        self.status_label.pack(side="left", fill="x", expand=True)
        ttk.Separator(outer).pack(side="bottom", fill="x")

        workspace = ttk.Frame(outer, padding=(12, 12, 12, 10))
        workspace.pack(fill="both", expand=True)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(0, weight=1)
        rail = ttk.Frame(workspace, width=280, style="Card.TFrame", padding=(12, 12))
        rail.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        rail.grid_propagate(False)
        rail.rowconfigure(1, weight=1)
        rail.columnconfigure(0, weight=1)
        ttk.Label(rail, text="Workspace", style="CardHeading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.settings_canvas = tk.Canvas(rail, width=240, highlightthickness=0, background=PANEL)
        self.settings_canvas.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(rail, orient="vertical", command=self.settings_canvas.yview)
        scroll.grid(row=1, column=1, sticky="ns", padx=(5, 0))
        def show_settings_scrollbar(first, last):
            scroll.set(first, last)
            if float(first) <= 0 and float(last) >= 1:
                scroll.grid_remove()
            else:
                scroll.grid()
        self.settings_canvas.configure(yscrollcommand=show_settings_scrollbar)
        controls = ttk.Frame(self.settings_canvas, style="Card.TFrame", padding=(0, 0, 4, 8))
        window = self.settings_canvas.create_window(0, 0, window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind("<Configure>", lambda e: self.settings_canvas.itemconfigure(window, width=e.width))

        def label(parent, title):
            ttk.Label(parent, text=title, style="CardMuted.TLabel").pack(anchor="w", pady=(13, 5))

        def combo(parent, variable, values, *, editable=False):
            widget = self._edit(ttk.Combobox(parent, textvariable=variable, values=values,
                                           state="normal" if editable else "readonly", width=18),
                                "normal" if editable else "readonly")
            widget.pack(fill="x", pady=(0, 3))
            return widget

        tasks = ttk.Frame(controls, style="Card.TFrame")
        tasks.pack(fill="x", pady=(0, 2))
        for title, value in (("Upscale", "Enhance artwork"), ("Translate", "Translate text")):
            self._edit(ttk.Radiobutton(tasks, text=title, value=value, variable=self.task,
                                      style="Segment.TRadiobutton")).pack(side="left", fill="x", expand=True)
        self.mode_options = ttk.Frame(controls, style="Card.TFrame")
        self.mode_options.pack(fill="x")
        self.enhance_options = ttk.Frame(self.mode_options, style="Card.TFrame")
        self.enhance_options.pack(fill="x")
        label(self.enhance_options, "Method")
        combo(self.enhance_options, self.method, ("OpenAI enhancement", NEURAL_METHOD, "Faithful resize (free)", "Pixel art (free)"))
        self.method_hint = tk.StringVar()
        ttk.Label(self.enhance_options, textvariable=self.method_hint, style="CardSmall.TLabel", wraplength=230).pack(anchor="w", pady=(3, 0))
        self.translation_options = ttk.Frame(self.mode_options, style="Card.TFrame")
        label(self.translation_options, "Source language")
        combo(self.translation_options, self.source_language, ("Auto-detect", *LANGUAGES), editable=True)
        label(self.translation_options, "Target language")
        combo(self.translation_options, self.target_language, LANGUAGES, editable=True)
        label(self.translation_options, "Text selection")
        combo(self.translation_options, self.translation_scope, ("Selected text areas", "All text (batch compatible)"))
        area_actions = ttk.Frame(self.translation_options, style="Card.TFrame")
        area_actions.pack(fill="x", pady=(6, 4))
        self._edit(ttk.Button(area_actions, text="Mark Areas", command=self.select_text_areas)).pack(side="left")
        self._edit(ttk.Button(area_actions, text="Wording", command=self.edit_wording)).pack(side="left", padx=6)
        ttk.Label(self.translation_options, textvariable=self.translation_summary, style="CardSmall.TLabel", wraplength=230).pack(anchor="w", pady=(0, 6))
        self._edit(ttk.Checkbutton(self.translation_options, text="Keep names and logos", style="Card.TCheckbutton", variable=self.keep_names)).pack(anchor="w")
        self._edit(ttk.Checkbutton(self.translation_options, text="Optimize request area", style="Card.TCheckbutton", variable=self.optimize_translation,
                                  command=self._update_plan)).pack(anchor="w")
        label(controls, "Output size")
        combo(controls, self.output_size, SIZE_CHOICES).configure(height=len(SIZE_CHOICES))
        size_line = ttk.Frame(controls, style="Card.TFrame")
        size_line.pack(fill="x")
        self.custom_size_frame = ttk.Frame(size_line, style="Card.TFrame")
        ttk.Label(self.custom_size_frame, text="Longest edge", style="CardSmall.TLabel").pack(side="left")
        self._edit(ttk.Entry(self.custom_size_frame, textvariable=self.custom_edge, width=7)).pack(side="left", padx=6)
        ttk.Label(self.custom_size_frame, text="px", style="CardSmall.TLabel").pack(side="left")
        self.custom_scale_frame = ttk.Frame(size_line, style="Card.TFrame")
        ttk.Label(self.custom_scale_frame, text="Scale factor", style="CardSmall.TLabel").pack(side="left")
        self._edit(ttk.Entry(self.custom_scale_frame, textvariable=self.custom_scale, width=7)).pack(side="left", padx=6)
        ttk.Label(self.custom_scale_frame, text="×", style="CardSmall.TLabel").pack(side="left")
        ttk.Label(controls, textvariable=self.size_hint, style="CardSmall.TLabel", wraplength=230).pack(anchor="w", pady=(3, 0))
        self.enhance_edits = ttk.Frame(controls, style="Card.TFrame")
        self.enhance_edits.pack(fill="x")
        label(self.enhance_edits, "Artwork protection")
        combo(self.enhance_edits, self.protection, ("Detail recovery", "Strong", "Moderate", "Off (allow edits)"))
        self._edit(ttk.Checkbutton(self.enhance_edits, text="Keep original alpha", style="Card.TCheckbutton", variable=self.preserve_alpha)).pack(anchor="w", pady=(3, 0))
        label(self.enhance_edits, "Additional instruction")
        self._edit(ttk.Entry(self.enhance_edits, textvariable=self.instruction)).pack(fill="x")
        ttk.Label(self.enhance_edits, text="Requires artwork protection Off", style="CardSmall.TLabel").pack(anchor="w", pady=(4, 0))
        self.actions_frame = ttk.Frame(controls, style="Card.TFrame")
        self.actions_frame.pack(fill="x")
        label(self.actions_frame, "Output format")
        combo(self.actions_frame, self.output_format, tuple(FORMAT_CHOICES))
        self.format_hint = tk.StringVar()
        ttk.Label(self.actions_frame, textvariable=self.format_hint, style="CardSmall.TLabel", wraplength=230).pack(anchor="w", pady=(3, 0))
        label(self.actions_frame, "Save to")
        self._edit(ttk.Entry(self.actions_frame, textvariable=self.output_location, state="readonly"), "readonly").pack(fill="x")
        output_actions = ttk.Frame(self.actions_frame, style="Card.TFrame")
        output_actions.pack(fill="x", pady=(5, 7))
        self._edit(ttk.Button(output_actions, text="Choose Folder", command=self.choose_output)).pack(side="left")
        ttk.Button(output_actions, text="Open Folder", command=self.open_output_folder).pack(side="left", padx=(6, 0))
        fixed_actions = ttk.Frame(rail, style="Card.TFrame", padding=(0, 12, 0, 0))
        fixed_actions.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Separator(fixed_actions).pack(fill="x", pady=(0, 10))
        self.run_button = ttk.Button(fixed_actions, text="Enhance Image", style="Accent.TButton", command=self._run_active)
        self.run_button.pack(fill="x")
        ttk.Label(fixed_actions, textvariable=self.request_hint, style="CardSmall.TLabel", wraplength=250).pack(anchor="w", pady=(7, 0))

        main = ttk.Frame(workspace)
        main.grid(row=0, column=1, sticky="nsew")
        self.tabs = ttk.Notebook(main)
        self.tabs.pack(fill="both", expand=True)
        panels = ttk.Frame(self.tabs, padding=(12, 12), style="Card.TFrame")
        self.preview_tab = panels
        self.tabs.add(panels, text="Preview")
        self._build_batch_tab()
        self.tabs.enable_traversal()
        self.tabs.bind("<<NotebookTabChanged>>", self._preview_tab_changed)
        self.tabs.bind("<ButtonRelease-1>", self._focus_tab_content)
        ttk.Label(panels, textvariable=self.filename, style="CardHeading.TLabel", wraplength=650).pack(anchor="w", pady=(0, 4))
        self.dimensions_label = ttk.Label(panels, textvariable=self.dimensions, style="CardSmall.TLabel", wraplength=650)
        self.dimensions_label.pack(anchor="w", pady=(0, 12))
        navigation = ttk.Frame(panels, style="Card.TFrame")
        navigation.pack(fill="x", pady=(0, 8))
        self.previous_button = self._navigation_button(navigation, -1)
        self.previous_button.pack(side="left")
        self.preview_position = tk.StringVar()
        ttk.Label(navigation, textvariable=self.preview_position, style="CardSmall.TLabel").pack(side="left", padx=6)
        self.next_button = self._navigation_button(navigation, 1)
        self.next_button.pack(side="left", padx=(0, 12))
        self.preview_annotate_button = ttk.Button(navigation, text="Annotate (A)", command=self.annotate_selected, state="disabled")
        self.preview_annotate_button.pack(side="left", padx=(0, 8))
        ttk.Button(navigation, text="Fit", command=lambda: self.set_preview_zoom(fit=True)).pack(side="left")
        ttk.Button(navigation, text="100%", command=lambda: self.set_preview_zoom(fit=False)).pack(side="left", padx=4)
        ttk.Label(navigation, text="Scroll to zoom  ·  Right-drag to pan", style="CardSmall.TLabel").pack(side="right")
        exports = ttk.Frame(panels, style="Card.TFrame")
        exports.pack(side="bottom", fill="x", pady=(10, 0))
        self.save_button = ttk.Button(exports, text="Save Copy", command=self.save_result, state="disabled")
        self.save_button.pack(side="left")
        self.png_button = ttk.Button(exports, text="Export PNG", command=self.export_png, state="disabled")
        self.png_button.pack(side="left", padx=6)
        self._edit(ttk.Button(exports, text="Import Edited PNG", command=self.import_edited_png)).pack(side="right")
        views = ttk.Frame(panels, style="Card.TFrame")
        views.pack(fill="both", expand=True)
        for column in (0, 1):
            views.columnconfigure(column, weight=1, uniform="previews")
        views.rowconfigure(0, weight=1)
        for column, name in enumerate(("Original", "Result")):
            view = ttk.Frame(views, style="Preview.TFrame", padding=1)
            view.grid(row=0, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
            caption = ttk.Frame(view, style="Card.TFrame", padding=(9, 6))
            caption.pack(fill="x")
            if column == 1:
                self.result_caption = tk.StringVar(value=name)
                ttk.Label(caption, textvariable=self.result_caption, style="CardMuted.TLabel").pack(side="left")
            else:
                ttk.Label(caption, text=name, style="CardMuted.TLabel").pack(side="left")
            zoom = tk.StringVar(value="—")
            ttk.Label(caption, textvariable=zoom, style="CardSmall.TLabel").pack(side="right")
            preview = ImagePreview(view, width=1, height=100, takefocus=True, empty_text="Add images or drop a folder" if column == 0 else "Your result appears here")
            preview.configure(highlightthickness=1, highlightbackground=preview["background"], highlightcolor=ACCENT)
            preview.bind("<Button-1>", lambda event: event.widget.focus_set(), add="+")
            preview.pack(fill="both", expand=True)
            preview.on_view_change = lambda p=preview, z=zoom: z.set(f"{p.scale*100:.0f}%")
            if column == 0:
                self.input_preview = preview
            else:
                self.output_preview = preview
        self.bind("<Configure>", self._update_wrap)
        def scroll_settings(event):
            number = getattr(event, "num", None)
            delta = getattr(event, "delta", 0)
            step = -1 if number == 4 or delta > 0 else 1
            if number in (4, 5) or delta:
                self.settings_canvas.yview_scroll(step * 3, "units")
            return "break"
        pending = [controls, self.settings_canvas]
        while pending:
            widget = pending.pop()
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(event, scroll_settings)
            pending.extend(widget.winfo_children())

    def _run_active(self) -> None:
        if self._batch_paths and not (self.task.get() == "Translate text" and
                                     (self.translation_scope.get() == "Selected text areas" or self.protected_regions)):
            self.process_batch()
        else:
            self.process()

    def _startup_setup(self) -> None:
        self._startup_id = None
        if not self._closed and (self._busy or not self.winfo_viewable()):
            self._startup_id = self.after(100, self._startup_setup)
        elif not self._closed:
            self.api_settings(startup=True)

    def api_settings(self, *, startup: bool = False) -> None:
        if self._busy or self._closed:
            return
        dialog = tk.Toplevel(self)
        dialog.title("Connect OpenAI" if startup else "API Settings")
        dialog.transient(self)
        dialog.configure(background=BG)
        dialog.geometry("470x390")
        dialog.minsize(450, 380)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", side="bottom", pady=(12, 0))
        key, quality = tk.StringVar(value=self.key.get()), tk.StringVar(value=self.quality.get())
        keep = tk.BooleanVar(value=self.keep_intermediates.get())
        ttk.Label(frame, text="Connect OpenAI" if startup else "OpenAI rendering", style="Section.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Enter your key to start enhancing images." if startup else MODEL,
                  style="Small.TLabel").pack(anchor="w", pady=(4, 12))
        ttk.Label(frame, text="API key · kept in memory only").pack(anchor="w")
        entry = ttk.Entry(frame, textvariable=key, show="*")
        entry.pack(fill="x", pady=(5, 14))
        ttk.Label(frame, text="Rendering quality").pack(anchor="w")
        ttk.Combobox(frame, textvariable=quality, values=QUALITIES, state="readonly").pack(fill="x", pady=5)
        ttk.Label(frame, text="Max is the highest quality and uses more output tokens.",
                  style="Small.TLabel", wraplength=410).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(frame, text="Keep intermediate PNGs after a successful export", variable=keep).pack(anchor="w", pady=(8, 0))
        ttk.Button(frame, text="Open Recovery Folder", command=lambda: self._open_directory(private_recovery_directory())).pack(anchor="w", pady=(7, 0))
        def apply():
            try:
                validated = validate_key(key.get()) if startup or key.get().strip() else ""
            except EnhancerError as exc:
                messagebox.showwarning("API Key Required", str(exc), parent=dialog)
                entry.focus_set()
                return
            self.key.set(validated)
            self.quality.set(quality.get())
            self.keep_intermediates.set(keep.get())
            dialog.destroy()
        def cancel():
            if startup:
                self.method.set(NEURAL_METHOD)
                self.status.set("Local mode selected. Open API Settings to enable OpenAI enhancement.")
            dialog.destroy()
        ttk.Button(buttons, text="Continue" if startup else "Save", style="Accent.TButton", command=apply).pack(side="right")
        ttk.Button(buttons, text="Use Local Mode" if startup else "Cancel", command=cancel).pack(side="right", padx=8)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Escape>", lambda _: cancel())
        dialog.bind("<Return>", lambda _: apply())
        center_dialog(dialog, self)
        dialog.wait_visibility()
        dialog.lift()
        dialog.grab_set()
        entry.focus_set()
        self.wait_window(dialog)
        key.set("")

    def _settings(self, *, for_plan: bool = False) -> Settings:
        method = {NEURAL_METHOD: "neural", "OpenAI enhancement": "openai", "Faithful resize (free)": "lanczos", "Pixel art (free)": "nearest",
                  "Local Lanczos": "lanczos", "Local nearest": "nearest"}.get(self.method.get(), "invalid")
        scale, target = output_target(self.output_size.get(), self.custom_edge.get(), self.custom_scale.get())
        protection = {"Detail recovery": "detail", "Strong": "strict", "Moderate": "balanced", "Off (allow edits)": "off"}.get(self.protection.get(), "invalid")
        translating = self.task.get() == "Translate text"
        if self.task.get() not in ("Enhance artwork", "Translate text"):
            raise EnhancerError("Choose Enhance artwork or Translate text.")
        scope = {"Selected text areas": "regions", "All text (batch compatible)": "all"}.get(self.translation_scope.get(), "invalid")
        # dimension previews must not disappear while language/wording selections are incomplete
        settings = Settings(scale=scale, quality=self.quality.get(),
                            instruction="" if translating or for_plan else self.instruction.get(),
                            preserve_alpha=False if translating else self.preserve_alpha.get(),
                            method="openai" if translating else method, target_edge=target,
                            source_protection="off" if translating else protection,
                            task="translate" if translating and not for_plan else "enhance",
                            source_language=self.source_language.get(), target_language=self.target_language.get(),
                            translation_scope=scope,
                            text_regions=self.text_regions if scope == "regions" else (),
                            replacements=self.replacements, protected_terms=self.protected_terms,
                            keep_names=self.keep_names.get(), protected_regions=self.protected_regions if translating else (),
                            optimize_translation=self.optimize_translation.get(), reuse_identical=self.reuse_identical.get(),
                            keep_intermediates=self.keep_intermediates.get(), output_format=FORMAT_CHOICES[self.output_format.get()])
        settings.validate()
        return settings

    def _task_changed(self) -> None:
        translating = self.task.get() == "Translate text"
        if self.task.get() != self._previous_task:
            if translating:
                self._enhance_size = self.output_size.get()
                self._enhance_method = self.method.get()
                self.method.set("OpenAI enhancement")
                self.output_size.set(self._translation_size)
            else:
                self._translation_size = self.output_size.get()
                self.method.set(self._enhance_method)
                self.output_size.set(self._enhance_size)
            self._previous_task = self.task.get()
        if translating:
            self.enhance_options.pack_forget()
            self.enhance_edits.pack_forget()
            self.translation_options.pack(fill="x")
        else:
            self.translation_options.pack_forget()
            self.enhance_options.pack(fill="x")
        self._update_translation_summary()
        self._update_plan()

    def _update_translation_summary(self) -> None:
        text = (f"{len(self.text_regions)} translate" if self.translation_scope.get() == "Selected text areas"
                else "All text")
        if self.protected_regions:
            text += f" · {len(self.protected_regions)} protected"
        count = len(parse_replacements(self.replacements))
        if count:
            text += f" · {count} wording pairs"
        self.translation_summary.set(text)
        self._refresh_previews()
        self._update_plan()

    def select_text_areas(self) -> None:
        if self._busy:
            return
        if not self.texture:
            messagebox.showwarning("Open Original First", "Open the image whose text you want to translate.", parent=self)
            return
        dialog = RegionEditor(self, self.texture.image, self.text_regions, self.protected_regions)
        self.wait_window(dialog)
        if dialog.result is not None:
            self.text_regions = dialog.result
            self.protected_regions = dialog.protected_result
            if self.text_regions:
                self.translation_scope.set("Selected text areas")
            self._update_translation_summary()

    def edit_wording(self) -> None:
        if self._busy:
            return
        dialog = WordingEditor(self, self.replacements, self.protected_terms)
        self.wait_window(dialog)
        if dialog.result is not None:
            self.replacements, self.protected_terms = dialog.result
            self._update_translation_summary()

    @staticmethod
    def _translation_description(settings: Settings) -> str:
        if settings.task != "translate":
            return ""
        area = (f"{len(settings.text_regions)} selected text areas; outside pixels restored before encoding"
                if settings.translation_scope == "regions" else "All text; the AI may change background artwork")
        return f"Translation: {settings.source_language} to {settings.target_language}\n{area}; {len(settings.protected_regions)} protected areas\n"

    def _build_batch_tab(self) -> None:
        self.batch_tab = ttk.Frame(self.tabs, padding=12, style="Card.TFrame")
        self.tabs.add(self.batch_tab, text="Files")
        toolbar = ttk.Frame(self.batch_tab, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 10))
        self.batch_run_button = self.run_button
        self.batch_stop_button = ttk.Button(toolbar, text="Stop After Current", command=self.stop_batch, state="disabled")
        self.batch_stop_button.pack(side="left")
        ttk.Label(toolbar, text="Find file", style="CardSmall.TLabel").pack(side="left", padx=(12, 4))
        self.file_search = ttk.Entry(toolbar, width=20)
        self.file_search.pack(side="left")
        self.file_search.bind("<Return>", self._find_file)
        self._edit(ttk.Checkbutton(toolbar, text="Reuse identical images", style="Card.TCheckbutton", variable=self.reuse_identical)).pack(side="right")
        self.drop_label = ttk.Label(self.batch_tab, text="One image at a time · Originals stay untouched", style="CardSmall.TLabel")
        self.drop_label.pack(anchor="w", pady=(0, 10))
        table_actions = ttk.Frame(self.batch_tab, style="Card.TFrame")
        table_actions.pack(fill="x", pady=(0, 7))
        self.rerun_button = ttk.Button(table_actions, text="Run Again", command=lambda: self.process_batch(rerun=True), state="disabled")
        self.rerun_button.pack(side="left")
        self._edit(ttk.Button(table_actions, text="Preview Selected", command=self.preview_selected)).pack(side="left", padx=6)
        self.annotate_button = ttk.Button(table_actions, text="Annotate (A)", command=self.annotate_selected, state="disabled")
        self.annotate_button.pack(side="left")
        self.clear_button = self._edit(ttk.Button(table_actions, text="Clear", command=self.clear_batch))
        self.clear_button.pack(side="right")
        self.remove_button = self._edit(ttk.Button(table_actions, text="Remove Selected", command=self.remove_batch))
        self.remove_button.pack(side="right", padx=(0, 6))
        table = ttk.Frame(self.batch_tab, style="Card.TFrame")
        table.pack(fill="both", expand=True)
        self.batch_tree = ttk.Treeview(table, columns=("name", "target", "state", "input", "annotation"),
                                       displaycolumns=("name", "input", "target", "annotation", "state"), show="headings", height=5, selectmode="extended")
        for column, title, width in (("name", "Source file", 300), ("input", "Input pixels", 108), ("target", "Output pixels", 115), ("annotation", "Annotation", 105), ("state", "Status", 90)):
            self.batch_tree.heading(column, text=title)
            self.batch_tree.column(column, width=width, minwidth=80, stretch=column == "name")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.batch_tree.pack(side="left", fill="both", expand=True)
        self.batch_tree.bind("<<TreeviewSelect>>", self._batch_selection)
        self.batch_tree.bind("<Double-1>", lambda _: self.preview_selected())
        self.batch_detail = tk.StringVar()
        ttk.Label(self.batch_tab, textvariable=self.batch_detail, style="CardSmall.TLabel", wraplength=600).pack(anchor="w", pady=(7, 0))
        self.batch_count = tk.StringVar(value="0 files queued")
        ttk.Label(self.batch_tab, textvariable=self.batch_count, style="CardSmall.TLabel").pack(anchor="w")

    def _focus_file_search(self, _event=None):
        self.tabs.select(self.batch_tab)
        self.file_search.focus_set()
        self.file_search.selection_range(0, "end")
        return "break"

    def show_files(self):
        index = self._preview_index() if self.tabs.select() == str(self.preview_tab) else -1
        self.tabs.select(self.batch_tab)
        if index >= 0:
            row = str(index)
            self.batch_tree.selection_set(row)
            self.batch_tree.focus(row)
            self.batch_tree.see(row)
        self.batch_tree.focus_set()

    def set_preview_zoom(self, *, fit):
        for view in (self.input_preview, self.output_preview):
            if fit:
                view.fit()
            else:
                view.set_zoom(1)

    def _focus_tab_content(self, _event):
        target = self.input_preview if self.tabs.select() == str(self.preview_tab) else self.batch_tree
        target.focus_set()

    def _find_file(self, _event=None):
        query = self.file_search.get().strip().casefold()
        if not query:
            return "break"
        count = len(self._batch_paths)
        selected = self.batch_tree.selection()
        start = int(selected[0]) + 1 if selected else 0
        for offset in range(count):
            index = (start + offset) % count
            if query in self._batch_paths[index].name.casefold():
                row = str(index)
                self.batch_tree.selection_set(row)
                self.batch_tree.focus(row)
                self.batch_tree.see(row)
                self._batch_selection()
                return "break"
        self.batch_tree.selection_remove(*selected)
        self.batch_detail.set("")
        self._annotation_button_state()
        self.status.set("No queued filename matches your search.")
        return "break"

    def _setup_drop(self) -> None:
        self.drag_drop_enabled = False
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            TkinterDnD.require(self)
            for widget in (self.drop_surface, self.input_preview, self.output_preview, self.batch_tree, self.drop_label):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._drop)
            self.drag_drop_enabled = True
        except (ImportError, RuntimeError, tk.TclError, OSError):
            self.drop_label.configure(text="Use Add Files or Add Folder. Drag-and-drop is unavailable.")

    def _drop(self, event) -> str:
        if self._busy:
            return "refuse_drop"
        try:
            paths = tuple(Path(p) for p in self.tk.splitlist(event.data))
        except (tk.TclError, TypeError, ValueError):
            self.status.set("Could not read the dropped file list.")
            return "refuse_drop"
        if not paths:
            return "refuse_drop"
        self.add_paths(paths)
        return "copy"

    def choose_output(self) -> None:
        if self._busy:
            return
        name = filedialog.askdirectory(parent=self, title="Choose Output Location", initialdir=str(self.recovery_directory))
        if name:
            self.recovery_directory = Path(name).absolute()
            self._last_output_directory = self.recovery_directory
            self.output_location.set(str(self.recovery_directory))
            self.status.set("Output location: " + str(self.recovery_directory))
            if self._preferences_path:
                try:
                    self._preferences_path.parent.mkdir(parents=True, exist_ok=True)
                    write_atomic(json.dumps({"output_directory": str(self.recovery_directory)}).encode("utf-8"),
                                 self._preferences_path, overwrite=True)
                except (OSError, EnhancerError):
                    self.status.set("Output folder selected for this session; the preference could not be saved.")

    def add_files(self) -> None:
        if self._busy:
            return
        names = filedialog.askopenfilenames(parent=self, filetypes=IMAGE_FILTERS)
        if names:
            self.add_paths(tuple(Path(n) for n in names))

    def add_folder(self) -> None:
        if self._busy:
            return
        name = filedialog.askdirectory(parent=self, title="Add Texture Folder (Includes Subfolders)")
        if name:
            self.add_paths((Path(name),))

    def add_paths(self, paths: tuple[Path, ...]) -> None:
        if self._busy:
            return
        self.show_files()
        self._set_busy(True)
        self.status.set("Scanning files and folders...")
        existing = tuple(self._batch_paths)
        destination = self.recovery_directory
        def work():
            try:
                files, notes = collect_files(paths, existing, output_directory=destination)
                remembered, warnings = self._annotation_store.load(files)
                self._batch_messages.put(("collected", (files, notes + warnings, remembered)))
            except Exception as exc:
                self._batch_messages.put(("error", redact(str(exc), key)))
        key = self.key.get()
        self._start_batch_worker(work)

    def remove_batch(self) -> None:
        if self._busy:
            return
        selected = set(self.batch_tree.selection())
        remaining = [(p, self.batch_tree.item(str(i), "values"), self._batch_details.get(i, ""), self._batch_outputs.get(i))
                     for i, p in enumerate(self._batch_paths) if str(i) not in selected]
        kept = {p for p, *_ in remaining}
        annotations = {p: note for p, note in self._annotations.items() if p in kept}
        remembered = self._remembered & kept
        exports = {p: saved for p, saved in self._saved_exports.items() if p in kept}
        self.clear_batch(retain_preview=self._preview_path in kept)
        self._annotations.update(annotations)
        self._remembered.update(remembered)
        self._saved_exports.update(exports)
        for i, (path, values, detail, output) in enumerate(remaining):
            self._batch_paths.append(path)
            self.batch_tree.insert("", "end", iid=str(i), values=values)
            self._batch_details[i] = detail
            if output:
                self._batch_outputs[i] = output
            if values[2] == "Done":
                self._batch_done.add(i)
        self.batch_count.set(f"{len(self._batch_paths)} files in queue")
        self._update_plan()

    def clear_batch(self, *, retain_preview=False) -> None:
        if self._busy:
            return
        if not retain_preview and self._preview_path in self._batch_paths and (not self.result or self.result.saved_path):
            self._preview_path = self.texture = self.result = self._saved_view = None
            self.text_regions = self.protected_regions = ()
            self._update_translation_summary()
            self.filename.set("Add images to get started")
            self.dimensions.set("JPG / JPEG · PNG · TGA · DDS")
            self.save_button.configure(state="disabled")
            self.png_button.configure(state="disabled")
            self._refresh_previews()
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)
        self._batch_paths.clear()
        self._annotations.clear()
        self._remembered.clear()
        self._batch_details.clear()
        self._batch_outputs.clear()
        self._saved_exports.clear()
        self._batch_done.clear()
        self.batch_detail.set("")
        self.batch_count.set("0 files queued")
        self._update_plan()

    def preview_selected(self) -> None:
        if self._busy:
            return
        selection = self.batch_tree.selection()
        if not selection:
            return
        self._show_queue_row(int(selection[0]))
        if self.tabs.select() == str(self.preview_tab):
            self.input_preview.focus_set()

    def _navigation_button(self, parent, direction):
        # Keep the arrows visible even when the system's Tk font lacks arrow glyphs.
        arrow = Image.new("RGBA", (48, 48))
        pen = ImageDraw.Draw(arrow)
        pen.line([(24, 8), (8, 24), (24, 40)], fill="#e6e9ef", width=5)
        pen.line([(10, 24), (41, 24)], fill="#e6e9ef", width=5)
        if direction > 0:
            arrow = arrow.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        side = max(12, round(self.winfo_fpixels("10p")))
        icon = ImageTk.PhotoImage(arrow.resize((side, side), Image.Resampling.LANCZOS), master=self)
        button = ttk.Button(parent, text="Previous" if direction < 0 else "Next", image=icon,
                            compound="left" if direction < 0 else "right", command=lambda: self._step_preview(direction))
        button.image = icon
        return button

    def _preview_index(self):
        try:
            return self._batch_paths.index(self._preview_path)
        except ValueError:
            return -1

    def _step_preview(self, direction):
        if not self._busy and self.tabs.select() == str(self.preview_tab):
            index = self._preview_index() + direction
            if 0 <= index < len(self._batch_paths):
                self._show_queue_row(index)

    def _preview_tab_changed(self, _event=None):
        self._update_plan()
        if self._busy or not self._batch_paths or self.tabs.select() != str(self.preview_tab):
            return
        if self.result and self.result.saved_path is None:
            return
        rows = self.batch_tree.selection()
        index = int(rows[0]) if rows else max(0, self._preview_index())
        self._show_queue_row(index)

    def _show_queue_row(self, index):
        path = self._batch_paths[index]
        same = path == self._preview_path
        if not same and not self._may_discard():
            return
        self.batch_tree.selection_set(str(index))
        self.batch_tree.focus(str(index))
        self.batch_tree.see(str(index))
        self.tabs.select(self.preview_tab)
        saved = self._saved_exports.get(path)
        if same and (self.result and self.result.saved_path is None or
                     self._saved_view and saved and self._saved_view.path == saved.path):
            return
        if not saved:
            if (not same or not self.texture) and not self.load_file(path, check_discard=False):
                self._apply_preview(path, None, None, None)
            return
        current = self.result if same and self.result and self.result.original_path == saved.path else None
        self._open_saved_preview(path, saved, current)

    def _open_saved_preview(self, path, saved, current):
        self._set_busy(True, timed=False)
        self.status.set("Opening saved output…")
        def work():
            texture = view = None
            errors = []
            try:
                texture = load_texture(path)
                if texture.sha256 != saved.source_sha256:
                    texture.image.close()
                    texture = None
                    errors.append("The original has changed since this output was saved.")
            except Exception as exc:
                errors.append("Original unavailable. " + str(exc))
            try:
                view = read_saved_export(saved.path, saved.size)
            except Exception as exc:
                errors.append("Saved output unavailable. " + str(exc))
            self._batch_messages.put(("preview", (path, texture, saved, view, current, errors)))
        self._start_batch_worker(work)

    def _apply_preview(self, path, texture, result, view):
        self._preview_path, self.texture, self.result, self._saved_view = path, texture, result, view
        self.text_regions = self.protected_regions = ()
        self._update_translation_summary()
        self.filename.set(path.name)
        if not texture:
            self.dimensions.set("Original unavailable")
        self._set_busy(False)
        self._refresh_previews()

    @staticmethod
    def _annotation_label(annotation):
        if not annotation:
            return ""
        return " ".join(annotation.exact_text.split())[:60] or "Visual note"

    def _annotation_index(self):
        if self._busy or self.task.get() != "Enhance artwork":
            return None
        if self.tabs.select() == str(self.preview_tab):
            index = self._preview_index()
            return index if index >= 0 else None
        rows = self.batch_tree.selection()
        return int(rows[0]) if len(rows) == 1 else None

    def annotate_selected(self) -> None:
        index = self._annotation_index()
        if index is None:
            return
        row = str(index)
        path = self._batch_paths[index]
        old = self._annotations.get(path)
        try:
            texture = load_texture(path)
        except (OSError, EnhancerError) as exc:
            messagebox.showerror("Cannot Open Image", redact(str(exc), self.key.get()), parent=self)
            return
        if self.tabs.select() == str(self.preview_tab):
            return_to = self.input_preview
        else:
            return_to = self.file_search if self.focus_get() is self.file_search else self.batch_tree
        try:
            dialog = AnnotationEditor(self, texture, old, path in self._remembered)
            self.wait_window(dialog)
        finally:
            texture.image.close()
        if self._closed:
            return
        return_to.focus_set()
        if dialog.result is None:
            return
        annotation, remember = dialog.result
        if remember or path in self._remembered:
            try:
                self._annotation_store.save(path, annotation if remember else None)
            except (OSError, EnhancerError) as exc:
                messagebox.showerror("Cannot Save Annotation", redact(str(exc), self.key.get()), parent=self)
                return
        self._remembered.discard(path)
        if remember:
            self._remembered.add(path)
        if annotation == old:
            return
        if annotation is None:
            self._annotations.pop(path, None)
        else:
            self._annotations[path] = annotation
        self.batch_tree.set(row, "annotation", self._annotation_label(annotation))
        self.batch_tree.set(row, "state", "Queued")
        self._batch_done.discard(index)
        self._batch_details[index] = "Annotation updated. Run Queue to process this file."
        self.batch_count.set(f"{len(self._batch_done)} / {len(self._batch_paths)} files completed")
        self._batch_selection()
        self._update_plan()

    def _annotation_button_state(self):
        state = "normal" if self._annotation_index() is not None else "disabled"
        self.annotate_button.configure(state=state)
        self.preview_annotate_button.configure(state=state)

    def _batch_selection(self, _event=None) -> None:
        self._annotation_button_state()
        selection = self.batch_tree.selection()
        if selection:
            index = int(selection[0])
            detail = str(self._batch_paths[index]) + " | " + self._batch_details.get(index, "Not checked yet")
            if index in self._batch_outputs:
                detail += " | Saved: " + str(self._batch_outputs[index])
            self.batch_detail.set(redact(detail, self.key.get()))
        else:
            self.batch_detail.set("")

    def _start_batch_worker(self, work: Callable) -> bool:
        try:
            threading.Thread(target=work, daemon=True, name="batch-worker").start()
            return True
        except Exception:
            self._finish_batch_ui()
            messagebox.showerror("Cannot Start Worker", "The background worker could not be started. No new request was sent.", parent=self)
            return False

    def process_batch(self, *, rerun: bool = False) -> None:
        if self._busy:
            return
        if not self._batch_paths:
            messagebox.showwarning("No Images", "Add images or a folder first.", parent=self)
            return
        self._batch_rows = tuple(i for i in range(len(self._batch_paths)) if self.batch_tree.set(str(i), "state") in ("Queued", "Ready"))
        rerun = rerun or not self._batch_rows
        if rerun:
            self._batch_rows = tuple(range(len(self._batch_paths)))
        paths = tuple(self._batch_paths[i] for i in self._batch_rows)
        try:
            settings = self._settings()
            if settings.task == "translate" and (settings.translation_scope == "regions" or settings.protected_regions):
                raise EnhancerError("Selected text areas belong to one image. Use Translate Image for one source, or choose All text without areas for a batch.")
            annotations = {p: self._annotations[p] for p in paths if p in self._annotations} if settings.task == "enhance" else {}
            key = self.key.get() if settings.method == "openai" or annotations else ""
        except (ValueError, EnhancerError) as exc:
            messagebox.showwarning("Cannot Start", str(exc), parent=self)
            return
        if not self._may_discard():
            return
        self._batch_rerun = rerun
        self._rerun_snapshot = ({row: (self.batch_tree.item(str(row), "values"), self._batch_details.get(row, ""))
                                 for row in self._batch_rows} if rerun else {})
        self._batch_settings, self._batch_key = settings, key
        self._batch_stop.clear()
        self._batch_active = True
        self._set_busy(True)
        self.batch_stop_button.configure(state="normal")
        self.status.set("Checking queue…")
        def work():
            try:
                items = prepare_batch(paths, settings, self._batch_stop, self._batch_event, annotations=annotations)
                self._batch_messages.put(("prepared", items))
            except Exception as exc:
                self._batch_messages.put(("error", redact(str(exc), key)))
        self._start_batch_worker(work)

    def _batch_event(self, event: BatchEvent) -> None:
        # worker threads never access Tk, including StringVar values
        self._batch_messages.put(("event", (event, time.monotonic())))

    def stop_batch(self) -> None:
        if self._batch_active:
            self._batch_stop.set()
            self.batch_stop_button.configure(state="disabled")
            self.status.set("Stopping after the current operation saves its output. An API request already sent cannot be cancelled.")

    def _finish_batch_ui(self) -> None:
        # a cancelled or failed preflight must not turn completed rows into pending jobs
        for row, (values, detail) in self._rerun_snapshot.items():
            self.batch_tree.item(str(row), values=values)
            self._batch_details[row] = detail
        self._rerun_snapshot.clear()
        self._batch_rerun = False
        self._batch_active = False
        self._batch_key = ""
        self._batch_settings = None
        self._set_busy(False)
        self.batch_stop_button.configure(state="disabled")

    def _prepared_batch(self, items) -> None:
        if self._batch_stop.is_set():
            self._finish_batch_ui()
            self.status.set("Queue checking stopped. No API request was sent.")
            return
        valid = tuple(item for item in items if not item.error)
        if not valid:
            details = "\n".join(f"{item.path.name}: {redact(item.error, self._batch_key)}" for item in items[:10] if item.error)
            rerun = self._batch_rerun
            self._finish_batch_ui()
            self.status.set("No compatible images in this queue. Select a row for details.")
            if rerun and details:
                messagebox.showwarning("Cannot Run Again", details, parent=self)
            return
        settings, key = self._batch_settings, self._batch_key
        count = request_count(valid, settings)
        if count:
            try:
                key = validate_key(key)
            except EnhancerError as exc:
                self._finish_batch_ui()
                messagebox.showwarning("API Key Required", str(exc) + " Enter it in API Settings, then run the queue again.",
                                       parent=self)
                return
            large = sum(item.plan.resampled_output for item in valid)
            paid = [item for item in valid if item.needs_request(settings)]
            reused = len(paid) - count
            image_label = "image" if len(valid) == 1 else "images"
            edit_label = "edit" if count == 1 else "edits"
            message = (f"Process {len(valid)} {image_label} and submit up to {count} paid image {edit_label}?\n\n"
                       f"Model: {MODEL}\nQuality: {settings.quality}\n"
                       f"{self._translation_description(settings)}"
                       f"Matching images sharing an earlier result: {reused}\n"
                       f"Invalid files skipped: {len(items) - len(valid)}\n"
                       f"Images needing final local enlargement: {large}\n\n"
                       "Every filename gets an output. API credits or paid balance are consumed. "
                       "The tool cannot check your balance or enforce a money limit. No automatic retries. "
                       "The queue stops on the first processing failure.\n\n"
                       "Review the dimensions in the queue. AI can change text and sprite alignment. Continue?")
        elif self._batch_rerun:
            message = f"Process {len(valid)} images again with the current settings?\n\nThis run makes no API requests."
        if count or self._batch_rerun:
            annotated = [item for item in valid if item.annotation]
            if annotated:
                names = ", ".join(redact(item.path.name, key) for item in annotated[:3])
                if len(annotated) > 3:
                    names += f" (+{len(annotated) - 3})"
                local_count = sum(not item.needs_request(settings) for item in valid)
                message += f"\n\nAnnotated files using OpenAI ({len(annotated)}): {names}\nFiles processed locally: {local_count}"
            sample = annotated[0] if annotated else valid[0]
            start, end = sample.plan.source_size, sample.output_size or sample.plan.target_size
            factor = f" ({settings.scale:g}×)" if settings.target_edge is None else ""
            message += (f"\n\nOutput format: {self.output_format.get()}\n{redact(sample.path.name, key)}\n"
                        f"{start[0]} x {start[1]} → {end[0]} x {end[1]} pixels{factor}")
            if self._batch_rerun:
                message = ("You are running this queue again. AI results are generated again and may incur new charges. "
                           "Existing outputs are kept; new copies receive numbered filenames.\n\n" if count else
                           "You are running this queue again. Existing outputs are kept; new copies receive numbered filenames.\n\n") + message
            if not messagebox.askyesno("Run Queue Again" if self._batch_rerun else "Confirm Batch API Requests", message, parent=self, default="no"):
                self._finish_batch_ui()
                self.status.set("Batch not started. No API request was sent.")
                return
        # processing duration excludes preflight and time spent in the approval dialog
        self._started_at = time.monotonic()
        self._file_started_at = None
        self._update_elapsed()
        directory = self.recovery_directory
        self._active_paid = bool(count)
        self.status.set(f"Processing {len(valid)} files, one at a time. Use Stop After Current to stop safely.")
        runner = self._runner
        def work():
            try:
                summary = run_batch(items, settings, key, directory, self._batch_stop, self._batch_event, runner=runner)
                self._batch_messages.put(("finished", summary))
            except Exception as exc:
                self._batch_messages.put(("error", redact(str(exc), key)))
        if self._start_batch_worker(work):
            self._rerun_snapshot.clear()
            self._batch_done.difference_update(self._batch_rows)

    def _poll_batch(self) -> None:
        for _ in range(100):
            try:
                kind, payload = self._batch_messages.get_nowait()
            except queue.Empty:
                return
            if kind == "collected":
                paths, notes, remembered = payload
                self._annotations.update(remembered)
                self._remembered.update(remembered)
                for path in paths:
                    index = len(self._batch_paths)
                    self._batch_paths.append(path)
                    self.batch_tree.insert("", "end", iid=str(index), values=(redact(path.name, self.key.get()), "—", "Queued", "—",
                                                            self._annotation_label(remembered.get(path))))
                self._set_busy(False)
                self.batch_count.set(f"{len(self._batch_paths)} files in queue")
                self.status.set(redact(f"Added {len(paths)} files. " + " ".join(notes), self.key.get()))
                if paths and len(paths) == len(self._batch_paths) and not self.batch_tree.selection():
                    self.batch_tree.selection_set("0")
                    self.batch_tree.focus("0")
                if len(self._batch_paths) == 1:
                    self.preview_selected()
                self._update_plan()
            elif kind == "preview":
                path, texture, saved, view, result, errors = payload
                if result is None and view and texture:
                    profile = output_profile(texture.profile, saved.settings.output_format)
                    working = orient_for_export(view.image, profile, inverse=True)
                    result = Result(working, path, saved.settings, saved.plan, None, saved.path, "", {}, (),
                                    texture.profile, saved.path)
                self._apply_preview(path, texture, result, view)
                self.status.set(redact(" ".join(errors) if errors else f"Saved output · {saved.path.name}", self.key.get()))
            elif kind == "event":
                event, occurred_at = payload
                if event.state == "Processing":
                    self._file_started_at = occurred_at
                elif event.state in ("Done", "Failed", "Ready"):
                    self._file_started_at = None
                row = self._batch_rows[event.index]
                self.batch_tree.set(str(row), "state", event.state)
                if event.source_dimensions:
                    self.batch_tree.set(str(row), "input", event.source_dimensions)
                if event.dimensions:
                    self.batch_tree.set(str(row), "target", event.dimensions)
                self._batch_details[row] = redact(event.message, self._batch_key)
                if event.output:
                    self._batch_outputs[row] = event.output
                if event.saved:
                    self._saved_exports[self._batch_paths[row]] = event.saved
                if event.state == "Done":
                    self._batch_done.add(row)
                self.batch_count.set(f"{len(self._batch_done)} / {len(self._batch_paths)} files completed")
                if event.state == "Processing":
                    self.batch_tree.see(str(row))
                    self.status.set(f"Processing {row + 1} / {len(self._batch_paths)} · {self._batch_paths[row].name}")
            elif kind == "prepared":
                self._prepared_batch(payload)
            elif kind == "finished":
                summary = payload
                if summary.last_result is not None:
                    self.result = summary.last_result
                    self.texture = summary.last_texture
                    self._preview_path = self.texture.path
                    self._saved_view = None
                    row = str(self._batch_paths.index(self.texture.path))
                    self.batch_tree.selection_set(row)
                    self.batch_tree.focus(row)
                    self.text_regions = ()
                    self.protected_regions = ()
                    self._update_translation_summary()
                    self.filename.set(self.texture.path.name)
                    self._refresh_previews()
                    self._update_plan()
                self._finish_batch_ui()
                self._last_output_directory = summary.directory
                self.status.set(f"Batch {'stopped' if summary.cancelled or summary.warning else 'finished'}: {summary.completed} saved, {summary.failed} failed. Use Preview to browse results.")
                if summary.warning:
                    recovery = f"\n\nRecovery files: {summary.recovery_directory}" if summary.recovery_directory else ""
                    messagebox.showwarning("Batch Stopped", summary.warning + recovery, parent=self)
                elif summary.last_result and self.tabs.select() == str(self.preview_tab):
                    self._show_queue_row(self._preview_index())
            elif kind == "error":
                self._finish_batch_ui()
                self.status.set("Batch stopped. Completed outputs are retained; no automatic retry was made.")
                messagebox.showerror("Batch Stopped", payload, parent=self)

    def _update_wrap(self, event: tk.Event) -> None:
        if not self._closed and event.widget is self:
            self.status_label.configure(wraplength=max(350, event.width - 440))
            self.dimensions_label.configure(wraplength=max(350, event.width - 355))

    def _schedule_preview(self, _event: tk.Event | None = None) -> None:
        if self._closed:
            return
        if self._preview_id:
            self.after_cancel(self._preview_id)
        self._preview_id = self.after(120, self._refresh_previews)

    def _preview(self, widget: ImagePreview, image: Image.Image | None, slot: int) -> None:
        overlays = []
        if slot == 0 and self.task.get() == "Translate text":
            if self.translation_scope.get() == "Selected text areas":
                overlays.extend((box, ACCENT, f"T{i}") for i, box in enumerate(self.text_regions, 1))
            overlays.extend((box, PROTECTED, f"P{i}") for i, box in enumerate(self.protected_regions, 1))
        widget.overlays = tuple(overlays)
        widget.set_image(image)

    def _refresh_previews(self) -> None:
        if self._preview_id:
            self.after_cancel(self._preview_id)
        self._preview_id = None
        if not self._closed:
            self._preview(self.input_preview, self.texture.image if self.texture else None, 0)
            shown = self._saved_view.image if self._saved_view else self.result.image if self.result else None
            self._preview(self.output_preview, shown, 1)
            title = f"Saved {self._saved_view.path.suffix[1:].upper()}" if self._saved_view else "Result"
            self.result_caption.set(f"{title} · {shown.width} × {shown.height}" if shown else "Result")

    def _update_plan(self) -> None:
        self._annotation_button_state()
        index, count = self._preview_index(), len(self._batch_paths)
        self.previous_button.configure(state="normal" if not self._busy and index > 0 else "disabled")
        self.next_button.configure(state="normal" if not self._busy and index+1 < count else "disabled")
        self.preview_position.set(f"{index+1} / {count}" if index >= 0 else f"{count} files" if count else "0 / 0")
        self.format_hint.set({"Retain original": "Keep each source encoding.", "JPG / JPEG": "JPG has no transparency. Artwork uses a white background.",
                              "PNG": "Lossless color and transparency.", "TGA": "New TGA files use 32-bit color.",
                              "DDS": "New DDS files use uncompressed RGBA and one mip level."}[self.output_format.get()])
        selection = self.output_size.get()
        for choice, frame in ((CUSTOM_EDGE, self.custom_size_frame), (CUSTOM_SCALE, self.custom_scale_frame)):
            if selection == choice:
                frame.pack(side="left")
            else:
                frame.pack_forget()
        if selection in SCALE_PRESETS:
            self.size_hint.set(f"Multiply width and height by {SCALE_PRESETS[selection]}.")
        elif selection in EDGE_PRESETS:
            self.size_hint.set(f"{EDGE_PRESETS[selection]} px longest edge · no downscaling")
        else:
            self.size_hint.set({"Original size": "Keep original dimensions",
                                CUSTOM_EDGE: "Preserve proportions · no downscaling",
                                CUSTOM_SCALE: "Multiply width and height, for example by 10 or 12."}.get(selection, "Choose an output size."))
        translating = self.task.get() == "Translate text"
        local = self.method.get() != "OpenAI enhancement" and not translating
        self.method_hint.set("Resizes your pixels without redrawing." if self.method.get() in ("Faithful resize (free)", "Local Lanczos")
                             else "Reconstructs contours and transparency with a local neural model." if self.method.get() == NEURAL_METHOD
                             else "Crisp pixel edges; no smoothing." if local
                             else "Generative edit · may change details")
        if not local and not translating:
            self.enhance_edits.pack(fill="x", before=self.actions_frame)
        else:
            self.enhance_edits.pack_forget()
        batch = bool(self._batch_paths) and not (translating and (self.translation_scope.get() == "Selected text areas" or self.protected_regions))
        states = tuple(self.batch_tree.set(row, "state") for row in self.batch_tree.get_children())
        pending = any(state in ("Queued", "Ready") for state in states)
        attempted = any(state in ("Done", "Failed", "Invalid") for state in states)
        self.rerun_button.configure(state="normal" if attempted and not self._busy else "disabled")
        self.run_button.configure(text=("Run Queue" if pending else "Run Again") if batch else "Translate Image" if self.task.get() == "Translate text"
                                  else "Upscale Image" if local else "Enhance Image")
        has_annotated_jobs = self._annotations and (not pending or any(
            path in self._annotations and state in ("Queued", "Ready") for path, state in zip(self._batch_paths, states)))
        if batch and local and has_annotated_jobs:
            self.request_hint.set("Annotated files use OpenAI · review requests before run")
        elif batch:
            self.request_hint.set("Local processing · no API charge" if local else f"{self.quality.get().title()} quality · review requests before run")
        if not self.texture:
            if not batch:
                self.request_hint.set("Local processing · no API charge" if local else f"{self.quality.get().title()} quality · OpenAI")
            return
        try:
            annotation = self._annotations.get(self.texture.path)
            settings = self._settings(for_plan=True).for_image(annotation)
            plan = plan_settings(self.texture.image.size, settings)
            try:
                actual = self._settings().for_image(annotation)
                plan = plan_settings(self.texture.image.size, actual)
            except EnhancerError:
                actual = settings
            if needs_api(actual, plan):
                hint = f"{actual.quality.title()} · {plan.size_string} · 1 request"
                if plan.source_crop:
                    hint += "\nOptimized crop"
                if not batch:
                    self.request_hint.set(hint)
            elif not batch:
                self.request_hint.set("Local processing · no API charge")
            profile = output_profile(self.texture.profile, actual.output_format) if self.texture.profile else None
            size = encoded_size(profile, plan.target_size) if profile else plan.target_size
            text = (f"Original: {self.texture.image.width} x {self.texture.image.height}  |  "
                    f"Output: {size[0]} x {size[1]} pixels")
            if actual.annotation:
                text += " | Annotated OpenAI edit"
            elif local:
                text += " | Local processing, no API charge"
            elif plan.resampled_output:
                text += " | AI result enlarged locally to this size"
            if profile:
                text += f"\n{profile.format} output" + (" · Original format retained" if profile is self.texture.profile else "")
            if settings.target_edge is not None and max(plan.target_size) != settings.target_edge:
                if is_up_to_date(actual, plan):
                    text += " | Already at target · " + ("copied without processing" if profile is self.texture.profile else "converted locally")
                elif actual.annotation and plan.target_size == plan.source_size:
                    text += " | Correct artwork at the original size"
                else:
                    text += " | Proportions rounded to the nearest pixel"
            self.dimensions.set(text)
        except (ValueError, EnhancerError) as exc:
            self.dimensions.set(str(exc))
            if not batch:
                self.request_hint.set("Review the output settings")

    def _may_discard(self) -> bool:
        return (self.result is None or self.result.saved_path is not None or
                messagebox.askyesno("Unsaved result", "This result has not been saved. Discard it?", parent=self, default="no"))

    def open_file(self) -> None:
        if self._busy:
            return
        name = filedialog.askopenfilename(parent=self, filetypes=IMAGE_FILTERS)
        if name:
            self.load_file(Path(name))

    def load_file(self, path: Path, *, check_discard=True) -> bool:
        if self._busy:
            return False
        try:
            texture = load_texture(path)
        except Exception as exc:
            messagebox.showerror("Cannot Open Texture", redact(str(exc), self.key.get()), parent=self)
            return False
        if check_discard and not self._may_discard():
            texture.image.close()
            return False
        self.texture = texture
        self._preview_path, self._saved_view = texture.path, None
        self.text_regions = ()
        self.protected_regions = ()
        self._update_translation_summary()
        self.result = None
        self.filename.set(texture.path.name[:100])
        self.save_button.configure(state="disabled")
        self.png_button.configure(state="disabled")
        self.status.set("Ready · " + texture.format + " image loaded")
        self._update_plan()
        self._refresh_previews()
        return True

    def _update_elapsed(self) -> None:
        if self._started_at is None:
            return
        now = time.monotonic()
        label = f"Elapsed: {format_elapsed(now - self._started_at)}"
        if self._file_started_at is not None:
            label += f"  |  Current file: {format_elapsed(now - self._file_started_at)}"
        self.elapsed.set(label)

    def _set_busy(self, value: bool, *, timed=True) -> None:
        if value and not self._busy and timed:
            self._started_at = time.monotonic()
            self._file_started_at = None
            self._update_elapsed()
        elif not value and self._busy:
            self._update_elapsed()
            self._started_at = None
            self._file_started_at = None
        self._busy = value
        for widget, original in self._editable:
            widget.configure(state="disabled" if value else original)
        self.run_button.configure(state="disabled" if value else "normal")
        self.save_button.configure(state="normal" if not value and self.result else "disabled")
        self.png_button.configure(state="normal" if not value and self.result else "disabled")
        if value:
            self.progress.pack(side="right", before=self.elapsed_label, padx=(12, 0))
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self._active_paid = False
            self.progress.stop()
            self.progress.pack_forget()
            self.progress.configure(mode="determinate", value=0)
        self._update_plan()

    def process(self) -> None:
        if self._busy:
            return
        if self.texture is None:
            messagebox.showwarning("No Texture", "Open a texture first.", parent=self)
            return
        try:
            settings = self._settings().for_image(self._annotations.get(self.texture.path))
            settings.validate()
            if settings.annotation:
                settings.annotation.check_source(self.texture.sha256)
            plan = plan_settings(self.texture.image.size, settings)
            use_api = needs_api(settings, plan)
            key = validate_key(self.key.get()) if use_api else ""
            if self.texture.profile is None:
                raise EnhancerError("Reopen the source to capture its original format settings.")
            if (self.texture.profile.data_texture and settings.method in ("neural", "openai")
                    and not is_up_to_date(settings, plan)):
                raise EnhancerError("Select Faithful resize or Pixel art for channel/data textures.")
            if use_api and self.texture.image.getchannel("A").getextrema()[1] == 0:
                raise EnhancerError("This texture is completely transparent; there is no visible artwork to enhance.")
        except (ValueError, EnhancerError) as exc:
            messagebox.showwarning("Cannot Start", str(exc), parent=self)
            return
        if not self._may_discard():
            return
        warning = "\nThis canvas exceeds the usual 2K range; large resolutions are experimental." if plan.experimental else ""
        if plan.resampled_output:
            warning += "\nThe exact output dimensions require local enlargement after the AI edit."
        size = encoded_size(output_profile(self.texture.profile, settings.output_format), plan.target_size)
        confirmed = not use_api or messagebox.askyesno(
            "Confirm One Paid API Request",
            f"Upload this texture to OpenAI and submit ONE paid edit?\n\n"
            f"Model: {MODEL}\nQuality: {settings.quality}\nAPI canvas: {plan.size_string}\n"
            f"Final output: {size[0]} x {size[1]} · {self.output_format.get()}\n{warning}\n"
            f"{self._translation_description(settings)}\n"
            "This consumes API credit or paid balance. This tool cannot verify your credit grant, "
            "remaining balance or final cost. No automatic retries are made. "
            "An unsuccessful or interrupted request may still be charged.\n\n"
            "AI can change text and sprite alignment. Continue?", parent=self, default="no")
        if not confirmed:
            return
        # capture a stable request snapshot; later UI values cannot rename or corrupt this result
        texture = self.texture
        self._active_paid = use_api
        self._set_busy(True)
        self.status.set("Processing image…")
        try:
            threading.Thread(target=self._work, args=(texture, settings, key), daemon=True,
                             name="image-edit-worker").start()
        except Exception:
            self._set_busy(False)
            self.status.set("Could not start the worker. No request was sent.")
            messagebox.showerror("Cannot Start Worker", "The background worker could not be started.", parent=self)

    def _work(self, texture: Texture, settings: Settings, key: str) -> None:
        try:
            result = self._runner(texture, settings, key, self.recovery_directory)
            self._queue.put((result, None))
        except Exception as exc:
            self._queue.put((None, redact(str(exc), key)))
        # no Tk calls, self.after calls, or unredacted exception logging on this thread

    def _poll(self) -> None:
        self._poll_id = None
        if self._closed:
            return
        self._update_elapsed()
        self._poll_batch()
        try:
            result, error = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            if error:
                self.status.set("Processing failed. No automatic API retry was made. The previous result, if any, is retained.")
                messagebox.showerror("Processing Failed", error, parent=self)
            elif result is not None:
                self.result = result
                self._saved_view = None
                self._last_output_directory = result.original_path.parent if result.original_path else self.recovery_directory
                self.save_button.configure(state="normal")
                self.png_button.configure(state="normal")
                self._refresh_previews()
                saved = ("Saved. Use Open Output Folder to view files." if result.original_path else
                         "Recovery PNG saved. Use Save Copy to export." if result.saved_path else
                         "Use Save Copy to keep this result.")
                self.status.set(f"Done: {result.image.width} x {result.image.height}. {saved}")
                if result.usage:
                    counts = " · ".join(f"{result.usage[name]:,} {label}" for name, label in
                                       (("input_tokens", "input tokens"), ("output_tokens", "output tokens")) if name in result.usage)
                    self.request_hint.set("Last API response: " + (counts or f"{result.usage.get('total_tokens', 0):,} tokens"))
                if result.warnings:
                    recovery_path = result.png_path or result.raw_path
                    recovery = f"\n\nRecovery file: {recovery_path}" if recovery_path else ""
                    messagebox.showwarning("Output Recovery Warning", "\n\n".join(result.warnings) + recovery, parent=self)
                elif result.original_path and result.profile and self.texture:
                    size = encoded_size(output_profile(result.profile, result.settings.output_format), result.plan.target_size)
                    saved = SavedExport(result.original_path, self.texture.sha256, result.settings, result.plan, size)
                    if result.source_path in self._batch_paths:
                        self._saved_exports[result.source_path] = saved
                    if self.tabs.select() == str(self.preview_tab):
                        self._open_saved_preview(result.source_path, saved, result)
        self._poll_id = self.after(100, self._poll)

    def import_edited_png(self) -> None:
        if self._busy:
            return
        if self.texture is None:
            messagebox.showwarning("Open Original First", "Open the original DDS/TGA/PNG/JPG first so its format settings can be retained.", parent=self)
            return
        name = filedialog.askopenfilename(parent=self, filetypes=[("Finished PNG", "*.png")])
        if not name:
            return
        try:
            from .formats import png as png_codec
            from .formats.common import Profile, to_srgb
            source = self.texture
            settings = self._settings().for_image(self._annotations.get(source.path))
            if settings.annotation:
                settings.annotation.check_source(source.sha256)
            detail_alpha = settings.method == "neural" or (settings.method == "openai" and settings.source_protection == "detail")
            if settings.task == "translate" or settings.annotation:
                plan = plan_local(source.image.size, settings.scale, target_edge=settings.target_edge)
            else:
                settings = replace(settings, method="nearest")
                plan = plan_settings(source.image.size, settings)
            path = Path(name).resolve(strict=True)
            if path.stat().st_size > MAX_OUTPUT_FILE_BYTES:
                raise EnhancerError("Edited PNG exceeds the import limit.")
            with path.open("rb") as handle:
                data = handle.read(MAX_OUTPUT_FILE_BYTES + 1)
            if len(data) > MAX_OUTPUT_FILE_BYTES:
                raise EnhancerError("Edited PNG exceeds the import limit.")
            info = png_codec.inspect(data, output=True)
            if (info.width, info.height) != plan.target_size:
                raise EnhancerError(f"The PNG must be the finished {plan.target_size[0]} x {plan.target_size[1]} image, not the padded API canvas.")
            with open_output_image(data, plan.target_size) as opened:
                opened.load()
                image = to_srgb(opened, Profile("PNG", opened.mode, opened.size, info, opened.info.get("icc_profile", b"")))
            if settings.task == "translate":
                image = composite_translation(image, source.image, settings)
            elif settings.annotation and settings.annotation.area:
                image = finish_annotated_area(image, source.image, plan, settings.annotation.area, None)
            elif settings.preserve_alpha:
                image.putalpha(source_coverage(source.image.getchannel("A"), plan.target_size,
                                               detail_alpha and not source.profile.data_texture))
            elif source.image.getchannel("A").getextrema() == (255, 255) and image.getchannel("A").getextrema()[0] < 255:
                raise EnhancerError("The original is opaque. Enable Keep original alpha before importing a transparent edit.")
            image.info.clear()
            if not self._may_discard():
                return
            self.result = Result(image, source.path, settings, plan, None, None, "", {}, (), source.profile)
            self._saved_view = None
            self.save_button.configure(state="normal")
            self.png_button.configure(state="normal")
            self._refresh_previews()
            self.status.set("Edited PNG imported. Use Save Copy to write the selected output format.")
        except Exception as exc:
            messagebox.showerror("Cannot Import Edited PNG", redact(str(exc), self.key.get()), parent=self)

    def save_result(self) -> None:
        if self._busy or not self.result:
            return
        result = self.result
        profile = result.profile
        if profile is None and self.texture and self.texture.path == result.source_path:
            profile = self.texture.profile
        if profile is None:
            messagebox.showerror("Missing Format Profile", "Reopen and process the original file to retain its compatibility.", parent=self)
            return
        choice = FORMAT_CHOICES[self.output_format.get()]
        profile = output_profile(profile, choice)
        view = self._saved_view
        initial_directory = result.original_path.parent if result.original_path else self.recovery_directory
        name = filedialog.asksaveasfilename(parent=self, initialfile=output_name(result.source_path, choice),
                initialdir=str(initial_directory), defaultextension=profile.extension,
                filetypes=[(profile.format, "*.jpg *.jpeg" if profile.format == "JPEG" else "*" + profile.extension)])
        if not name:
            return
        destination = Path(name)
        overwrite = destination.exists()
        if overwrite and not messagebox.askyesno("Replace Output", "Replace this existing output file?\n\n" + str(destination), parent=self, default="no"):
            return
        # encoding can be expensive; do it off the Tk event loop, like the API job
        self._active_paid = False
        self._set_busy(True)
        self.status.set(f"Saving {profile.format} output…")
        def work():
            try:
                if view and EXTENSIONS.get(view.path.suffix.lower()) == profile.format == EXTENSIONS.get(destination.suffix.lower()):
                    saved = write_atomic(view.verified_bytes(), destination, result.source_path, overwrite=overwrite)
                else:
                    saved = save_original_atomic(result.image, profile, destination, result.source_path, overwrite=overwrite)
                self._queue.put((replace(result, settings=replace(result.settings, output_format=choice),
                                         saved_path=saved, original_path=saved), None))
            except Exception as exc:
                self._queue.put((None, redact(str(exc), key)))
        key = self.key.get()
        try:
            threading.Thread(target=work, daemon=True, name="native-export-worker").start()
        except Exception:
            self._set_busy(False)
            messagebox.showerror("Cannot Start Export", "The export worker could not be started.", parent=self)

    def export_png(self) -> None:
        if self._busy or not self.result:
            return
        result = self.result
        name = filedialog.asksaveasfilename(parent=self,
            initialfile=f"{result.source_path.stem}_{result.image.width}x{result.image.height}_enhanced.png",
            defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not name:
            return
        destination = Path(name)
        overwrite = destination.exists()
        if overwrite and not messagebox.askyesno("Replace Output", "Replace this existing PNG?", parent=self, default="no"):
            return
        self._active_paid = False
        self._set_busy(True)
        self.status.set("Saving the sRGB intermediate PNG...")
        key = self.key.get()
        def work():
            try:
                saved = save_image_atomic(result.image, destination, result.source_path, overwrite=overwrite)
                self._queue.put((replace(result, png_path=saved, saved_path=result.original_path or saved), None))
            except Exception as exc:
                self._queue.put((None, redact(str(exc), key)))
        try:
            threading.Thread(target=work, daemon=True, name="png-export-worker").start()
        except Exception:
            self._set_busy(False)
            messagebox.showerror("Cannot Start Export", "The PNG export worker could not be started.", parent=self)

    def open_output_folder(self) -> None:
        self._open_directory(self._last_output_directory)

    def _open_directory(self, directory: Path) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(directory))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except OSError as exc:
            messagebox.showerror("Output Folder", str(exc), parent=self)

    def on_close(self) -> None:
        if self._busy:
            warning = ("Closing will stop waiting but cannot cancel work already received by OpenAI. "
                       "You may be charged and lose the result. Close anyway?" if self._active_paid else
                       "Local processing or export is still running. Closing may discard unfinished output. Close anyway?")
            if not messagebox.askyesno("Processing Still In Progress", warning, parent=self, default="no"):
                return
        if not self._may_discard():
            return
        self.destroy()

    def destroy(self) -> None:
        self._closed = True
        for timer in (self._poll_id, self._preview_id, self._startup_id):
            if timer:
                try:
                    self.after_cancel(timer)
                except tk.TclError:
                    pass
        self.key.set("")
        self._batch_stop.set()
        self._batch_key = ""
        super().destroy()


def create_app() -> App:
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    return App()


def main() -> None:
    create_app().mainloop()
