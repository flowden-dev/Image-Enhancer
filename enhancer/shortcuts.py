"""Keyboard commands for the main window and its image editors."""
import tkinter as tk
from tkinter import ttk

from .theme import center_dialog


def primary_modifier(window):
    return "Command" if window.tk.call("tk", "windowingsystem") == "aqua" else "Control"


def primary_label(window):
    return "Cmd" if primary_modifier(window) == "Command" else "Ctrl"


def _prioritize_shortcuts(window):
    # Run the window's bindings after widget-specific bindings but before Tk
    # inserts a newline or changes notebook tabs. Dialogs have their own tag.
    tag = str(window)
    widgets = list(window.winfo_children())
    while widgets:
        widget = widgets.pop()
        if isinstance(widget, tk.Toplevel):
            continue
        tags = list(widget.bindtags())
        if tag in tags:
            tags.remove(tag)
            tags.insert(1, tag)
            widget.bindtags(tags)
        widgets.extend(widget.winfo_children())


def bind_editor_keys(window, accept, *, undo=None):
    primary = primary_modifier(window)

    def invoke(command):
        def handle(_event):
            command()
            return "break"
        return handle

    _prioritize_shortcuts(window)
    commands = {"<Escape>": window.destroy, f"<{primary}-Return>": accept}
    if primary == "Command":
        commands["<Control-Return>"] = accept
    if undo is not None:
        commands[f"<{primary}-z>"] = undo
    for sequence, command in commands.items():
        window.bind(sequence, invoke(command))


def install_workspace_keys(app):
    _prioritize_shortcuts(app)
    primary = primary_modifier(app)
    aqua = primary == "Command"

    def has_grab():
        return bool(app.tk.call("grab", "current", app))

    def editing(event):
        return isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, tk.Listbox, tk.Spinbox))

    def navigate(event):
        if app.tabs.select() != str(app.preview_tab) or editing(event):
            return
        if isinstance(event.widget, (ttk.Radiobutton, ttk.Scale, ttk.Scrollbar)):
            return
        app._step_preview(-1 if event.keysym.lower() == "left" else 1)
        return "break"

    def review(event):
        if has_grab():
            return
        key = event.keysym.lower()
        if key == "f1":
            show_shortcuts(app)
            return "break"
        if key in ("left", "right"):
            return navigate(event)
        if app._busy or (editing(event) and not (event.widget is app.file_search and key == "f2")):
            return
        if key in ("a", "f2"):
            app.annotate_selected()
        elif app.tabs.select() == str(app.preview_tab):
            if key in ("0", "1"):
                app.set_preview_zoom(fit=key == "0")
            elif key == "escape":
                app.show_files()
            else:
                return
        elif event.widget is app.batch_tree and key in ("return", "kp_enter"):
            app.preview_selected()
        else:
            return
        return "break"

    def accelerator(event, command):
        if has_grab():
            return
        key = event.keysym.lower()
        if command == "navigate" and key in ("left", "right"):
            return navigate(event)
        if app._busy:
            return
        if command == "main" and key == "o":
            app.add_files()
        elif command == "folder" and key == "o":
            app.add_folder()
        elif key == "f" and (command == "main" or command == "find" and not editing(event)):
            app._focus_file_search()
        elif command == "main" and key == "s" and app.tabs.select() == str(app.preview_tab):
            app.save_result()
        else:
            return
        return "break"

    app.bind("<KeyPress>", review)
    # Tk selects the most specific modifier pattern. Let all other chords
    # continue to native controls, including Shift+arrows and AltGr typing.
    other_modifiers = ["Shift", "Meta", "Control-Alt", "Shift-Alt"]
    if aqua:
        other_modifiers.extend(("Command-Option", "Shift-Option", "Control-Command", "Control-Option"))
    if app.tk.call("tk", "windowingsystem") == "x11":
        other_modifiers.extend(("Mod4", "Mod5", "Control-Mod5"))
    for mods in other_modifiers:
        app.bind(f"<{mods}-KeyPress>", "# native shortcut")
    chords = {primary: "main", f"{primary}-Shift": "folder", "Alt": "navigate"}
    if aqua:
        chords.update({"Option": "navigate", "Control": "find"})
    for mods, command in chords.items():
        app.bind(f"<{mods}-KeyPress>", lambda event, c=command: accelerator(event, c))


def show_shortcuts(parent):
    previous_focus = parent.focus_get()
    dialog = tk.Toplevel(parent)
    dialog.title("Keyboard shortcuts")
    dialog.transient(parent)
    body = ttk.Frame(dialog, padding=16)
    body.pack(fill="both", expand=True)
    primary = primary_label(parent)
    rows = (
        ("A or F2", "Annotate the current file in Upscale mode"),
        ("Up / Down   ·   Enter", "Select a file, then open its preview"),
        ("Left / Right   ·   Esc", "Browse previews, then return to Files"),
        ("0 / 1", "Fit both previews / view at 100%"),
        (f"{primary}+Enter   ·   Esc", "Save editor changes / cancel"),
        (f"{primary}+Z", "Undo the last box in Mark Areas"),
        (f"{primary}+O / {primary}+Shift+O", "Add images / add a folder"),
        (f"{primary}+F", "Find a file. Enter selects the next match"),
        (f"{primary}+S", "Save a copy of the previewed result"),
        ("F1", "Show keyboard shortcuts"),
    )
    for row, (keys, action) in enumerate(rows):
        ttk.Label(body, text=keys).grid(row=row, column=0, sticky="w", padx=(0, 18), pady=3)
        ttk.Label(body, text=action).grid(row=row, column=1, sticky="w", pady=3)
    def dismiss(_event=None):
        if previous_focus is not None and previous_focus.winfo_exists():
            previous_focus.focus_set()
        dialog.grab_release()
        dialog.destroy()
        return "break"

    close = ttk.Button(body, text="Close", command=dismiss)
    close.grid(row=len(rows), column=1, sticky="e", pady=(12, 0))
    dialog.bind("<Escape>", dismiss)
    dialog.protocol("WM_DELETE_WINDOW", dismiss)
    center_dialog(dialog, parent)
    dialog.grab_set()
    close.focus_set()
