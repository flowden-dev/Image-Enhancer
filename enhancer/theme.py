"""Compact dark controls shared by the workspace and editing dialogs."""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font, ttk

BG = "#111318"
PANEL = "#1a1d24"
FIELD = "#242831"
BORDER = "#30353f"
TEXT = "#e6e9ef"
MUTED = "#969eae"
ACCENT = "#a293ee"
PROTECTED = "#60c9aa"
VIEW = "#101217"


def center_dialog(window, parent) -> None:
    window.update_idletasks()
    width, height = window.winfo_width(), window.winfo_height()
    x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    x = max(0, min(x, window.winfo_screenwidth() - width))
    y = max(0, min(y, window.winfo_screenheight() - height))
    window.geometry(f"+{x}+{y}")


def apply_theme(root: tk.Tk) -> None:
    preferred = "Segoe UI" if sys.platform == "win32" else "Helvetica Neue" if sys.platform == "darwin" else "DejaVu Sans"
    available = {name.casefold(): name for name in font.families(root)}
    family = next((available[name.casefold()] for name in (preferred, "Helvetica", "Nimbus Sans L", "Arial")
                   if name.casefold() in available), font.nametofont("TkDefaultFont").actual("family"))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        font.nametofont(name).configure(family=family, size=10)
    root.configure(background=BG)
    root.option_add("*background", BG)
    root.option_add("*foreground", TEXT)
    root.option_add("*Text.background", FIELD)
    root.option_add("*Text.foreground", TEXT)
    root.option_add("*Text.font", "TkTextFont")
    root.option_add("*Text.insertBackground", TEXT)
    root.option_add("*Text.selectBackground", "#49425f")
    root.option_add("*Text.relief", "flat")
    root.option_add("*Text.borderWidth", 0)
    root.option_add("*Text.highlightThickness", 1)
    root.option_add("*Text.highlightBackground", BORDER)
    root.option_add("*Text.highlightColor", ACCENT)
    root.option_add("*Text.padX", 10)
    root.option_add("*Text.padY", 8)
    root.option_add("*Listbox.background", FIELD)
    root.option_add("*Listbox.foreground", TEXT)
    root.option_add("*Listbox.selectBackground", "#49425f")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=TEXT, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, troughcolor=BG,
                    focuscolor=ACCENT, font=(family, 10))
    style.configure("TFrame", background=BG)
    style.configure("TSeparator", background=BORDER)
    style.configure("Card.TFrame", background=PANEL)
    style.configure("Preview.TFrame", background=BORDER)
    style.configure("Card.TLabel", background=PANEL)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Small.TLabel", foreground=MUTED, font=(family, 9))
    style.configure("CardMuted.TLabel", foreground=MUTED, background=PANEL, font=(family, 9))
    style.configure("CardSmall.TLabel", foreground=MUTED, background=PANEL, font=(family, 9))
    style.configure("CardHeading.TLabel", background=PANEL, font=(family, 10, "bold"))
    style.configure("Title.TLabel", font=(family, 15, "bold"))
    style.configure("Section.TLabel", font=(family, 11, "bold"))
    style.configure("TButton", background=FIELD, foreground=TEXT, padding=(10, 4), width=0,
                    relief="flat", borderwidth=1, lightcolor=FIELD, darkcolor=FIELD, font=(family, 9))
    style.map("TButton", background=[("disabled", PANEL), ("pressed", "#383e4a"), ("active", "#303641")],
              foreground=[("disabled", "#626b7c")], bordercolor=[("focus", ACCENT)])
    style.configure("Quiet.TButton", background=BG, bordercolor=BG, lightcolor=BG, darkcolor=BG, padding=(8, 5))
    style.map("Quiet.TButton", bordercolor=[("focus", ACCENT), ("active", FIELD)])
    style.configure("Accent.TButton", background=ACCENT, foreground="#191625", bordercolor=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT, padding=(12, 4), font=(family, 10, "bold"))
    style.map("Accent.TButton", background=[("disabled", "#3f3b50"), ("pressed", "#8d7bdf"), ("active", "#b4a7f5")],
              bordercolor=[("disabled", "#3f3b50"), ("focus", TEXT)], foreground=[("disabled", "#827d92")])
    style.layout("Segment.TRadiobutton", style.layout("TButton"))
    style.configure("Segment.TRadiobutton", background=FIELD, foreground=MUTED, padding=(12, 6),
                    anchor="center", relief="flat", borderwidth=1, bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD,
                    font=(family, 9))
    style.map("Segment.TRadiobutton", background=[("selected", "#373148"), ("active", "#2e323d")],
              foreground=[("disabled", "#626b7c"), ("selected", "#c4b8ff")],
              bordercolor=[("focus", ACCENT), ("selected", "#504463")])
    style.map("Protect.Segment.TRadiobutton", background=[("selected", "#263d37"), ("active", "#2e323d")],
              foreground=[("selected", PROTECTED)], bordercolor=[("focus", PROTECTED), ("selected", "#38564b")])
    style.configure("TEntry", fieldbackground=FIELD, insertcolor=TEXT, padding=5, borderwidth=1,
                    lightcolor=BORDER, darkcolor=BORDER)
    style.map("TEntry", bordercolor=[("focus", ACCENT)])
    style.configure("TCombobox", fieldbackground=FIELD, background=FIELD, arrowcolor=MUTED, padding=(6, 4),
                    arrowsize=11, borderwidth=1, lightcolor=BORDER, darkcolor=BORDER)
    style.map("TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
              foreground=[("readonly", TEXT), ("disabled", "#626b7c")], bordercolor=[("focus", ACCENT)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
    style.configure("TCheckbutton", indicatorbackground=FIELD, padding=(0, 3), indicatorsize=12, font=(family, 9))
    style.map("TCheckbutton", background=[("active", BG)], indicatorbackground=[("selected", ACCENT)],
              foreground=[("disabled", "#626b7c")])
    style.configure("Card.TCheckbutton", background=PANEL)
    style.map("Card.TCheckbutton", background=[("active", PANEL)])
    style.configure("TRadiobutton", indicatorbackground=FIELD, padding=3)
    style.map("TRadiobutton", background=[("active", BG)], indicatorbackground=[("selected", ACCENT)])
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=BG, bordercolor=BG, lightcolor=BG, darkcolor=BG,
                    padding=(16, 7), font=(family, 9))
    style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEXT), ("!selected", MUTED)],
              bordercolor=[("selected", PANEL)], lightcolor=[("selected", PANEL)], darkcolor=[("selected", PANEL)])
    style.configure("Treeview", background=VIEW, fieldbackground=VIEW, foreground=TEXT,
                    rowheight=27, borderwidth=0, font=(family, 9))
    style.configure("Treeview.Heading", background=FIELD, foreground=MUTED, padding=(8, 6), relief="flat", font=(family, 9))
    style.map("Treeview", background=[("selected", "#3a344c")], foreground=[("selected", TEXT)])
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD, borderwidth=0, thickness=3)
    for direction in ("Vertical", "Horizontal"):
        style.configure(f"{direction}.TScrollbar", background=BORDER, troughcolor=PANEL, arrowcolor=MUTED,
                        lightcolor=BORDER, darkcolor=BORDER, bordercolor=PANEL, borderwidth=0, arrowsize=10)
        style.map(f"{direction}.TScrollbar", background=[("active", "#48505e")])
