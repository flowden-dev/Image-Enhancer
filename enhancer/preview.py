"""Viewport-sized rendering, pointer-anchored zoom and right-button panning."""
from __future__ import annotations

import math
import sys
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from .theme import VIEW, MUTED


class ImagePreview(tk.Canvas):
    def __init__(self, master, *, centered=True, empty_text="Drop an image to begin", **kwargs):
        super().__init__(master, background=VIEW, highlightthickness=0, **kwargs)
        self.source = None
        self.empty_text = empty_text
        self.scale = 1.0
        self.fit_mode = True
        self.centered = centered
        self.offset = (0.0, 0.0)
        self.photo = None
        self.overlays = ()
        self._render_id = None
        self._panning = False
        self.on_view_change = None
        self.bind("<Configure>", self._resize)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", self._wheel)
        self.bind("<Button-5>", self._wheel)
        self.bind("<ButtonPress-3>", self._pan_start)
        self.bind("<B3-Motion>", self._pan_move)
        self.bind("<ButtonRelease-3>", self._pan_end)
        self.bind("<ButtonPress-2>", self._pan_start)
        self.bind("<B2-Motion>", self._pan_move)
        self.bind("<ButtonRelease-2>", self._pan_end)
        if sys.platform == "darwin":
            self.bind("<Control-ButtonPress-1>", self._pan_start)
            self.bind("<Control-B1-Motion>", self._pan_move)
            self.bind("<Control-ButtonRelease-1>", self._pan_end)
        self.bind("<Double-Button-1>", lambda _: self.fit())

    def set_image(self, image):
        if image is not self.source:
            self.source = image
            self.fit_mode = True
            self.xview_moveto(0)
            self.yview_moveto(0)
        self.render()

    def _resize(self, _event=None):
        self.schedule_render()

    def schedule_render(self):
        if self._render_id is None:
            self._render_id = self.after(16, self.render)

    def fit(self):
        self.fit_mode = True
        self.xview_moveto(0)
        self.yview_moveto(0)
        self.render()

    def set_zoom(self, scale):
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            self.fit_mode = False
            self.scale = min(32.0, max(0.01, float(scale)))
            self._geometry()
            self.xview_moveto(0)
            self.yview_moveto(0)
            self.render()
            return
        self.zoom_at(scale, self.winfo_width()/2, self.winfo_height()/2)

    def _geometry(self):
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self.source is None:
            return width, height
        if self.fit_mode:
            self.scale = min(width/self.source.width, height/self.source.height)
        iw, ih = self.source.width*self.scale, self.source.height*self.scale
        self.offset = (max(0.0, (width-iw)/2), max(0.0, (height-ih)/2)) if self.centered else (0.0, 0.0)
        self.configure(scrollregion=(0, 0, max(width, math.ceil(iw)), max(height, math.ceil(ih))))
        return width, height

    def image_point(self, x, y, *, clamp=False):
        if self.source is None:
            return None
        px, py = (self.canvasx(x)-self.offset[0])/self.scale, (self.canvasy(y)-self.offset[1])/self.scale
        if clamp:
            return min(self.source.width, max(0.0, px)), min(self.source.height, max(0.0, py))
        if not (0 <= px < self.source.width and 0 <= py < self.source.height):
            return None
        return px, py

    def screen_box(self, box):
        ox, oy = self.offset
        return box[0]*self.scale+ox, box[1]*self.scale+oy, box[2]*self.scale+ox, box[3]*self.scale+oy

    def zoom_at(self, scale, x, y):
        if self.source is None:
            return
        point = self.image_point(x, y, clamp=True)
        self.fit_mode = False
        self.scale = min(32.0, max(0.01, float(scale)))
        width, height = self._geometry()
        iw, ih = max(width, math.ceil(self.source.width*self.scale)), max(height, math.ceil(self.source.height*self.scale))
        self.xview_moveto((point[0]*self.scale+self.offset[0]-x)/iw)
        self.yview_moveto((point[1]*self.scale+self.offset[1]-y)/ih)
        self.render()

    def _wheel(self, event):
        if getattr(event, "num", None) in (4, 5):
            steps = 1 if event.num == 4 else -1
        else:
            delta = getattr(event, "delta", 0)
            steps = delta / (120 if abs(delta) >= 120 else (10 if sys.platform == "darwin" else 1))
            steps = min(3, max(-3, steps))
        if steps:
            self.zoom_at(self.scale * 1.2**steps, event.x, event.y)
        return "break"

    def _pan_start(self, event):
        self._panning = True
        self.scan_mark(event.x, event.y)
        self.configure(cursor="fleur")
        return "break"

    def _pan_move(self, event):
        if self._panning:
            self.scan_dragto(event.x, event.y, gain=1)
            self.schedule_render()
        return "break"

    def _pan_end(self, _event):
        self._panning = False
        self.configure(cursor="")
        return "break"

    def scroll_x(self, *args):
        self.xview(*args)
        self.schedule_render()

    def scroll_y(self, *args):
        self.yview(*args)
        self.schedule_render()

    def render(self):
        if self._render_id is not None:
            self.after_cancel(self._render_id)
            self._render_id = None
        width, height = self._geometry()
        self.delete("all")
        if self.source is None:
            self.photo = None
            self.create_text(width/2, height/2, text=self.empty_text, fill=MUTED)
            return
        # only resize the visible source crop; a 32x view never allocates a giant bitmap
        vx, vy = self.canvasx(0), self.canvasy(0)
        ox, oy = self.offset
        x0, y0 = max(0, math.floor((vx-ox)/self.scale)-2), max(0, math.floor((vy-oy)/self.scale)-2)
        x1, y1 = min(self.source.width, math.ceil((vx+width-ox)/self.scale)+2), min(self.source.height, math.ceil((vy+height-oy)/self.scale)+2)
        if x1 > x0 and y1 > y0:
            size = max(1, round((x1-x0)*self.scale)), max(1, round((y1-y0)*self.scale))
            region = self.source if (x0, y0, x1, y1) == (0, 0, *self.source.size) else self.source.crop((x0, y0, x1, y1))
            thumb = region.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
            chess = Image.new("RGB", size, "#252d3b")
            draw = ImageDraw.Draw(chess)
            for y in range(0, size[1], 20):
                for x in range(0, size[0], 20):
                    if (x//20+y//20) % 2:
                        draw.rectangle((x, y, x+19, y+19), fill="#303b4c")
            chess.paste(thumb, (0, 0), thumb)
            self.photo = ImageTk.PhotoImage(chess, master=self)
            self.create_image(ox+x0*self.scale, oy+y0*self.scale, image=self.photo, anchor="nw")
        for box, color, label in self.overlays:
            coords = self.screen_box(box)
            self.create_rectangle(*coords, outline=color, width=2, tags="text_area")
            self.create_text(coords[0]+4, coords[1]+4, text=label, fill=color, anchor="nw", tags="text_area")
        if self.on_view_change:
            self.on_view_change()

    def destroy(self):
        if self._render_id is not None:
            self.after_cancel(self._render_id)
            self._render_id = None
        self.photo = None
        self.source = None
        super().destroy()
