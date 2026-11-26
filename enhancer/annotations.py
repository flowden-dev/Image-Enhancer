"""Per-file artwork instructions and optional local storage for remembered annotations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from .errors import EnhancerError
from .super_resolution import upscale_artwork
from .storage import write_atomic
from .translation import map_box, validate_text

if TYPE_CHECKING:
    from .core import Plan

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageAnnotation:
    source_sha256: str
    exact_text: str = ""
    note: str = ""
    area: Box | None = None

    def validate(self):
        validate_text(self.exact_text, "Exact text", 512)
        validate_text(self.note, "Visual note", 2000)
        if not self.exact_text.strip() and not self.note.strip():
            raise EnhancerError("Enter the exact text or a visual note, or remove the annotation.")
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.source_sha256):
            raise EnhancerError("Reopen this image to bind the annotation to its source.")
        if self.area is not None and (not isinstance(self.area, tuple) or len(self.area) != 4
                or any(type(v) is not int for v in self.area)
                or not (0 <= self.area[0] < self.area[2] and 0 <= self.area[1] < self.area[3])):
            raise EnhancerError("Draw a nonempty rectangle inside the image.")

    def check_source(self, digest):
        if digest != self.source_sha256:
            raise EnhancerError("The image changed since it was annotated. Open Annotate and review the current image.")

    def check_bounds(self, size):
        if self.area and (self.area[2] > size[0] or self.area[3] > size[1]):
            raise EnhancerError("The annotation area is outside the current image. Draw it again.")

    def canvas_area(self, plan: Plan) -> Box | None:
        if self.area is None:
            return None
        x, y, right, bottom = plan.crop
        return map_box(self.area, plan.source_size, (right-x, bottom-y), (x, y))

    def reuse_key(self, pixel_hash: str) -> str:
        # Source encodings can differ. The decoded artwork, exact text, note and area must agree.
        data = json.dumps([pixel_hash, self.exact_text, self.note, self.area], ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def prompt(self, plan: Plan) -> str:
        data = {key: value for key, value in (("exact_text", self.exact_text), ("visual_note", self.note)) if value}
        prompt = (
            "Upscale this artwork. Use the following annotation for this image only. "
            "The exact_text value, when present, is the intended lettering even if the blurry source "
            "appears to spell something else. Render it verbatim, including case, accents, umlauts, "
            "punctuation and line breaks. Treat it as label content. Do not translate it or add a caption. "
            "Replace the ambiguous strokes and keep the original lettering style, weight, color and position. "
            "The visual_note provides restoration guidance and must not appear as printed text. "
            "Preserve the rest of the design, including other labels, icons, borders, shading and transparency. "
            f"Return the full {plan.size_string} canvas. Artwork bounds are {plan.crop} in canvas pixels. "
        )
        area = self.canvas_area(plan)
        if area:
            prompt += (f"Edit only the marked rectangle {area}, in canvas pixels with exclusive right and bottom "
                       "edges. Rebuild the background behind removed strokes and fit every accent inside the rectangle. ")
        return prompt + json.dumps(data, ensure_ascii=False)


def finish_annotated_area(raw, source, plan, area: Box, guide):
    """Keep the local reconstruction outside the requested label, including its alpha."""
    if raw.size != plan.canvas_size or source.size != plan.source_size:
        raise EnhancerError("The annotation response dimensions do not match the request.")
    result = guide if guide is not None and guide.size == plan.target_size else upscale_artwork(source, plan.target_size)
    box = map_box(area, source.size, plan.target_size)
    x, y, right, bottom = plan.crop
    sx, sy = (right - x) / result.width, (bottom - y) / result.height
    region = (x + box[0] * sx, y + box[1] * sy,
              min(right, x + box[2] * sx), min(bottom, y + box[3] * sy))
    patch = raw.convert("RGBA").resize((box[2]-box[0], box[3]-box[1]), Image.Resampling.LANCZOS, region)
    background = result.crop(box)
    if background.getchannel("A").getextrema() == (255, 255):
        patch = Image.alpha_composite(background, patch)
    # On transparent lettering the returned alpha can form entirely new strokes and accents.
    result.paste(patch, box)
    result.info.clear()
    return result


class AnnotationStore:
    """One atomic record per opted-in source path. Reading a queue never creates files."""
    def __init__(self, directory: Path):
        self.directory = directory

    def _record(self, source: Path) -> Path:
        key = os.fsencode(os.path.normcase(os.path.abspath(source)))
        return self.directory / (hashlib.sha256(key).hexdigest() + ".json")

    def load(self, paths):
        found = {}
        warnings = []
        # Enumerate the handful of stored annotations once, avoiding a disk probe per queue row.
        try:
            saved = {p.name for p in self.directory.iterdir() if p.suffix == ".json"}
        except FileNotFoundError:
            return found, warnings
        except OSError:
            return found, ["Could not read the remembered annotations. Review marked files before running."]
        if not saved:
            return found, warnings
        for path in paths:
            record = self._record(path)
            if record.name not in saved:
                continue
            try:
                with record.open("rb") as stream:
                    data = stream.read(32769)
                if len(data) > 32768:
                    raise ValueError("oversized record")
                fields = json.loads(data)
                area = fields["area"]
                note = ImageAnnotation(fields["source_sha256"], fields["exact_text"], fields["note"],
                                       None if area is None else tuple(area))
                note.validate()
                found[path] = note
            except (OSError, ValueError, TypeError, KeyError, EnhancerError, RecursionError):
                warnings.append(f"Could not load the saved annotation for {path.name}.")
        return found, warnings

    def save(self, source: Path, annotation: ImageAnnotation | None):
        record = self._record(source)
        if annotation is None:
            record.unlink(missing_ok=True)
            return
        annotation.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"source_sha256": annotation.source_sha256, "exact_text": annotation.exact_text,
                           "note": annotation.note, "area": annotation.area}, ensure_ascii=False)
        write_atomic(data.encode("utf-8"), record, overwrite=True)
