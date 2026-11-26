# Implementation notes

| Code | Responsibility |
| --- | --- |
| `gui.py`, `preview.py`, `*_gui.py` | Desktop controls, viewport rendering and area editors. Workers send results to the Tk event loop through queues. |
| `core.py`, `super_resolution.py`, `fidelity.py` | Output dimensions, padded model canvases, tiled local reconstruction and source protection. |
| `annotations.py`, `translation.py` | Exact wording, source-coordinate masks and local compositing. |
| `api.py`, `batch.py` | Preflight, confirmed model requests, sequential work, duplicate reuse and recovery. |
| `formats/`, `storage.py` | Validated decoding, target encoding, color management and atomic file publication. |

Batch preflight checks source semantics and exercises the chosen encoder before a paid edit. Work is grouped by dimensions, decoded RGBA, channel semantics and per-file instructions. Matching source bytes can share a verified encoded output. Different encodings share the working pixels and are exported separately. There is at most one request per new model result and no automatic paid retry.

The queue retains paths and job settings for completed files. Preview reads the selected export on a worker and applies output limits so a large generated texture can be inspected. The queue does not hold decoded copies of every image.

A boxed annotation combines the returned label with a local neural reconstruction outside the rectangle. Translation performs its own masked compositing. The final dimensions are independent of the API canvas. The local Anime4K weights and their [license](../enhancer/models/Anime4K-LICENSE.txt) ship with the package.

Preferences, optionally remembered annotations and recovery files use the per-user app data directory. An absolute `IMAGE_ENHANCER_DATA_DIR` overrides it for isolated tests. Job records omit keys, prompts and exact wording.
