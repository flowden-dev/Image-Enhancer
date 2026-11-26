# Contributing

Use 64-bit Python 3.11–3.14 with Tkinter. Create a virtual environment, then install the application and development tools from the repository root.

```sh
python -m pip install -e . ruff build
python -m ruff check .
python scripts/check.py --require-gui
python -m build
```

On headless Linux, run the GUI suite with `xvfb-run -a python scripts/check.py --require-gui`. Tests use synthetic image fixtures and mocked HTTP responses.

## Pull requests

- Describe the user-visible change and how you tested it. Include a small regression test for a bug fix.
- Keep Tk calls on the main thread and image processing in workers.
- Verify output dimensions, source protection, alpha handling and recovery when changing the image pipeline.
- Report unsupported encodings before any paid request. Keep retries and model changes explicit.
- Use redistributable sample images. Keep credentials, private images and local logs out of submissions.

For sensitive reports, follow the [security policy](SECURITY.md).

[Implementation notes](docs/architecture.md) · [Test coverage](docs/validation.md)
