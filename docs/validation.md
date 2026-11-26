# Validation

See [Contributing](../CONTRIBUTING.md) for the test and build commands. The desktop tests require Tk and a display. On headless Linux, run `xvfb-run -a python scripts/check.py --require-gui`.

The offline suite covers source decoding, format conversion, encoded dimensions, transparency, JPEG settings, DDS mipmaps, folder reuse, annotations and browsing saved outputs. Mock HTTP responses exercise the OpenAI path. No API key is required.

For larger format fixtures, run `python tests/stress_native.py`. The GitHub Actions workflow builds and tests the project on Windows, Linux and macOS, then checks the installed wheel entry point.
