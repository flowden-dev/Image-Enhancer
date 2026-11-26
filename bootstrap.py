"""Create an isolated environment; later launches need no package download."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import venv

ROOT = Path(__file__).resolve().parent


def launch_windowed(executable: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="image-enhancer-start-") as directory:
        ready = Path(directory) / "ready"
        child = subprocess.Popen(
            [str(executable), str(ROOT / "app.py"), "--startup-dir", directory],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            while True:
                status = child.poll()
                if ready.is_dir():
                    return status or 0
                if status is not None:
                    print(f"Image Enhancer exited before its window opened (exit code {status}).")
                    error = Path(directory) / "error.txt"
                    if error.is_file():
                        print(error.read_text(encoding="utf-8").rstrip())
                    return status or 1
                time.sleep(0.05)
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise


def launch_app(executable: Path) -> int:
    if os.name == "nt":
        return launch_windowed(executable.with_name("pythonw.exe"))
    return subprocess.run([str(executable), str(ROOT / "app.py")], cwd=ROOT).returncode


def main() -> int:
    if not (3, 11) <= sys.version_info < (3, 15) or struct.calcsize("P") != 8:
        print("Install 64-bit Python 3.11 through 3.14 (3.13 recommended), including Tcl/Tk, from python.org.")
        return 1
    if any(argument not in ("--test",) for argument in sys.argv[1:]):
        print("Supported option: --test")
        return 1
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("Tkinter is missing.")
        if sys.platform == "darwin":
            print("Install the macOS Python installer from python.org, which includes Tcl/Tk.")
        elif os.name == "nt":
            print("Modify your Python installation and enable Tcl/Tk and IDLE.")
        else:
            print("Install your distribution's Tk and venv packages, e.g. python3-tk and python3-venv on Debian/Ubuntu.")
        return 1
    env = ROOT / ".venv"
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if not executable.exists():
            print("Creating a private Python environment in .venv...", flush=True)
            venv.EnvBuilder(with_pip=True).create(env)
        pins = {}
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.partition("#")[0].strip()
            if not line:
                continue
            name, separator, version = line.partition("==")
            if not separator or not name or not version or any(c.isspace() for c in line):
                raise RuntimeError("requirements.txt must contain exact package==version pins.")
            pins[name] = version
        check = (
            "import sys, tkinter, PIL.Image, PIL.ImageTk, httpx, numpy, tkinterdnd2; "
            "from importlib.metadata import version; "
            f"pins = {pins!r}; "
            "sys.exit(0 if (3, 11) <= sys.version_info < (3, 15) and sys.maxsize > 2**32 "
            "and all(version(name) == expected for name, expected in pins.items()) else 1)"
        )
        ready = subprocess.run([str(executable), "-c", check], cwd=ROOT, capture_output=True).returncode == 0
        if not ready:
            print("Installing the tested dependencies. This step needs internet access...", flush=True)
            subprocess.run([str(executable), "-m", "pip", "install", "--disable-pip-version-check",
                            "--require-virtualenv", "--only-binary=:all:", "-r", str(ROOT / "requirements.txt")],
                           cwd=ROOT, check=True)
            subprocess.run([str(executable), "-c", check], cwd=ROOT, check=True)
        if "--test" in sys.argv[1:]:
            command = [str(executable), "-m", "unittest", "discover", "-s", "tests", "-v"]
            return subprocess.run(command, cwd=ROOT).returncode
        print("Starting Image Enhancer...", flush=True)
        return launch_app(executable)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ImportError) as exc:
        print(f"Setup or launch failed: {exc}")
        print("Extract the entire ZIP to a writable folder. Check internet access for the first setup.")
        print("If Python was moved or upgraded, close the app, delete only this tool's .venv folder and rerun.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
