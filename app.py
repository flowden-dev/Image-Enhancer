"""Run the desktop application from a prepared Python environment."""
from pathlib import Path
import sys
import traceback


def main(startup_dir: Path | None = None) -> None:
    try:
        from enhancer.gui import create_app

        app = create_app()
        if startup_dir is not None:
            app.update_idletasks()
            if not app.winfo_viewable():
                app.wait_visibility()
            # publish readiness without holding a file handle during Windows cleanup
            (startup_dir / "ready").mkdir()
    except Exception:
        if startup_dir is not None:
            (startup_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    app.mainloop()


if __name__ == "__main__":
    directory = Path(sys.argv[2]) if len(sys.argv) == 3 and sys.argv[1] == "--startup-dir" else None
    main(directory)
