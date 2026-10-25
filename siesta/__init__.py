__version__ = "0.1.0"


def _load_env():
    """Tiny .env reader (KEY=value lines). Real environment variables win."""
    import os
    from pathlib import Path
    for d in (Path.cwd(), Path(__file__).resolve().parent.parent):
        f = d / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and not k.startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return


_load_env()
