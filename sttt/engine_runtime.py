"""Resolve optional, project-local engine dependencies without /tmp paths."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def activate_runtime():
    runtime = ROOT / 'engines' / 'runtime'
    if runtime.is_dir() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))


def activate_utttai():
    activate_runtime()
    vendor = ROOT / 'engines' / 'vendor' / 'utttai'
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
