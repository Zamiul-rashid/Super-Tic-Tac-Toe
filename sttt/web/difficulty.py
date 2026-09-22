"""Difficulty tiers, loaded from versioned configuration.

Kept in ``configs/web/difficulty.json`` rather than inline so changing an
opponent's strength is a config edit, not a code edit.
"""
import json
import os
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "web" / "difficulty.json"


class Tiers:
    def __init__(self, path=None):
        path = Path(path or os.environ.get("STTT_WEB_TIERS", DEFAULT_CONFIG))
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported difficulty schema in {path}: "
                             f"{data.get('schema_version')!r}")
        self.tiers = data["tiers"]
        self.default = data["default"]
        if self.default not in self.tiers:
            raise ValueError(f"default tier {self.default!r} is not defined in {path}")

    def __contains__(self, key):
        return key in self.tiers

    def __getitem__(self, key):
        return self.tiers[key]

    def public(self):
        """What the client needs to render the selector."""
        return [{"key": key, **value} for key, value in self.tiers.items()]
