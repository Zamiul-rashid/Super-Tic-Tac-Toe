"""Named external-engine configurations for reproducible tournaments."""
import json
import sys
from pathlib import Path

from .bots import ExternalProcessBot


def load_engine_registry(path: str | Path) -> dict[str, dict]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Engine registry not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid engine registry JSON: {path}: {exc}") from exc
    engines = data.get("engines") if isinstance(data, dict) else None
    if not isinstance(engines, dict):
        raise ValueError("Engine registry must contain an 'engines' object")
    return {str(name): _validate(name, spec, path.parent) for name, spec in engines.items()}


def _validate(name: str, spec: dict, base: Path) -> dict:
    if not isinstance(spec, dict):
        raise ValueError(f"Engine '{name}' must be an object")
    command = spec.get("command")
    if not command or not isinstance(command, (str, list)):
        raise ValueError(f"Engine '{name}' needs a non-empty string or argv list command")
    if isinstance(command, list) and not all(isinstance(part, str) and part for part in command):
        raise ValueError(f"Engine '{name}' command argv must contain non-empty strings")
    protocol = spec.get("protocol", "codingame")
    if protocol not in ("codingame", "action", "action_index", "state_json"):
        raise ValueError(f"Engine '{name}' has unsupported protocol '{protocol}'")
    timeout = float(spec.get("timeout", 5.0))
    if not 0 < timeout <= 60:
        raise ValueError(f"Engine '{name}' timeout must be in (0, 60] seconds")
    cwd = spec.get("cwd")
    if cwd and not Path(cwd).is_absolute():
        cwd = str((base / cwd).resolve())
    result = dict(spec, protocol=protocol, timeout=timeout, cwd=cwd)
    import shlex
    parts = shlex.split(command) if isinstance(command, str) else command
    result['command'] = [part.replace('{python}', sys.executable) for part in parts]
    result["name"] = name
    return result


def create_configured_engine(name: str, registry: dict[str, dict]) -> ExternalProcessBot | None:
    """Create a named external engine, or return None for built-in bot names."""
    spec = registry.get(name)
    if spec is None:
        return None
    import shlex
    command = shlex.split(spec['command']) if isinstance(spec['command'], str) else spec['command']
    command = [part.replace('{simulations}', str(spec.get('simulations', 128)))
               .replace('{seed}', str(spec.get('seed', 0))) for part in command]
    return ExternalProcessBot(
        command=command, timeout=spec["timeout"], protocol=spec["protocol"],
        fallback=spec.get("fallback", "raise"),
        auto_restart=bool(spec.get("auto_restart", True)),
        restart_on_reset=bool(spec.get("restart_on_reset", True)),
        cwd=spec.get("cwd"), name=spec["name"],
    )
