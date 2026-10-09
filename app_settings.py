"""User preferences and microphone discovery, independent of the display."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

DEFAULTS = dict(device="", silence=1.0, threshold=0.01, voice=True, captions=True)


def settings_path():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "GemmaLive/settings.json"


def validate_preferences(values):
    if not isinstance(values, dict):
        raise ValueError("Settings must be an object.")
    result = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in values:
            result[key] = values[key]
    if not isinstance(result["device"], str) or any(c in result["device"] for c in "\r\n\x00"):
        raise ValueError("Select a valid microphone.")
    for key in ("voice", "captions"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be enabled or disabled.")
    for key, lower, upper in (("silence", 0.1, 10), ("threshold", 0.0001, 0.5)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key.capitalize()} must be between {lower} and {upper}.")
    return result


def load_preferences(path=None):
    path = Path(path or settings_path())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported settings version")
        return validate_preferences(payload.get("preferences")), ""
    except FileNotFoundError:
        return dict(DEFAULTS), ""
    except (OSError, ValueError, TypeError) as exc:
        return dict(DEFAULTS), f"Saved settings could not be read; defaults restored. {exc}"


def save_preferences(values, path=None):
    values = validate_preferences(values)
    path = Path(path or settings_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids losing good settings on an interrupted write.
    fd, temporary = tempfile.mkstemp(prefix="settings-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(dict(version=1, preferences=values), output, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_microphones(output):
    devices = []
    for line in output.splitlines():
        match = re.search(r'"(.*)"\s+\(audio\)', line)
        if match and match[1] not in devices:
            devices.append(match[1])
    return devices


def discover_microphones():
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true",
                                 "-f", "dshow", "-i", "dummy"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        return [], "FFmpeg is missing. Install it on PATH, then refresh microphones."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"Could not list microphones. Refresh to retry. {exc}"
    # FFmpeg intentionally returns a nonzero exit code for device listing.
    devices = parse_microphones(result.stderr + "\n" + result.stdout)
    return devices, "" if devices else "No microphones found. Connect a microphone, then refresh."
