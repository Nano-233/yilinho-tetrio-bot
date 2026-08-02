"""Profile and app-settings I/O for the control UI."""
import json
import os
import re
import shutil

from bot import (
    CONFIG_FILE,
    DEFAULT_ACTION_DELAY_MS,
    DEFAULT_DELAY_VARIANCE_PERCENT,
    DEFAULT_KEYBINDS,
    DEFAULT_MOVE_DELAY_MS,
    load_config,
    save_config,
)

PROFILES_DIR = "profiles"
APP_SETTINGS_FILE = "app_settings.json"
DEFAULT_PROFILE_NAME = "default"

DEFAULT_APP_HOTKEYS = {
    "start_bot": "f6",
    "stop_bot": "f7",
    "toggle_overlay": "f8",
    "toggle_play_mode": "f9",
}

DEFAULT_AI = {
    "mp": 16,
    "pruning_moves": 5,
    "pruning_breadth": 5,
    "ai_engine": "legacy",  # "legacy" | "cold_clear"
    "spin_ruleset": "all_mini_plus",
    "cc_think_ms": 50,
    "trust_expected_board": True,
    "simple_placements": False,
    "show_autodrop_ghost": True,
}


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\- ]+$")


def _profile_path(name):
    return os.path.join(PROFILES_DIR, f"{name}.json")


def ensure_profiles_dir():
    os.makedirs(PROFILES_DIR, exist_ok=True)


def normalize_profile(config):
    """Fill missing delay / AI / keybind defaults."""
    if config is None:
        config = {}
    config.setdefault("move_delay_ms", DEFAULT_MOVE_DELAY_MS)
    config.setdefault("action_delay_ms", DEFAULT_ACTION_DELAY_MS)
    config.setdefault("delay_variance_percent", DEFAULT_DELAY_VARIANCE_PERCENT)
    config.setdefault("mp", DEFAULT_AI["mp"])
    config.setdefault("pruning_moves", DEFAULT_AI["pruning_moves"])
    config.setdefault("pruning_breadth", DEFAULT_AI["pruning_breadth"])
    config.setdefault("ai_engine", DEFAULT_AI["ai_engine"])
    config.setdefault("spin_ruleset", DEFAULT_AI["spin_ruleset"])
    config.setdefault("cc_think_ms", DEFAULT_AI["cc_think_ms"])
    config.setdefault("trust_expected_board", DEFAULT_AI["trust_expected_board"])
    config.setdefault("simple_placements", DEFAULT_AI["simple_placements"])
    config.setdefault("show_autodrop_ghost", DEFAULT_AI["show_autodrop_ghost"])
    binds = dict(DEFAULT_KEYBINDS)
    binds.update(config.get("keybinds") or {})
    config["keybinds"] = binds
    return config


def empty_profile(label="New profile"):
    return normalize_profile({
        "label": label,
        "screen_offset": [0, 0],
        "screen_resolution": [1920, 1080],
        "board_top_left": [0, 0],
        "board_bottom_right": [100, 200],
        "next_piece_xy_0": [0, 0],
        "next_piece_xy_4": [0, 100],
        "held_piece_xy": [0, 0],
    })


def migrate_legacy_config():
    """Move config.json into profiles/default.json on first UI launch."""
    ensure_profiles_dir()
    dest = _profile_path(DEFAULT_PROFILE_NAME)
    if os.path.exists(dest):
        return
    if os.path.exists(CONFIG_FILE):
        config = load_config(CONFIG_FILE) or empty_profile()
        if not config.get("label"):
            config["label"] = "Current layout"
        save_config(normalize_profile(config), dest)
    else:
        save_config(empty_profile("Default"), dest)


def list_profiles():
    ensure_profiles_dir()
    names = []
    for fname in sorted(os.listdir(PROFILES_DIR)):
        if fname.endswith(".json"):
            names.append(fname[:-5])
    return names


def load_profile(name):
    path = _profile_path(name)
    if not os.path.exists(path):
        return None
    return normalize_profile(load_config(path) or {})


def save_profile(name, config):
    ensure_profiles_dir()
    save_config(normalize_profile(config), _profile_path(name), quiet=True)


def validate_profile_name(name):
    name = (name or "").strip()
    if not name:
        return None, "Name is empty"
    if not _SAFE_NAME.match(name):
        return None, "Use letters, numbers, spaces, _ or -"
    return name, None


def create_profile(name, base=None):
    name, err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    if os.path.exists(_profile_path(name)):
        raise ValueError(f"Profile '{name}' already exists")
    config = normalize_profile(dict(base) if base else empty_profile(name))
    if not config.get("label"):
        config["label"] = name
    save_profile(name, config)
    return name


def duplicate_profile(source, new_name):
    config = load_profile(source)
    if config is None:
        raise ValueError(f"Profile '{source}' not found")
    config = dict(config)
    config["label"] = new_name
    return create_profile(new_name, config)


def delete_profile(name):
    path = _profile_path(name)
    if not os.path.exists(path):
        raise ValueError(f"Profile '{name}' not found")
    if len(list_profiles()) <= 1:
        raise ValueError("Cannot delete the last profile")
    os.remove(path)


def load_app_settings():
    settings = {
        "active_profile": DEFAULT_PROFILE_NAME,
        "hotkeys": dict(DEFAULT_APP_HOTKEYS),
        "play_mode": "autodrop",  # "autodrop" | "suggest"
    }
    if os.path.exists(APP_SETTINGS_FILE):
        try:
            with open(APP_SETTINGS_FILE, "r") as f:
                data = json.load(f)
            settings["active_profile"] = data.get(
                "active_profile", DEFAULT_PROFILE_NAME
            )
            hotkeys = dict(DEFAULT_APP_HOTKEYS)
            hotkeys.update(data.get("hotkeys") or {})
            settings["hotkeys"] = hotkeys
            pm = data.get("play_mode", "autodrop")
            settings["play_mode"] = pm if pm in ("autodrop", "suggest") else "autodrop"
        except (json.JSONDecodeError, IOError):
            pass
    # Ensure active profile exists
    profiles = list_profiles()
    if settings["active_profile"] not in profiles:
        settings["active_profile"] = profiles[0] if profiles else DEFAULT_PROFILE_NAME
    return settings


def save_app_settings(settings):
    out = {
        "active_profile": settings.get("active_profile", DEFAULT_PROFILE_NAME),
        "hotkeys": {
            **DEFAULT_APP_HOTKEYS,
            **(settings.get("hotkeys") or {}),
        },
        "play_mode": settings.get("play_mode", "autodrop"),
    }
    with open(APP_SETTINGS_FILE, "w") as f:
        json.dump(out, f, indent=2)


def sync_legacy_config(name):
    """Keep config.json in sync with the active profile for CLI users."""
    src = _profile_path(name)
    if os.path.exists(src):
        shutil.copyfile(src, CONFIG_FILE)


def repair_monitor_if_stale(config):
    """If profile screen_* no longer matches any monitor, remap onto a scaled match.

    Common on Windows when a display is at 150% DPI: calibration may have been
    stored as 2880x1620 @ offset 3840 while mss now reports 1920x1080 @ 2560.
    """
    if not config or "screen_offset" not in config:
        return config, False
    try:
        import mss
    except ImportError:
        return config, False

    ox, oy = config["screen_offset"]
    sw, sh = config["screen_resolution"]
    with mss.MSS() as sct:
        mons = sct.monitors[1:]
        for mon in mons:
            if (mon["left"] == ox and mon["top"] == oy
                    and mon["width"] == sw and mon["height"] == sh):
                return config, False
        for mon in mons:
            if sw <= 0 or sh <= 0 or mon["width"] <= 0 or mon["height"] <= 0:
                continue
            sx = mon["width"] / sw
            sy = mon["height"] / sh
            if abs(sx - sy) > 0.05 or not (0.4 <= sx <= 2.5):
                continue
            fixed = dict(config)
            fixed["screen_offset"] = [mon["left"], mon["top"]]
            fixed["screen_resolution"] = [mon["width"], mon["height"]]
            for key in (
                "board_top_left",
                "board_bottom_right",
                "next_piece_xy_0",
                "next_piece_xy_4",
                "held_piece_xy",
            ):
                if key in config:
                    x, y = config[key]
                    fixed[key] = [int(round(x * sx)), int(round(y * sy))]
            return normalize_profile(fixed), True
    return config, False
