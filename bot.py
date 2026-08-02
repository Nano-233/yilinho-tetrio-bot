from multiprocessing import Pool
import argparse
import json
import os
import random
import colorsys
import time
import math

import mss
import numpy as np
import pyautogui
from PIL import Image, ImageDraw

from constants import colors, colors_name, tetris_pieces, tetris_pieces_trimmed, NUM_ROW, NUM_COL
from tetris_ai import find_best_move as find_best_move_legacy
from tetris_ai import _get_board_terrain

CONFIG_FILE = "config.json"

_DPI_AWARE = False
# bettercam: Desktop Duplication — needed when TETR.IO is on a discrete-GPU
# monitor (mss BitBlt returns a black frame there).
_BETTERCAM = None  # ((width, height, device, output), camera) | None
_BETTERCAM_TRIED = False
# DXGI's AcquireNextFrame(0, ...) is a non-blocking poll: grab() returns None
# whenever the desktop hasn't repainted since the last grab, which happens
# constantly when polling faster than the game redraws. That is NOT a
# capture failure — the last frame is still accurate — but treating it like
# one used to fall through to mss, which returns solid black on this GPU's
# monitor. Cache the last good frame and reuse it instead.
_BETTERCAM_LAST_FRAME = None
_BETTERCAM_IMPORT_WARNED = False

# DXGI VendorId: NVIDIA=0x10DE (4318), Intel=0x8086 (32902), AMD=0x1002
_NVIDIA_VENDOR = 4318


def _parse_bettercam_topology():
    """List (device_idx, output_idx, w, h, primary, is_nvidia, name) without opening cams."""
    import bettercam
    import re

    devices = {}  # idx -> (name, is_nvidia)
    raw = bettercam.device_info().replace("\n", " ")
    for m in re.finditer(r"Device\[(\d+)\]:<Device Name:(.+?)>", raw):
        idx = int(m.group(1))
        blob = m.group(2)
        is_nv = ("NVIDIA" in blob.upper()) or (f"VendorId:{_NVIDIA_VENDOR}" in blob)
        name = blob.split(" Dedicated")[0].strip()
        devices[idx] = (name, is_nv)

    outs = []
    for m in re.finditer(
        r"Device\[(\d+)\] Output\[(\d+)\]: Res:\((\d+), (\d+)\)[^\n]*Primary:(\w+)",
        bettercam.output_info(),
    ):
        d, o = int(m.group(1)), int(m.group(2))
        w, h = int(m.group(3)), int(m.group(4))
        primary = m.group(5).lower() == "true"
        name, is_nv = devices.get(d, (f"Device{d}", False))
        outs.append((d, o, w, h, primary, is_nv, name))
    return outs


def _pick_bettercam_target(screen_resolution, screen_offset=None):
    """Pick DXGI (device, output) for the game monitor — prefer NVIDIA, no probe creates."""
    sw, sh = int(screen_resolution[0]), int(screen_resolution[1])
    try:
        outs = _parse_bettercam_topology()
    except Exception as e:
        print(f"bettercam topology parse failed: {e}", flush=True)
        outs = []

    if not outs:
        # Last resort: try common pairs without blasting 4x4 creates
        return _pick_bettercam_target_probe(screen_resolution)

    scored = []
    for d, o, w, h, primary, is_nv, name in outs:
        if w <= 0 or h <= 0:
            continue
        if (w, h) == (sw, sh):
            # Exact size: prefer NVIDIA, then non-primary (external) for laptop+dGPU setups
            score = (0 if is_nv else 1, 0 if not primary else 1, d, o)
            scored.append((score, d, o, w, h, is_nv, name))
            continue
        sx, sy = sw / w, sh / h
        if abs(sx - sy) > 0.08:
            continue
        # Scale mismatch: worse than exact; still prefer NVIDIA
        score = (2, abs(math.log(sx)), 0 if is_nv else 1, d, o)
        scored.append((score, d, o, w, h, is_nv, name))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    _, d, o, w, h, is_nv, name = scored[0]
    return d, o, w, h, is_nv, name


def _pick_bettercam_target_probe(screen_resolution):
    """Fallback: open only NVIDIA-first candidates, never a 4x4 blast."""
    import bettercam

    sw, sh = int(screen_resolution[0]), int(screen_resolution[1])
    # Prefer device 1 (typical dGPU) before 0 (iGPU)
    order = [(1, 0), (0, 0), (1, 1), (0, 1), (2, 0), (2, 1)]
    best = None
    for d, o in order:
        cam = None
        try:
            cam = bettercam.create(device_idx=d, output_idx=o, output_color="RGB")
            w, h = int(cam.width), int(cam.height)
        except Exception:
            continue
        finally:
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
                try:
                    del cam
                except Exception:
                    pass
        if w <= 0 or h <= 0:
            continue
        if (w, h) == (sw, sh):
            return (d, o, w, h, d >= 1, f"device{d}")
        sx, sy = sw / w, sh / h
        if abs(sx - sy) > 0.08:
            continue
        score = abs(math.log(sx))
        cand = (score, d, o, w, h)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[1] >= 1, f"device{best[1]}"


def _grab_bettercam(screen_resolution, screen_offset=None):
    """Grab via bettercam, resized to config resolution. None if unavailable."""
    global _BETTERCAM, _BETTERCAM_TRIED, _BETTERCAM_LAST_FRAME, _BETTERCAM_IMPORT_WARNED
    width, height = int(screen_resolution[0]), int(screen_resolution[1])
    try:
        import bettercam  # noqa: F401
    except ImportError:
        if not _BETTERCAM_IMPORT_WARNED:
            _BETTERCAM_IMPORT_WARNED = True
            print(
                "bettercam not installed — using mss (may capture black on a "
                "GPU-driven monitor). pip install bettercam to fix.",
                flush=True,
            )
        return None

    # Key includes device so a profile swap can reopen the right adapter
    key = (width, height)
    if _BETTERCAM is not None and _BETTERCAM[0][:2] != key:
        try:
            _BETTERCAM[1].release()
        except Exception:
            pass
        _BETTERCAM = None
        _BETTERCAM_TRIED = False
        _BETTERCAM_LAST_FRAME = None

    if _BETTERCAM is None and not _BETTERCAM_TRIED:
        _BETTERCAM_TRIED = True
        pick = _pick_bettercam_target(screen_resolution, screen_offset)
        if pick is None:
            print(
                "bettercam: no matching output for "
                f"{width}x{height} — falling back to mss (may capture black "
                "on a GPU-driven monitor)",
                flush=True,
            )
            return None
        if len(pick) == 6:
            d, o, nw, nh, is_nv, name = pick
        else:
            d, o, nw, nh = pick[:4]
            is_nv, name = False, f"device{d}"
        try:
            # nvidia_gpu=True is CuPy color convert — more GPU load, not "pick NVIDIA".
            # DXGI duplication already runs on the adapter that owns the output.
            cam = bettercam.create(device_idx=d, output_idx=o, output_color="RGB")
        except Exception as e:
            print(f"bettercam open failed: {e}", flush=True)
            return None
        _BETTERCAM = ((width, height, d, o), cam)
        tag = "NVIDIA" if is_nv else "non-NVIDIA"
        print(
            f"Capture: bettercam {tag} '{name}' device={d} output={o} "
            f"native={nw}x{nh} -> config {width}x{height}",
            flush=True,
        )

    if _BETTERCAM is None:
        return None
    frame = _BETTERCAM[1].grab()
    if frame is None:
        # No repaint since the last grab — briefly retry, then fall back to
        # the last real frame (still accurate) rather than let the caller
        # hit mss's black-frame path on this monitor.
        for _ in range(5):
            time.sleep(0.004)
            frame = _BETTERCAM[1].grab()
            if frame is not None:
                break
        if frame is None:
            return _BETTERCAM_LAST_FRAME
    img = Image.fromarray(frame)
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    _BETTERCAM_LAST_FRAME = img
    return img


def capture_screen(screen_offset, screen_resolution):
    """Grab a screen region. Coords match pyautogui / calibration.

    Prefers bettercam (DXGI) so discrete-GPU monitors aren't black. Falls
    back to mss. Resizes to screen_resolution so calibration coords line up.

    Note: DXGI must capture from the GPU that *drives* that monitor (Intel for
    the laptop panel, NVIDIA for a dGPU-attached display). That is not the same
    as CUDA — enabling bettercam's nvidia_gpu flag would only add CuPy load.
    """
    ensure_dpi_awareness()
    width, height = int(screen_resolution[0]), int(screen_resolution[1])

    img = _grab_bettercam(screen_resolution, screen_offset)
    if img is not None:
        return img

    left, top = screen_offset
    with mss.MSS() as sct:
        region = {
            "left": int(left),
            "top": int(top),
            "width": max(1, width),
            "height": max(1, height),
        }
        shot = sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        return img


def ensure_dpi_awareness():
    """Align mss/pyautogui with the UI's coordinate space.

    Creating a Qt QApplication (if none exists) makes mss report the same
    multi-monitor layout the overlay/calibration UI uses. Calling Win32
    SetProcessDpiAwareness *before* Qt breaks secondary-monitor sizes on
    scaled displays (1920 logical vs 2880 physical).
    """
    global _DPI_AWARE
    if _DPI_AWARE:
        return
    try:
        from PyQt5.QtWidgets import QApplication
        if QApplication.instance() is None:
            # ponytail: headless QApp so CLI --use-config matches UI-calibrated coords
            QApplication([])
    except Exception:
        pass
    _DPI_AWARE = True


# Piece hues on PIL's 0-255 HSV scale, in the same order as `colors`.
# Matching on hue (not raw RGB) survives TETR.IO greying out the held piece
# after a hold: that drops saturation/brightness but leaves hue intact.
PIECE_HUES = np.array(
    [colorsys.rgb_to_hsv(*[ch / 255 for ch in c])[0] * 255 for c in colors]
)
# A pixel must be this colourful/bright to count as part of a piece. Measured
# margin: pieces dimmed to 25% still read sat>=86 val>=74, while TETR.IO's dark
# (slightly blue) backgrounds top out at sat 73 val 60. Without these floors,
# background pixels match J's blue-purple hue and phantom pieces appear.
PIECE_MIN_SAT = 80
PIECE_MIN_VAL = 65
PIECE_MAX_HUE_DIST = 8  # closest two piece hues are ~17.8 apart
PIECE_MIN_PIXELS = 8
# Garbage blocks are a flat grey (near-zero saturation) — unlike any piece
# hue, so they need their own gate instead of a low sat/val fallback. A loose
# "bright enough" fallback here previously also matched hue-independent bleed
# from a custom (non-default) TETR.IO background, painting phantom blocks
# wherever the scenery showed through the board. If your board background is
# NOT fully opaque, board reads will be unreliable no matter the thresholds —
# set it to a solid/opaque background in TETR.IO settings.
GARBAGE_MAX_SAT = 25
GARBAGE_MIN_VAL = 90


def cell_filled(hue, sat, val) -> bool:
    """True if a cell's sampled HSV pixels (each an array) look like a real
    mino or garbage block, rather than empty board / ghost tint."""
    hue, sat, val = np.asarray(hue), np.asarray(sat), np.asarray(val)
    lit = (sat >= PIECE_MIN_SAT) & (val >= PIECE_MIN_VAL)
    if np.any(lit):
        dist = np.abs(hue[lit][:, None] - PIECE_HUES[None, :])
        dist = np.minimum(dist, 255 - dist)
        if np.any(dist.min(axis=1) <= PIECE_MAX_HUE_DIST):
            return True
    return bool(sat.mean() <= GARBAGE_MAX_SAT and val.mean() >= GARBAGE_MIN_VAL)
# Greyed-out held piece after a hold — lower sat/val but hue still valid
HELD_MIN_SAT = 35
HELD_MIN_VAL = 30
HELD_MIN_PIXELS = 4
QUEUE_READ_RETRIES = 6
QUEUE_READ_DELAY = 0.025
MAX_UNREADABLE_POLLS = 40
SPAWN_MASK_ROWS = 4       # top rows — active piece lives here, hide from AI
LOCK_SETTLE_SEC = 0.05    # wait after hard drop before next read
BOARD_STABLE_TRIES = 4
BOARD_STABLE_GAP = 0.025
SPAWN_COLUMN = 3          # default piece spawn column for movement math (legacy left-edge)
SPAWN_CENTER_X = 4        # guideline SRS center x (Cold Clear / TBP)
SPAWN_SETTLE_SEC = 0.035  # wait after queue shift before reading pieces
MIN_INPUT_GAP_SEC = 0.001   # ~1 frame when move/action delay is 0
DEFAULT_PAUSE_SEC = 3.0   # --pause: seconds to wait before each drop

# Default delay settings (in milliseconds)
DEFAULT_MOVE_DELAY_MS = 0
DEFAULT_ACTION_DELAY_MS = 0
DEFAULT_DELAY_VARIANCE_PERCENT = 0
CALIBRATION_COUNTDOWN_SEC = 5
STARTUP_COUNTDOWN_SEC = 5

# TETR.IO default binds. Override these in config.json under "keybinds"
# if you use a custom layout (e.g. WASD to move, arrows to rotate).
DEFAULT_KEYBINDS = {
    "move_left": "left",
    "move_right": "right",
    "soft_drop": "down",
    "hard_drop": "space",
    "rotate_cw": "x",
    "rotate_ccw": "z",
    "rotate_180": "a",
    "hold": "c",
}

# pyautogui adds ~0.1s between actions by default — too slow for tetris
pyautogui.PAUSE = 0

try:
    import pydirectinput as _pydirectinput
    _pydirectinput.PAUSE = 0
    _USE_DIRECT_INPUT = True
except ImportError:
    _pydirectinput = None
    _USE_DIRECT_INPUT = False


def tap_key(key):
    """Press and release a key. pydirectinput on Windows for browser games."""
    if _USE_DIRECT_INPUT:
        _pydirectinput.keyDown(key)
        _pydirectinput.keyUp(key)
    else:
        pyautogui.keyDown(key)
        pyautogui.keyUp(key)


def hold_key(key):
    if _USE_DIRECT_INPUT:
        _pydirectinput.keyDown(key)
    else:
        pyautogui.keyDown(key)


def release_key(key):
    if _USE_DIRECT_INPUT:
        _pydirectinput.keyUp(key)
    else:
        pyautogui.keyUp(key)


def monitor_containing(x, y):
    """Return the mss monitor dict that contains point (x, y)."""
    ensure_dpi_awareness()
    with mss.MSS() as sct:
        for mon in sct.monitors[1:]:
            if (mon["left"] <= x < mon["left"] + mon["width"] and
                    mon["top"] <= y < mon["top"] + mon["height"]):
                return mon
        return sct.monitors[1]


def countdown_capture(label, instruction, seconds=CALIBRATION_COUNTDOWN_SEC):
    """Countdown, then capture the current mouse position (works across Spaces)."""
    print(f"\n--- {label} ---")
    print(f"  {instruction}")
    print(f"  1. Press Enter here to start a {seconds}s countdown")
    print(f"  2. Switch to TETR.IO and hover the target")
    print(f"  3. Keep the mouse still until capture")
    input("  >> Press Enter to start countdown... ")
    for remaining in range(seconds, 0, -1):
        print(f"  Capturing in {remaining}...", flush=True)
        time.sleep(1)
    pos = pyautogui.position()
    print(f"  Captured: ({pos[0]}, {pos[1]})")
    return (pos[0], pos[1])

def load_config(config_path=CONFIG_FILE):
    """Load configuration from JSON if it exists.
    Applies default values for any missing delay settings."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Apply defaults for delay settings if not present
            if 'move_delay_ms' not in config:
                config['move_delay_ms'] = DEFAULT_MOVE_DELAY_MS
            if 'action_delay_ms' not in config:
                config['action_delay_ms'] = DEFAULT_ACTION_DELAY_MS
            if 'delay_variance_percent' not in config:
                config['delay_variance_percent'] = DEFAULT_DELAY_VARIANCE_PERCENT
            return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
    return None


def save_config(config, config_path=CONFIG_FILE, quiet=False):
    """Save configuration to JSON."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    if not quiet:
        print(f"\nConfiguration saved to {config_path}")


def format_board_rows(board, rows=4):
    """Top rows of the board (# = filled). Row 0 is the top/spawn area."""
    lines = []
    for r in range(min(rows, board.shape[0])):
        lines.append(''.join('#' if board[r, c] else '.' for c in range(board.shape[1])))
    return '\n'.join(lines)


def format_board_full(board, mark_col=None):
    """Full board for debug. Row 0 = top (spawn), row 19 = bottom — matches TETR.IO."""
    lines = ['      ' + ' '.join(str(c) for c in range(NUM_COL)) + '  (columns)']
    lines.append('      ' + 'v top/spawn'.ljust(NUM_COL * 2))
    for r in range(board.shape[0]):
        cells = ' '.join('#' if board[r, c] else '.' for c in range(board.shape[1]))
        lines.append(f'{r:2d} | {cells}')
    lines.append('      ' + '^ bottom'.ljust(NUM_COL * 2))
    if mark_col is not None and 0 <= mark_col < NUM_COL:
        pointer = [' '] * (NUM_COL * 2 - 1)
        pointer[mark_col * 2] = '^'
        lines.append('      ' + ''.join(pointer))
    return '\n'.join(lines)


def describe_inputs(keys, col, rotations, need_hold, offset):
    """Human-readable summary of keys the bot will send."""
    parts = []
    if need_hold:
        parts.append('hold')
    rot = rotations[0] if rotations else 0
    if rot == 1:
        parts.append('rot_cw')
    elif rot == 2:
        parts.append('rot_180')
    elif rot == 3:
        parts.append('rot_ccw')
    target = col - offset
    delta = target - SPAWN_COLUMN
    if delta < 0:
        parts.append(f'left x{abs(delta)}')
    elif delta > 0:
        parts.append(f'right x{delta}')
    if len(rotations) > 1:
        parts.append(f'spin({list(rotations[1:])})')
    parts.append('hard_drop')
    return ', '.join(parts)


def classify_piece(image, center_fraction=None, min_sat=None, min_val=None, min_pixels=None):
    """Return the piece letter shown in a cropped preview, or None.

    Matches on hue rather than raw RGB for two reasons: TETR.IO dims the held
    piece after a hold (which changes brightness/saturation but not hue), and
    an unmatched sample must report None. The old nearest-colour approach had
    no cutoff, so empty dark background always resolved to the darkest piece
    colour (J) and silently fed the AI a queue of phantom pieces.

    center_fraction: if set (e.g. 0.5), only the inner portion is used. The
    hold preview has a purple UI border that otherwise reads as a T piece.
    """
    min_sat = PIECE_MIN_SAT if min_sat is None else min_sat
    min_val = PIECE_MIN_VAL if min_val is None else min_val
    min_pixels = PIECE_MIN_PIXELS if min_pixels is None else min_pixels

    if center_fraction:
        w, h = image.size
        mx = int(w * (1 - center_fraction) / 2)
        my = int(h * (1 - center_fraction) / 2)
        image = image.crop((mx, my, w - mx, h - my))

    hsv = np.array(image.convert("HSV")).reshape(-1, 3).astype(np.int32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    lit = (sat >= min_sat) & (val >= min_val)
    if not np.any(lit):
        return None

    distance = np.abs(hue[lit][:, None] - PIECE_HUES[None, :])
    distance = np.minimum(distance, 255 - distance)
    matched = distance.argmin(axis=1)[distance.min(axis=1) <= PIECE_MAX_HUE_DIST]
    if matched.size < min_pixels:
        return None

    return colors_name[np.bincount(matched, minlength=len(colors)).argmax()]


def startup_countdown(seconds=STARTUP_COUNTDOWN_SEC):
    """Give the user time to switch to the TETR.IO window."""
    print(f"\nSwitch to TETR.IO now — starting in {seconds}s...", flush=True)
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...", flush=True)
        time.sleep(1)
    print("  Go!\n", flush=True)


def prompt_keybinds():
    """Ask for the TETR.IO keybinds, defaulting to TETR.IO's stock layout."""
    print("\n--- KEYBINDS ---")
    print("  These must match your TETR.IO controls exactly.")
    print("  Press Enter to accept the default shown in [brackets].")
    print("  Use names like: left, right, up, down, space, a, w, s, d, shift\n")

    labels = {
        "move_left": "Move left",
        "move_right": "Move right",
        "soft_drop": "Soft drop",
        "hard_drop": "Hard drop",
        "rotate_cw": "Rotate clockwise",
        "rotate_ccw": "Rotate counter-clockwise",
        "rotate_180": "Rotate 180",
        "hold": "Hold",
    }
    keybinds = {}
    for action, default in DEFAULT_KEYBINDS.items():
        answer = input(f"  {labels[action]} [{default}]: ").strip().lower()
        keybinds[action] = answer or default
    return keybinds


def run_detection_test(bot):
    """Print what the bot currently sees, to verify calibration."""
    print("Detection test — sampling every second. Ctrl+C to stop.\n")
    try:
        while True:
            bot.refresh_screen_image()
            next_pieces = bot.get_next_pieces()
            held = bot.get_held_piece()
            board = bot.get_tetris_board()
            filled = int(board.sum())

            means = []
            for xy in bot.next_piece_xy:
                rgb = np.array(bot.sample_box(xy)).reshape(-1, 3).mean(axis=0)
                means.append("(" + ",".join(f"{int(c):3}" for c in rgb) + ")")
            print(f"next={next_pieces}  held={held}  filled={filled}", flush=True)
            print(f"  next avg RGB: {' '.join(means)}", flush=True)
            if any(p is None for p in next_pieces):
                print("  ^ None = no piece colour at that point. Run --snapshot "
                      "to see where it is sampling.", flush=True)
            if filled == NUM_ROW * NUM_COL:
                print("  ^ board reads completely full — board coords are wrong", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


def save_snapshot(bot, path="snapshot.png"):
    """Save a screenshot annotated with every region the bot samples."""
    bot.refresh_screen_image()
    img = bot.screen_image.copy()
    draw = ImageDraw.Draw(img)
    half = bot.pixel_area_half

    for i, (x, y) in enumerate(bot.next_piece_xy):
        draw.rectangle([x - half, y - half, x + half, y + half],
                       outline=(0, 255, 0), width=2)
        draw.text((x + half + 4, y - 6), f"next{i}", fill=(0, 255, 0))

    hx, hy = bot.held_piece_xy
    draw.rectangle([hx - half, hy - half, hx + half, hy + half],
                   outline=(255, 0, 255), width=2)
    draw.text((hx + half + 4, hy - 6), "held", fill=(255, 0, 255))

    left, top = bot.board_top_left
    right, bottom = bot.board_bottom_right
    draw.rectangle([left, top, right, bottom], outline=(255, 255, 0), width=2)
    cell_w = (right - left) / NUM_COL
    cell_h = (bottom - top) / NUM_ROW
    for col in range(1, NUM_COL):
        draw.line([left + col * cell_w, top, left + col * cell_w, bottom],
                  fill=(255, 255, 0))
    for row in range(1, NUM_ROW):
        draw.line([left, top + row * cell_h, right, top + row * cell_h],
                  fill=(255, 255, 0))

    img.save(path)
    print(f"\nSaved {path}")
    print("  green  = next-piece samples (must sit on the coloured blocks)")
    print("  purple = held-piece sample")
    print("  yellow = board grid (cells must line up with the playfield)")
    print(f"\n  next={bot.get_next_pieces()}  held={bot.get_held_piece()}")


def run_calibration_wizard(config_path=CONFIG_FILE):
    """Capture board coordinates via countdown (no global hotkeys — macOS-safe)."""
    existing = load_config(config_path)
    print("\n" + "=" * 60)
    print("       TETRIOBOT CALIBRATION WIZARD (macOS)")
    print("=" * 60)
    print(f"\nSaving to: {config_path}")
    print("\nOpen TETR.IO first and leave the board visible.")
    print("Each step uses a countdown so you can three-finger-swipe")
    print("to the game, hover the spot, and wait for capture.\n")
    print("Required macOS permissions for later (not needed for calibrate):")
    print("  System Settings → Privacy & Security →")
    print("    • Accessibility (Terminal / iTerm / Cursor)")
    print("    • Screen Recording (same app)\n")

    steps = [
        ("Step 1/5: BOARD TOP-LEFT",
         "Hover the TOP-LEFT corner of the playfield (where pieces stack)"),
        ("Step 2/5: BOARD BOTTOM-RIGHT",
         "Hover the BOTTOM-RIGHT corner of the playfield"),
        ("Step 3/5: NEXT PIECE #1",
         "Hover the CENTER of the FIRST (top) next-piece preview"),
        ("Step 4/5: NEXT PIECE #5",
         "Hover the CENTER of the FIFTH (bottom) next-piece preview"),
        ("Step 5/5: HELD PIECE",
         "Hover the CENTER of the HELD piece display"),
    ]

    absolute = [countdown_capture(name, instruction) for name, instruction in steps]
    board_top_left_abs = absolute[0]
    board_bottom_right_abs = absolute[1]
    next_piece_0_abs = absolute[2]
    next_piece_4_abs = absolute[3]
    held_piece_abs = absolute[4]

    mon = monitor_containing(*board_top_left_abs)
    screen_offset = (mon["left"], mon["top"])
    screen_resolution = (mon["width"], mon["height"])
    ox, oy = screen_offset

    def rel(pt):
        return [pt[0] - ox, pt[1] - oy]

    board_top_left = rel(board_top_left_abs)
    board_bottom_right = rel(board_bottom_right_abs)
    next_piece_0 = rel(next_piece_0_abs)
    next_piece_4 = rel(next_piece_4_abs)
    held_piece = rel(held_piece_abs)

    print(f"\n--- Screen ---")
    print(f"  Monitor offset: {screen_offset}")
    print(f"  Monitor size:   {screen_resolution}")

    print(f"\n--- Step 6/6: DELAY SETTINGS ---")
    print("  Configure delays to make inputs appear more human-like.\n")

    print(f"  Move Delay (default: {DEFAULT_MOVE_DELAY_MS}ms)")
    move_delay_input = input("  Enter move delay in ms (or Enter for default): ").strip()
    move_delay_ms = int(move_delay_input) if move_delay_input else DEFAULT_MOVE_DELAY_MS

    print(f"\n  Action Delay (default: {DEFAULT_ACTION_DELAY_MS}ms)")
    action_delay_input = input("  Enter action delay in ms (or Enter for default): ").strip()
    action_delay_ms = int(action_delay_input) if action_delay_input else DEFAULT_ACTION_DELAY_MS

    print(f"\n  Delay Variance (default: {DEFAULT_DELAY_VARIANCE_PERCENT}%)")
    variance_input = input("  Enter variance percentage (or Enter for default): ").strip()
    delay_variance_percent = int(variance_input) if variance_input else DEFAULT_DELAY_VARIANCE_PERCENT

    keybinds = prompt_keybinds()

    config = {
        "screen_offset": list(screen_offset),
        "screen_resolution": list(screen_resolution),
        "board_top_left": board_top_left,
        "board_bottom_right": board_bottom_right,
        "next_piece_xy_0": next_piece_0,
        "next_piece_xy_4": next_piece_4,
        "held_piece_xy": held_piece,
        "move_delay_ms": move_delay_ms,
        "action_delay_ms": action_delay_ms,
        "delay_variance_percent": delay_variance_percent,
        "keybinds": keybinds,
    }
    if existing and existing.get("label"):
        config["label"] = existing["label"]

    print("\n" + "=" * 60)
    print("       CALIBRATION COMPLETE!")
    print("=" * 60)
    print(f"\n  Board Top-Left:     {board_top_left}")
    print(f"  Board Bottom-Right: {board_bottom_right}")
    print(f"  Next Piece #1:      {next_piece_0}")
    print(f"  Next Piece #5:      {next_piece_4}")
    print(f"  Held Piece:         {held_piece}")
    print(f"\n  Delays: move={move_delay_ms}ms action={action_delay_ms}ms variance={delay_variance_percent}%")
    print(f"  Keybinds: {keybinds}")

    save_config(config, config_path)
    print("\nRun the bot with:")
    print(f"  python3 bot.py --config {config_path}\n")
    return config


wait_time = 0.01
soft_drop_delay = 0.03
# Extra wait after line clears — TETR.IO animates; mid-clear vision desyncs CC.
CLEAR_SETTLE_SEC = 0.12
# Game Settings - DAS 40ms, ARR 0ms, SDF max, lowest graphic


def input_gap_sec(base_delay_ms, variance_percent):
    """Delay between inputs; 0 ms config still leaves one frame for the game."""
    if base_delay_ms <= 0:
        return MIN_INPUT_GAP_SEC
    return get_delay_with_variance(base_delay_ms, variance_percent)


def get_delay_with_variance(base_delay_ms, variance_percent):
    """Calculate delay with random variance.
    
    Args:
        base_delay_ms: Base delay in milliseconds
        variance_percent: Variance percentage (e.g., 20 for ±20%)
    
    Returns:
        Delay in seconds (for use with time.sleep)
    """
    if base_delay_ms <= 0:
        return 0
    variance = variance_percent / 100.0
    actual_delay_ms = base_delay_ms * (1 + random.uniform(-variance, variance))
    return max(0, actual_delay_ms / 1000.0)  # Convert to seconds


class TetrioBot:
    def __init__(
        self,
        screen_offset,
        screen_resolution,
        board_top_left,
        board_bottom_right,
        next_piece_xy_0,
        next_piece_xy_4,
        held_piece_xy,
        pruning_moves,
        pruning_breadth,
        mp,
        move_delay_ms=DEFAULT_MOVE_DELAY_MS,
        action_delay_ms=DEFAULT_ACTION_DELAY_MS,
        delay_variance_percent=DEFAULT_DELAY_VARIANCE_PERCENT,
        keybinds=None,
        debug=False,
        pause_sec=0,
        ai_engine="legacy",
        cc_binary=None,
        cc_weights=None,
        play_mode="autodrop",
        spin_ruleset="all_mini_plus",
        cc_think_ms=50,
        trust_expected_board=True,
    ):
        self.keys = dict(DEFAULT_KEYBINDS)
        if keybinds:
            self.keys.update(keybinds)
        self.debug = debug or pause_sec > 0
        self.pause_sec = pause_sec
        self.move_count = 0
        self.ai_engine = ai_engine or "legacy"
        self.cc_binary = cc_binary
        self.cc_weights = cc_weights
        self.play_mode = play_mode or "autodrop"  # "autodrop" | "suggest"
        try:
            from spin_path import normalize_ruleset
            self.spin_ruleset = normalize_ruleset(spin_ruleset)
        except Exception:
            self.spin_ruleset = spin_ruleset or "all_mini_plus"
        self.cc_think_ms = 50 if cc_think_ms is None else int(cc_think_ms)
        self.trust_expected_board = bool(trust_expected_board)
        # Optional callback: publish_ghost([(col, row_top), ...], label)
        self.publish_ghost = None
        self._cc_ai_board = None  # bottom-up board at last CC decision
        # After a confirmed placement, next cycle uses calculated expected board
        # instead of re-reading vision (spawn settle + stable capture).
        self._board_trusted = False
        self.screen_offset = screen_offset
        self.screen_resolution = screen_resolution
        self.board_top_left = board_top_left
        self.board_bottom_right = board_bottom_right

        x0, y0 = next_piece_xy_0
        x4, y4 = next_piece_xy_4
        self.next_piece_xy = (
            next_piece_xy_0,
            ((x0+x4)//2, y0 + math.floor(((y4-y0)/4)*1)),
            ((x0+x4)//2, y0 + math.floor(((y4-y0)/4)*2)),
            ((x0+x4)//2, y0 + math.floor(((y4-y0)/4)*3)),
            next_piece_xy_4
        )
        pixel_area = (y4 - y0) // NUM_COL
        self.pixel_area_half = max(1, pixel_area // 2)
        self.held_piece_xy = held_piece_xy

        self.pruning_moves = pruning_moves
        self.pruning_breadth = pruning_breadth
        self._mp_workers = int(mp) if mp else 1
        # Cold Clear is a single subprocess — a legacy mp Pool here only
        # spawns ~16 idle Python processes and hangs on shutdown (AppHang).
        self.mp_pool = None
        if self.ai_engine != "cold_clear" and self._mp_workers > 1:
            self.mp_pool = Pool(processes=self._mp_workers)
        
        # Delay settings
        self.move_delay_ms = move_delay_ms
        self.action_delay_ms = action_delay_ms
        self.delay_variance_percent = delay_variance_percent

        self.last_good_next = None
        self.last_held_piece = None
        self.hold_used_this_piece = False
        self._stop = False

        self.screen_image = Image.new("RGB", self.screen_resolution)
        self.refresh_screen_image()

    def stop(self):
        """Request exit ASAP — unblock CC suggest and release stuck keys."""
        self._stop = True
        try:
            for k in set(self.keys.values()):
                try:
                    pyautogui.keyUp(k)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            from cold_clear import shutdown_shared_engine
            shutdown_shared_engine()
        except Exception:
            pass

    def _sleep(self, seconds: float) -> bool:
        """Sleep in short slices; return False if stop was requested."""
        if seconds <= 0:
            return not self._stop
        end = time.time() + seconds
        while time.time() < end:
            if self._stop:
                return False
            time.sleep(min(0.05, end - time.time()))
        return not self._stop

    def apply_config(self, config):
        """Hot-reload geometry, delays, and keybinds from a profile dict."""
        self.screen_offset = tuple(config["screen_offset"])
        self.screen_resolution = tuple(config["screen_resolution"])
        self.board_top_left = tuple(config["board_top_left"])
        self.board_bottom_right = tuple(config["board_bottom_right"])
        next_0 = tuple(config["next_piece_xy_0"])
        next_4 = tuple(config["next_piece_xy_4"])
        x0, y0 = next_0
        x4, y4 = next_4
        self.next_piece_xy = (
            next_0,
            ((x0 + x4) // 2, y0 + math.floor(((y4 - y0) / 4) * 1)),
            ((x0 + x4) // 2, y0 + math.floor(((y4 - y0) / 4) * 2)),
            ((x0 + x4) // 2, y0 + math.floor(((y4 - y0) / 4) * 3)),
            next_4,
        )
        pixel_area = (y4 - y0) // NUM_COL
        self.pixel_area_half = max(1, pixel_area // 2)
        self.held_piece_xy = tuple(config["held_piece_xy"])
        self.move_delay_ms = config.get("move_delay_ms", DEFAULT_MOVE_DELAY_MS)
        self.action_delay_ms = config.get("action_delay_ms", DEFAULT_ACTION_DELAY_MS)
        self.delay_variance_percent = config.get(
            "delay_variance_percent", DEFAULT_DELAY_VARIANCE_PERCENT
        )
        self.keys = dict(DEFAULT_KEYBINDS)
        self.keys.update(config.get("keybinds") or {})
        if "pruning_moves" in config:
            self.pruning_moves = config["pruning_moves"]
        if "pruning_breadth" in config:
            self.pruning_breadth = config["pruning_breadth"]
        if "ai_engine" in config:
            self.ai_engine = config["ai_engine"] or "legacy"
        if "cc_binary" in config:
            self.cc_binary = config.get("cc_binary")
        if "cc_weights" in config:
            self.cc_weights = config.get("cc_weights")
        if "cc_think_ms" in config:
            v = config.get("cc_think_ms")
            self.cc_think_ms = 50 if v is None else int(v)
        if "trust_expected_board" in config:
            self.trust_expected_board = bool(config.get("trust_expected_board"))
        if "spin_ruleset" in config:
            try:
                from spin_path import normalize_ruleset
                self.spin_ruleset = normalize_ruleset(config.get("spin_ruleset"))
            except Exception:
                self.spin_ruleset = config.get("spin_ruleset") or self.spin_ruleset

    def refresh_screen_image(self):
        self.screen_image = capture_screen(self.screen_offset, self.screen_resolution)

    def sample_box(self, xy):
        x, y = xy
        return self.screen_image.crop((
            x - self.pixel_area_half,
            y - self.pixel_area_half,
            x + self.pixel_area_half,
            y + self.pixel_area_half
        ))

    def get_next_pieces(self):
        return [classify_piece(self.sample_box(xy)) for xy in self.next_piece_xy]

    def read_next_pieces(self):
        """Read the next queue, retrying through scroll/line-clear animations."""
        best = None
        best_count = -1
        for _ in range(QUEUE_READ_RETRIES):
            self.refresh_screen_image()
            pieces = self.get_next_pieces()
            count = sum(p is not None for p in pieces)
            if count == len(pieces):
                self.last_good_next = pieces
                return pieces
            if count > best_count:
                best, best_count = pieces, count
            time.sleep(QUEUE_READ_DELAY)

        if best and self.last_good_next:
            merged = [
                p if p is not None else self.last_good_next[i]
                for i, p in enumerate(best)
            ]
            if all(p is not None for p in merged):
                self.last_good_next = merged
                return merged

        if best and best_count >= 1:
            self.last_good_next = best
        return best

    def get_held_piece(self):
        # Inner 50% avoids the hold box's purple UI border reading as T.
        # Relaxed thresholds catch the greyed-out piece after a hold.
        return classify_piece(
            self.sample_box(self.held_piece_xy),
            center_fraction=0.5,
            min_sat=HELD_MIN_SAT,
            min_val=HELD_MIN_VAL,
            min_pixels=HELD_MIN_PIXELS,
        )

    def read_held_piece(self):
        held = self.get_held_piece()
        if held is not None:
            self.last_held_piece = held
        elif self.hold_used_this_piece and self.last_held_piece is not None:
            # Greyed-out held piece can still fall below thresholds mid-animation.
            held = self.last_held_piece
        return held

    def board_looks_obscured(self, board_image_L=None) -> bool:
        """True if a bright overlay (e.g. OUT OF FOCUS) covers mid-board.

        Uses a majority vote, not an average: thin yellow calib grid lines can
        sit on a sample pixel and inflate the mean without covering the field.
        """
        if board_image_L is None:
            board_image_L = self.screen_image.crop((
                self.board_top_left[0],
                self.board_top_left[1],
                self.board_bottom_right[0],
                self.board_bottom_right[1],
            )).convert("L")
        w, h = board_image_L.size
        # Sample a horizontal band through mid-board (where the banner sits)
        ys = (h // 2 - h // 20, h // 2, h // 2 + h // 20)
        xs = (w // 4, w // 2, 3 * w // 4)
        vals = [board_image_L.getpixel((x, y)) for y in ys for x in xs]
        # Empty playfield is near-black; OUT OF FOCUS banner lights most samples.
        bright = sum(1 for v in vals if v > 120)
        return bright >= 6

    def get_tetris_board(self):
        """Read filled cells from a 3x3 sample per cell (row 0 = top).

        A cell counts as filled if either:
          - it hue-matches a piece colour at the same sat/val floor already
            validated for next-piece reads (`classify_piece`), or
          - it's a flat grey at real-block brightness (garbage — garbage has
            no hue at all, so it needs its own gate rather than a looser
            catch-all that also matches translucent ghost-piece tint).
        """
        board_image = self.screen_image.crop((
            self.board_top_left[0],
            self.board_top_left[1],
            self.board_bottom_right[0],
            self.board_bottom_right[1]
        )).convert('RGB')
        hsv = np.array(board_image.convert('HSV'), dtype=np.int32)
        img_h, img_w = hsv.shape[:2]
        block_width = board_image.width / NUM_COL
        block_height = board_image.height / NUM_ROW

        board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
        for row in range(NUM_ROW):
            cy = math.floor(row * block_height + block_height / 2)
            y0, y1 = max(0, cy - 1), min(img_h, cy + 2)
            for col in range(NUM_COL):
                cx = math.floor(col * block_width + block_width / 2)
                x0, x1 = max(0, cx - 1), min(img_w, cx + 2)
                patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
                filled = cell_filled(patch[:, 0], patch[:, 1], patch[:, 2])
                board[row][col] = 1 if filled else 0
        return board

    def place_piece(self, best_position, rotations, need_hold):
        """Place a piece with configurable delays for more human-like inputs."""
        move_delay = lambda: input_gap_sec(self.move_delay_ms, self.delay_variance_percent)
        action_delay = lambda: input_gap_sec(self.action_delay_ms, self.delay_variance_percent)

        if need_hold:
            tap_key(self.keys["hold"])
            time.sleep(action_delay())
        if rotations[0] != 0:
            match rotations[0]:
                case 1:
                    key = self.keys["rotate_cw"]
                case 2:
                    key = self.keys["rotate_180"]
                case 3:
                    key = self.keys["rotate_ccw"]
                case _:
                    raise NotImplementedError
            tap_key(key)
            time.sleep(action_delay())

        if best_position < SPAWN_COLUMN:
            for _ in range(SPAWN_COLUMN - best_position):
                tap_key(self.keys["move_left"])
                time.sleep(move_delay())
        elif best_position > SPAWN_COLUMN:
            for _ in range(best_position - SPAWN_COLUMN):
                tap_key(self.keys["move_right"])
                time.sleep(move_delay())
        if len(rotations) > 1:
            hold_key(self.keys["soft_drop"])
            time.sleep(soft_drop_delay)
            for rot in rotations[1:]:
                match rot:
                    case 1:
                        key = self.keys["rotate_cw"]
                    case 3:
                        key = self.keys["rotate_ccw"]
                    case 11:
                        key = self.keys["move_left"]
                    case 12:
                        key = self.keys["move_right"]
                    case _:
                        raise NotImplementedError
                tap_key(key)
                time.sleep(move_delay())
            release_key(self.keys["soft_drop"])
        tap_key(self.keys["hard_drop"])
        time.sleep(action_delay())

    def _emit_ghost(self, cells, label=""):
        cb = self.publish_ghost
        if cb:
            try:
                cb(cells, label)
            except Exception:
                pass

    def _clear_ghost(self):
        self._emit_ghost([], "")

    def _read_stable_board(self):
        """Re-read until two consecutive captures match (survives line-clear anim)."""
        prev = None
        for _ in range(BOARD_STABLE_TRIES):
            self.refresh_screen_image()
            board = self.get_tetris_board()
            if prev is not None and np.array_equal(board, prev):
                return board
            prev = board
            time.sleep(BOARD_STABLE_GAP)
        return prev if prev is not None else self.get_tetris_board()

    def _path_for_last_cc(self):
        """BFS inputs (incl. soft-drop tucks) to last CC placement, or None."""
        try:
            from cold_clear import get_shared_engine
            from spin_path import path_for_placement
            eng = get_shared_engine(self.cc_binary, self.cc_weights)
            placement = getattr(eng, "last_placement", None)
            if placement is None or self._cc_ai_board is None:
                return None
            return path_for_placement(
                self._cc_ai_board, placement, self.spin_ruleset
            )
        except Exception as e:
            print(f"  path lookup failed: {e}", flush=True)
            return None

    def _verify_suggest_placement(self, expected_board):
        """After a manual drop in suggest mode: does vision match AI expected?

        Returns True if boards match (caller should confirm CC pending).
        Prints a diff when they don't — this is the main vision-vs-intent check.
        """
        time.sleep(LOCK_SETTLE_SEC)
        actual = self._read_stable_board()
        actual_m = actual.copy()
        actual_m[0:SPAWN_MASK_ROWS, :] = 0
        exp_m = expected_board.copy()
        exp_m[0:SPAWN_MASK_ROWS, :] = 0
        # Line-clear animations need longer; if cells still disagree, wait once more.
        if int(np.sum(actual_m != exp_m)) > 0:
            time.sleep(CLEAR_SETTLE_SEC)
            actual = self._read_stable_board()
            actual_m = actual.copy()
            actual_m[0:SPAWN_MASK_ROWS, :] = 0
        mismatch = int(np.sum(actual_m != exp_m))
        if mismatch == 0:
            print("  SUGGEST OK: vision board matches expected placement", flush=True)
            return True
        # Diff in AI coords (row 0 = bottom) for the same language as CC logs
        actual_ai = np.flipud(actual_m)
        exp_ai = np.flipud(exp_m)
        try:
            from cold_clear import _board_diff_summary
            detail = _board_diff_summary(actual_ai, exp_ai)
        except Exception:
            detail = f"{mismatch} cells"
        print(f"  SUGGEST MISMATCH: {mismatch} cells — {detail}", flush=True)
        print("  EXPECTED (after suggested drop):", flush=True)
        print(format_board_full(exp_m), flush=True)
        print("  ACTUAL (vision, spawn masked):", flush=True)
        print(format_board_full(actual_m), flush=True)
        print(
            "  → If you placed on the cyan ghost and still mismatch: vision/calib.\n"
            "  → If you placed elsewhere: ignore (retry on ghost).\n"
            "  → If ghost looked wrong vs game: ghost math or board TL/BR calib.\n"
            "  → After line clears: wait for animation; bot now re-reads until stable.",
            flush=True,
        )
        return False

    def _ghost_cells_legacy(self, ai_board, piece, position_col, rotations):
        """Landing cells as (col, screen_row) with screen_row 0 = top."""
        rot = rotations[0] if rotations else 0
        shapes = tetris_pieces_trimmed[piece]
        if piece in "SZI" and rot == 3:
            shape, terrain = shapes[1]
        else:
            idx = rot if rot < len(shapes) else 0
            shape, terrain = shapes[idx]
        board_terrain = _get_board_terrain(ai_board)
        x = int(position_col)
        if x < 0 or x + len(terrain) > NUM_COL:
            return []
        row = max(board_terrain[x + i] - ht for i, ht in enumerate(terrain))
        cells = []
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    by = row + r
                    bx = x + c
                    if 0 <= by < NUM_ROW and 0 <= bx < NUM_COL:
                        cells.append((bx, NUM_ROW - 1 - by))
        return cells

    def _ghost_cells_cc(self):
        try:
            from cold_clear import get_shared_engine, placement_minos
            eng = get_shared_engine(self.cc_binary, self.cc_weights)
            placement = getattr(eng, "last_placement", None)
            if not placement:
                return []
            cells = []
            for x, y in placement_minos(placement):
                if 0 <= y < NUM_ROW and 0 <= x < NUM_COL:
                    cells.append((x, NUM_ROW - 1 - y))
            return cells
        except Exception:
            return []

    def place_cc(self, center_x, rot_index, need_hold):
        """Execute Cold Clear placement.

        Always prefers SRS BFS path (soft-drop tucks + slides). Greedy
        rotate→shift→hard-drop cannot reach under overhangs — CC assumes that.
        """
        from spin_path import (
            CW, CCW, ROT180, LEFT, RIGHT, SD, HD,
        )

        move_delay = lambda: input_gap_sec(self.move_delay_ms, self.delay_variance_percent)
        action_delay = lambda: input_gap_sec(self.action_delay_ms, self.delay_variance_percent)

        placement = None
        try:
            from cold_clear import get_shared_engine
            placement = getattr(
                get_shared_engine(self.cc_binary, self.cc_weights),
                "last_placement",
                None,
            )
        except Exception:
            placement = None

        if need_hold:
            tap_key(self.keys["hold"])
            if not self._sleep(action_delay()):
                return

        spin = (placement or {}).get("spin", "none") or "none"
        path = self._path_for_last_cc()

        key_map = {
            LEFT: self.keys["move_left"],
            RIGHT: self.keys["move_right"],
            CW: self.keys["rotate_cw"],
            CCW: self.keys["rotate_ccw"],
            ROT180: self.keys["rotate_180"],
            HD: self.keys["hard_drop"],
        }

        def do_rotate(idx):
            if self._stop:
                return
            if idx == 1:
                tap_key(self.keys["rotate_cw"])
            elif idx == 2:
                tap_key(self.keys["rotate_180"])
            elif idx == 3:
                tap_key(self.keys["rotate_ccw"])
            else:
                return
            self._sleep(action_delay())

        def do_shift(cx):
            dx = int(cx) - SPAWN_CENTER_X
            if dx < 0:
                for _ in range(-dx):
                    if self._stop:
                        return
                    tap_key(self.keys["move_left"])
                    if not self._sleep(move_delay()):
                        return
            elif dx > 0:
                for _ in range(dx):
                    if self._stop:
                        return
                    tap_key(self.keys["move_right"])
                    if not self._sleep(move_delay()):
                        return

        if path:
            has_sd = SD in path
            print(
                f"  path{' (soft-drop tuck)' if has_sd else ''}: {' '.join(path)}",
                flush=True,
            )
            for action in path:
                if self._stop:
                    try:
                        release_key(self.keys["soft_drop"])
                    except Exception:
                        pass
                    return
                if action == SD:
                    # Sonic soft-drop (SDF max): hold briefly to floor/rest
                    hold_key(self.keys["soft_drop"])
                    if not self._sleep(soft_drop_delay):
                        release_key(self.keys["soft_drop"])
                        return
                    release_key(self.keys["soft_drop"])
                    if not self._sleep(move_delay()):
                        return
                elif action in key_map:
                    tap_key(key_map[action])
                    delay = action_delay() if action in (CW, CCW, ROT180, HD) else move_delay()
                    if not self._sleep(delay):
                        return
            return

        # Greedy fallback — only flat tops; wrong for tucks under walls
        if self._stop:
            return
        print(
            f"  greedy FALLBACK (no path): rot={rot_index} x={center_x} spin={spin}",
            flush=True,
        )
        do_rotate(rot_index)
        do_shift(center_x)
        if self._stop:
            return
        tap_key(self.keys["hard_drop"])
        self._sleep(action_delay())

    def run(self):
        combo = 0
        b2b = 0

        if self.ai_engine == "cold_clear":
            from cold_clear import reset_log, LOG_PATH
            reset_log()
            print(f"Cold Clear diagnostics will be written to {LOG_PATH}", flush=True)

        print("TetrioBot started. Waiting for game...", flush=True)
        print(f"  engine={self.ai_engine}  spin_ruleset={getattr(self, 'spin_ruleset', '?')}"
              f"  cc_think_ms={self.cc_think_ms}  play_mode={self.play_mode}"
              f"  trust_board={self.trust_expected_board}",
              flush=True)
        print(
            f"  input={'pydirectinput' if _USE_DIRECT_INPUT else 'pyautogui (install pydirectinput)'}",
            flush=True,
        )
        self.refresh_screen_image()
        last_next_pieces = self.read_next_pieces()
        print(f"Initial next pieces detected: {last_next_pieces}", flush=True)
        if last_next_pieces is None or all(p is None for p in last_next_pieces):
            means = []
            for xy in self.next_piece_xy:
                rgb = np.array(self.sample_box(xy)).reshape(-1, 3).mean(axis=0)
                means.append(float(rgb.mean()))
            if means and max(means) < 15:
                print(
                    "WARNING: next-piece samples are nearly black. "
                    "TETR.IO must be visible on the calibrated monitor. "
                    "If the overlay looks right but capture is black, you need "
                    "bettercam (pip install bettercam opencv-python-headless) "
                    "for discrete-GPU displays.",
                    flush=True,
                )
            else:
                print(
                    "WARNING: could not classify next queue — re-check next-piece "
                    "calibration (green boxes must sit on the coloured minos).",
                    flush=True,
                )
        expected_board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
        poll_count = 0
        unreadable = 0
        while not self._stop:
            next_pieces = self.read_next_pieces()
            while next_pieces == last_next_pieces and not self._stop:
                poll_count += 1
                unreadable_poll = next_pieces is None or all(
                    p is None for p in next_pieces
                )
                if unreadable_poll and poll_count >= MAX_UNREADABLE_POLLS:
                    print("Queue unreadable while waiting — retrying capture",
                          flush=True)
                    break
                if poll_count % 100 == 0:
                    print(f"Polling for piece change... ({poll_count} iterations, "
                          f"current: {next_pieces})", flush=True)
                if not self._sleep(wait_time):
                    break
                next_pieces = self.read_next_pieces()
            if self._stop:
                break
            poll_count = 0
            self.hold_used_this_piece = False
            self.last_good_next = None  # don't merge with stale cache after shift

            prev_queue = last_next_pieces
            queue_piece = last_next_pieces[0] if last_next_pieces else None
            last_next_pieces = next_pieces

            if (queue_piece is None or next_pieces is None or
                    any(p is None for p in next_pieces)):
                unreadable += 1
                if unreadable % 10 == 1:
                    print(f"Cannot read next queue {next_pieces} — retrying",
                          flush=True)
                time.sleep(0.05)
                continue
            unreadable = 0
            # Falling piece = old next[0] before the queue shifted (queue inference
            # is reliable). Board color read fails on custom backgrounds — the
            # orange sky reads as L every time — so we don't use it.
            falling_piece = queue_piece
            current_piece = falling_piece

            # Happy path (opt-in): last placement confirmed → reuse calculated stack.
            # One-frame peek still catches garbage/desync without spawn-settle.
            use_trust = self.trust_expected_board and self._board_trusted
            if use_trust:
                self.refresh_screen_image()
                peek = self.get_tetris_board()
                peek_m = peek.copy()
                peek_m[0:SPAWN_MASK_ROWS, :] = 0
                exp_m = expected_board.copy()
                exp_m[0:SPAWN_MASK_ROWS, :] = 0
                if np.array_equal(peek_m, exp_m):
                    current_board = exp_m
                    raw_board = expected_board
                else:
                    print(
                        "  board peek mismatch (garbage/desync?) — vision resync",
                        flush=True,
                    )
                    self._board_trusted = False
                    use_trust = False
            if not use_trust:
                time.sleep(SPAWN_SETTLE_SEC)
                if self._stop:
                    break
                # Stable read: line-clear / lock flash mid-frame corrupts CC expected.
                raw_board = self._read_stable_board()
                if self.board_looks_obscured():
                    print(
                        "Board looks obscured (window unfocused / covered?) — waiting.",
                        flush=True,
                    )
                    time.sleep(0.25)
                    self.refresh_screen_image()
                    # Restore pre-shift queue so we retry this piece instead of
                    # consuming the queue change and never emitting a ghost.
                    last_next_pieces = prev_queue
                    self._board_trusted = False
                    continue
                # The falling piece sits in the top rows — including it makes the AI
                # think the stack is taller/wrong shape and pick bad placements.
                current_board = raw_board.copy()
                current_board[0:SPAWN_MASK_ROWS, :] = 0
                if not np.all(np.equal(raw_board, expected_board)):
                    expected_board = raw_board.copy()

            held_piece = self.read_held_piece()
            # Empty hold at game start is normal — CC treats hold=null correctly.
            # Do NOT auto-press hold here; that skipped the first real cycles.
            if held_piece is None and self.last_held_piece is not None:
                # Hold was used this cycle; slot looks empty but AI still
                # needs a label — use current piece (hold == current → no swap).
                held_piece = current_piece

            t1 = time.time()
            # Screen scan uses row 0 = top; AI engines use row 0 = bottom.
            ai_board = np.flipud(current_board)
            if self.ai_engine == "cold_clear":
                from cold_clear import find_best_move as find_best_move_cc
                self._cc_ai_board = np.array(ai_board, dtype=np.int32).copy()
                score, (position, rotations, need_hold, combo, b2b, ai_expected) = find_best_move_cc(
                    ai_board, current_piece, next_pieces, held_piece, combo, b2b,
                    self.pruning_moves,
                    self.pruning_breadth,
                    mp_pool=self.mp_pool,
                    binary_path=self.cc_binary,
                    config_path=self.cc_weights,
                    spin_ruleset=self.spin_ruleset,
                    think_time_sec=(self.cc_think_ms or 0) / 1000.0,
                )
            else:
                score, (position, rotations, need_hold, combo, b2b, ai_expected) = find_best_move_legacy(
                    ai_board, current_piece, next_pieces, held_piece, combo, b2b,
                    self.pruning_moves,
                    self.pruning_breadth,
                    mp_pool=self.mp_pool,
                )
            expected_board = np.flipud(ai_expected)
            t2 = time.time()

            print(f"nodes: {round(score):6}   b2b: {b2b:2}    time: {t2-t1:.3f}"
                  f"   [{self.ai_engine}]  piece={falling_piece} cells={int(current_board.sum())}",
                  flush=True)
            if self.ai_engine == "cold_clear":
                try:
                    from cold_clear import get_shared_engine
                    pl = getattr(
                        get_shared_engine(self.cc_binary, self.cc_weights),
                        "last_placement",
                        None,
                    ) or {}
                    loc = pl.get("location") or {}
                    print(
                        f"  CC place: {loc.get('type')} {loc.get('orientation')} "
                        f"@x={loc.get('x')} y={loc.get('y')} "
                        f"spin={pl.get('spin','none')} hold={need_hold}",
                        flush=True,
                    )
                except Exception:
                    pass
            if t2 - t1 < wait_time:
                time.sleep(wait_time - t2 + t1)

            if score < -50000:
                expected_board = raw_board.copy()
                self._board_trusted = False
                continue

            place_piece = falling_piece
            if need_hold:
                place_piece = next_pieces[0] if held_piece is None else held_piece

            if self.ai_engine == "cold_clear":
                target_col = position  # SRS center x for debug display
                offset = 0
            else:
                if place_piece in "SZI" and rotations[0] == 3:
                    best_piece_pos_rot = tetris_pieces[place_piece][1]
                else:
                    best_piece_pos_rot = tetris_pieces[place_piece][rotations[0]]
                offset = 0
                for i in range(best_piece_pos_rot.shape[1]):
                    if not any(best_piece_pos_rot[:, i]):
                        offset += 1
                    else:
                        break
                target_col = position - offset
            if self.debug:
                self.move_count += 1
                print(f"\n{'='*44}", flush=True)
                print(f"  MOVE {self.move_count}", flush=True)
                print(f"{'='*44}", flush=True)
                print(f"  FALLING piece : {falling_piece}", flush=True)
                print(f"  NEXT preview  : {next_pieces}", flush=True)
                print(f"  HELD          : {held_piece}", flush=True)
                if self.ai_engine == "cold_clear":
                    print(f"  TARGET center : {position}  (spawn center={SPAWN_CENTER_X})",
                          flush=True)
                else:
                    print(f"  TARGET column : {target_col}  (0=left, 9=right; ^ below)",
                          flush=True)
                print(f"  ROTATION      : {rotations}", flush=True)
                print(f"  HOLD this move: {need_hold}", flush=True)
                print(f"  AI score      : {round(score)}", flush=True)
                print(f"\n  BOARD the AI sees (#=filled, top row=0, spawn masked):",
                      flush=True)
                mark = target_col if self.ai_engine != "cold_clear" else None
                print(format_board_full(current_board, mark_col=mark),
                      flush=True)
                raw_cells = int(raw_board.sum())
                ai_cells = int(current_board.sum())
                if raw_cells != ai_cells:
                    print(f"\n  RAW board ({raw_cells} cells, includes falling piece in top rows):",
                          flush=True)
                    print(format_board_full(raw_board), flush=True)
                print(f"\n  EXPECTED board after drop:", flush=True)
                print(format_board_full(expected_board), flush=True)
                if self.ai_engine != "cold_clear":
                    print(f"  KEYS: {describe_inputs(self.keys, position, rotations, need_hold, offset)}",
                          flush=True)
            if self.pause_sec > 0:
                print(f"\n  >> dropping in {self.pause_sec:.0f}s — compare board above to screen\n",
                      flush=True)
                time.sleep(self.pause_sec)

            # Ghost overlay (suggest mode always; also useful briefly in autodrop)
            if self.ai_engine == "cold_clear":
                ghost = self._ghost_cells_cc()
                path = self._path_for_last_cc()
                from spin_path import SD as _SD
                if path is None:
                    reach = " NO-PATH"
                elif _SD in path:
                    reach = " soft-drop"
                else:
                    reach = ""
            else:
                ghost = self._ghost_cells_legacy(
                    ai_board, place_piece, position, rotations
                )
                reach = ""
            mode = self.play_mode
            ghost_label = (
                f"{'HOLD+' if need_hold else ''}{place_piece} [{mode}]{reach}"
            )
            if not ghost:
                print(
                    f"  WARNING: empty ghost for {ghost_label}",
                    flush=True,
                )
            else:
                print(
                    f"  ghost: {len(ghost)} cells {ghost[:4]}{'…' if len(ghost) > 4 else ''}"
                    f"  label={ghost_label}",
                    flush=True,
                )
                if reach == " NO-PATH":
                    print(
                        "  WARNING: CC placement has no soft-drop path from spawn "
                        "(kick tables / board desync). Ghost may look impossible.",
                        flush=True,
                    )
                elif reach == " soft-drop":
                    print(
                        "  note: needs soft-drop tuck/slide — hard-drop alone won't reach",
                        flush=True,
                    )
            self._emit_ghost(ghost, ghost_label)

            placed = False
            decision_queue = list(next_pieces) if next_pieces is not None else None
            while not self._stop:
                mode = self.play_mode
                if mode == "autodrop":
                    if self.ai_engine == "cold_clear":
                        self.place_cc(
                            position, rotations[0] if rotations else 0, need_hold
                        )
                    else:
                        self.place_piece(target_col, rotations, need_hold)
                    placed = True
                    break
                # suggest: wait for manual drop (queue change) or switch to autodrop
                if not self._sleep(0.05):
                    break
                cur = self.read_next_pieces()
                if decision_queue is not None and cur != decision_queue:
                    # User placed. Do NOT set last_next_pieces = cur — that skipped
                    # the next piece (outer loop waited for yet another queue shift).
                    # Leave last_next_pieces as decision_queue so the next cycle
                    # sees the shift immediately and uses decision_queue[0] as falling.
                    self._clear_ghost()  # before vision read — ghost would pollute capture
                    time.sleep(0.02)
                    placed = self._verify_suggest_placement(expected_board)
                    if not placed and self.ai_engine == "cold_clear":
                        from cold_clear import cancel_pending_placement
                        cancel_pending_placement(self.cc_binary, self.cc_weights)
                        self._board_trusted = False
                    break

            if not placed:
                if self._stop:
                    break
                continue

            if self.ai_engine == "cold_clear":
                from cold_clear import confirm_pending_placement
                confirm_pending_placement(self.cc_binary, self.cc_weights)

            # Next piece can use calculated stack — no vision re-sync needed
            # unless garbage arrives / user misses (then mismatch clears the flag).
            self._board_trusted = True
            self._clear_ghost()
            if need_hold:
                self.hold_used_this_piece = True
                self.last_held_piece = falling_piece
            # Short settle only for autodrop key timing / clear anim before next
            # queue poll — not for re-reading the board when trusted.
            settle = LOCK_SETTLE_SEC
            if self.play_mode == "autodrop":
                settle += get_delay_with_variance(
                    self.action_delay_ms, self.delay_variance_percent
                )
            if int(expected_board.sum()) < int(current_board.sum()):
                settle += CLEAR_SETTLE_SEC
            time.sleep(settle)
        print("TetrioBot stopped.", flush=True)
        self._clear_ghost()

    def close(self):
        """Release the multiprocessing pool / CC engine if any."""
        if self.mp_pool is not None:
            try:
                self.mp_pool.terminate()
                self.mp_pool.join()
            except Exception:
                try:
                    self.mp_pool.close()
                    self.mp_pool.join()
                except Exception:
                    pass
            self.mp_pool = None
        try:
            from cold_clear import shutdown_shared_engine
            shutdown_shared_engine()
        except Exception:
            pass


def bot_from_config(config, debug=False, pause_sec=0):
    """Build a TetrioBot from a profile/config dict."""
    return TetrioBot(
        screen_offset=tuple(config["screen_offset"]),
        screen_resolution=tuple(config["screen_resolution"]),
        board_top_left=tuple(config["board_top_left"]),
        board_bottom_right=tuple(config["board_bottom_right"]),
        next_piece_xy_0=tuple(config["next_piece_xy_0"]),
        next_piece_xy_4=tuple(config["next_piece_xy_4"]),
        held_piece_xy=tuple(config["held_piece_xy"]),
        pruning_moves=config.get("pruning_moves", 5),
        pruning_breadth=config.get("pruning_breadth", 5),
        mp=config.get("mp", 16),
        move_delay_ms=config.get("move_delay_ms", DEFAULT_MOVE_DELAY_MS),
        action_delay_ms=config.get("action_delay_ms", DEFAULT_ACTION_DELAY_MS),
        delay_variance_percent=config.get(
            "delay_variance_percent", DEFAULT_DELAY_VARIANCE_PERCENT
        ),
        keybinds=config.get("keybinds"),
        debug=debug,
        pause_sec=pause_sec,
        ai_engine=config.get("ai_engine", "legacy"),
        cc_binary=config.get("cc_binary"),
        cc_weights=config.get("cc_weights"),
        play_mode=config.get("play_mode", "autodrop"),
        spin_ruleset=config.get("spin_ruleset", "all_mini_plus"),
        cc_think_ms=config.get("cc_think_ms", 50),
        trust_expected_board=config.get("trust_expected_board", True),
    )


def _self_check():
    """Board-cell classification: piece colours, garbage, empty, ghost tint."""
    for name, rgb in zip(colors_name, colors):
        h, s, v = colorsys.rgb_to_hsv(*[c / 255 for c in rgb])
        px = np.array([h, s, v]) * 255
        assert cell_filled([px[0]] * 9, [px[1]] * 9, [px[2]] * 9), \
            f"{name} piece colour should read as filled"

    assert not cell_filled([0] * 9, [0] * 9, [10] * 9), \
        "near-black empty board should read as empty"

    assert cell_filled([0] * 9, [5] * 9, [130] * 9), \
        "flat mid-grey garbage should read as filled"

    # Ghost piece: real colour hue, but faded well below real-block sat/val —
    # must NOT be counted, or the AI thinks a spawn preview is a locked block.
    l_h, l_s, l_v = colorsys.rgb_to_hsv(*[c / 255 for c in colors[colors_name.index("L")]])
    ghost = np.array([l_h * 255, l_s * 255 * 0.25, l_v * 255 * 0.35])
    assert not cell_filled([ghost[0]] * 9, [ghost[1]] * 9, [ghost[2]] * 9), \
        "faded ghost-piece tint should not read as filled"

    print("bot self-check ok")



def main():
    ensure_dpi_awareness()
    parser = argparse.ArgumentParser(description='TetrioBot - An AI player for TETR.IO')
    parser.add_argument('--self-check', action='store_true',
                        help='Run offline vision self-check (no game needed) and exit')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run the calibration wizard to configure screen coordinates')
    parser.add_argument('--use-config', action='store_true',
                        help='Use saved configuration from config.json')
    parser.add_argument('--config', default=None, metavar='FILE',
                        help='Config file to load/save (e.g. config.old.json). '
                             'Same as --use-config when FILE is config.json.')
    parser.add_argument('--pruning-moves', type=int, default=5,
                        help='Number of moves for pruning (default: 5)')
    parser.add_argument('--pruning-breadth', type=int, default=5,
                        help='Breadth for pruning (default: 5)')
    parser.add_argument('--mp', type=int, default=16,
                        help='Number of multiprocessing workers (default: 16)')
    parser.add_argument('--delay', type=int, default=None,
                        help='Override move delay in milliseconds (e.g., --delay 50)')
    parser.add_argument('--action-delay', type=int, default=None,
                        help='Override action delay in milliseconds')
    parser.add_argument('--delay-variance', type=int, default=None,
                        help='Override delay variance percentage')
    parser.add_argument('--test', action='store_true',
                        help='Print what the bot detects (next/held/board) without playing')
    parser.add_argument('--keys', action='store_true',
                        help='Re-set keybinds in config.json without redoing coordinates')
    parser.add_argument('--snapshot', action='store_true',
                        help='Save snapshot.png showing every region the bot samples')
    parser.add_argument('--debug', action='store_true',
                        help='Print full board + plan each move (auto --pause 5)')
    parser.add_argument('--pause', type=float, default=None, metavar='SEC',
                        help='Seconds before each drop (default: 5 with --debug, else 0)')
    parser.add_argument('--ai-engine', choices=['legacy', 'cold_clear'], default=None,
                        help='Heuristics engine: legacy (yilinho) or cold_clear')
    parser.add_argument(
        '--spin-ruleset',
        choices=['t_spins', 'all_mini', 'all_mini_plus', 'all_spin', 'none'],
        default=None,
        help='TETR.IO spin detection preset (default: all_mini_plus)',
    )
    parser.add_argument('--countdown', type=int, default=STARTUP_COUNTDOWN_SEC,
                        help=f'Seconds to wait before starting so you can switch windows (default: {STARTUP_COUNTDOWN_SEC})')

    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return

    if args.pause is None:
        args.pause = DEFAULT_PAUSE_SEC if args.debug else 0

    if args.calibrate:
        run_calibration_wizard(args.config or CONFIG_FILE)
        return

    config_path = args.config or (CONFIG_FILE if args.use_config else None)

    if args.keys:
        path = args.config or CONFIG_FILE
        config = load_config(path)
        if config is None:
            print(f"Error: No config at {path}. Run with --calibrate first.")
            return
        config['keybinds'] = prompt_keybinds()
        save_config(config, path)
        return
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
        if config is None:
            print(f"Error: No config at {config_path}. Run with --calibrate first.")
            return
        
        # Apply CLI overrides for delay settings
        move_delay_ms = args.delay if args.delay is not None else config.get('move_delay_ms', DEFAULT_MOVE_DELAY_MS)
        action_delay_ms = args.action_delay if args.action_delay is not None else config.get('action_delay_ms', DEFAULT_ACTION_DELAY_MS)
        delay_variance_percent = args.delay_variance if args.delay_variance is not None else config.get('delay_variance_percent', DEFAULT_DELAY_VARIANCE_PERCENT)
        
        keybinds = config.get('keybinds')

        if config.get('label'):
            print(f"Profile: {config['label']}")
        print(f"Loaded configuration from {config_path}")
        print(f"Delay settings: move={move_delay_ms}ms, action={action_delay_ms}ms, variance={delay_variance_percent}%")
        print(f"Keybinds: {keybinds or DEFAULT_KEYBINDS}")
        bot = TetrioBot(
            screen_offset=tuple(config['screen_offset']),
            screen_resolution=tuple(config['screen_resolution']),
            board_top_left=tuple(config['board_top_left']),
            board_bottom_right=tuple(config['board_bottom_right']),
            next_piece_xy_0=tuple(config['next_piece_xy_0']),
            next_piece_xy_4=tuple(config['next_piece_xy_4']),
            held_piece_xy=tuple(config['held_piece_xy']),
            pruning_moves=args.pruning_moves,
            pruning_breadth=args.pruning_breadth,
            mp=args.mp,
            move_delay_ms=move_delay_ms,
            action_delay_ms=action_delay_ms,
            delay_variance_percent=delay_variance_percent,
            keybinds=keybinds,
            debug=args.debug,
            pause_sec=args.pause,
            ai_engine=args.ai_engine or config.get('ai_engine', 'legacy'),
            cc_binary=config.get('cc_binary'),
            cc_weights=config.get('cc_weights'),
            spin_ruleset=args.spin_ruleset or config.get('spin_ruleset', 'all_mini_plus'),
            cc_think_ms=config.get('cc_think_ms', 50),
        )
    else:
        # Use default/hardcoded values
        # Note: These values are based on a secondary-screen which has a TETR.IO window title bar(22px) but no windows-taskbar.
        #       If you have only 1 monitor, you may hide your windows-taskbar or measure the values for your own setting.
        
        # Apply CLI overrides for delay settings or use defaults
        move_delay_ms = args.delay if args.delay is not None else DEFAULT_MOVE_DELAY_MS
        action_delay_ms = args.action_delay if args.action_delay is not None else DEFAULT_ACTION_DELAY_MS
        delay_variance_percent = args.delay_variance if args.delay_variance is not None else DEFAULT_DELAY_VARIANCE_PERCENT
        
        print(f"Delay settings: move={move_delay_ms}ms, action={action_delay_ms}ms, variance={delay_variance_percent}%")
        bot = TetrioBot(
            # screen_offset=(0, 0),  # most common case
            screen_offset=(-1920, 0),
            screen_resolution=(1920, 1080),
            board_top_left=(787, 220),
            board_bottom_right=(1133, 899),
            next_piece_xy_0=(1260, 300),
            next_piece_xy_4=(1260, 721),
            held_piece_xy=(691, 300),
            pruning_moves=args.pruning_moves,
            pruning_breadth=args.pruning_breadth,
            mp=args.mp,
            move_delay_ms=move_delay_ms,
            action_delay_ms=action_delay_ms,
            delay_variance_percent=delay_variance_percent,
            debug=args.debug,
            pause_sec=args.pause,
            ai_engine=args.ai_engine or 'legacy',
            spin_ruleset=args.spin_ruleset or 'all_mini_plus',
            cc_think_ms=50,
        )

    if args.pause > 0:
        print(f"Pause: {args.pause}s before each drop — compare printed board to screen.",
              flush=True)

    if move_delay_ms < 25 or action_delay_ms < 25:
        print("Warning: delays under 25ms often cause missed keypresses in "
              "TETR.IO. Try --delay 40 --action-delay 50 if pieces land wrong.",
              flush=True)

    if args.snapshot:
        startup_countdown(args.countdown)
        save_snapshot(bot)
        return

    if args.test:
        startup_countdown(args.countdown)
        run_detection_test(bot)
        return

    startup_countdown(args.countdown)
    bot.run()


if __name__ == "__main__":
    main()
