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

from constants import colors, colors_name, tetris_pieces, NUM_ROW, NUM_COL
from tetris_ai import find_best_move

CONFIG_FILE = "config.json"

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
# Greyed-out held piece after a hold — lower sat/val but hue still valid
HELD_MIN_SAT = 35
HELD_MIN_VAL = 30
HELD_MIN_PIXELS = 4
MAX_EMPTY_HOLD_RETRIES = 5
QUEUE_READ_RETRIES = 6
QUEUE_READ_DELAY = 0.025
MAX_UNREADABLE_POLLS = 40
SPAWN_MASK_ROWS = 4       # top rows — active piece lives here, hide from AI
LOCK_SETTLE_SEC = 0.08    # wait after hard drop before next read
SPAWN_COLUMN = 3          # default piece spawn column for movement math
SPAWN_SETTLE_SEC = 0.12   # wait after queue shift before reading pieces
DEFAULT_PAUSE_SEC = 3.0   # --pause: seconds to wait before each drop

# Default delay settings (in milliseconds)
DEFAULT_MOVE_DELAY_MS = 30
DEFAULT_ACTION_DELAY_MS = 50
DEFAULT_DELAY_VARIANCE_PERCENT = 20
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


def tap_key(key):
    """Press and release a key. Works on macOS without root (needs Accessibility)."""
    pyautogui.keyDown(key)
    pyautogui.keyUp(key)


def capture_screen(screen_offset, screen_resolution):
    """Grab a screen region. Coords match pyautogui / calibration (logical pixels).

    If mss returns a HiDPI (Retina) bitmap larger than the requested region,
    resize down so crop coordinates still line up.
    """
    left, top = screen_offset
    width, height = screen_resolution
    with mss.MSS() as sct:
        region = {
            "left": int(left),
            "top": int(top),
            "width": max(1, int(width)),
            "height": max(1, int(height)),
        }
        shot = sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        return img


def monitor_containing(x, y):
    """Return the mss monitor dict that contains point (x, y)."""
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


def save_config(config, config_path=CONFIG_FILE):
    """Save configuration to JSON."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
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


wait_time = 0.03
soft_drop_delay = 0.1
# Game Settings - DAS 40ms, ARR 0ms, SDF max, lowest graphic


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
    ):
        self.keys = dict(DEFAULT_KEYBINDS)
        if keybinds:
            self.keys.update(keybinds)
        self.debug = debug or pause_sec > 0
        self.pause_sec = pause_sec
        self.move_count = 0
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
        self.pixel_area_half = pixel_area // 2
        self.held_piece_xy = held_piece_xy

        self.pruning_moves = pruning_moves
        self.pruning_breadth = pruning_breadth
        self.mp_pool = Pool(processes=mp) if mp > 1 else None
        
        # Delay settings
        self.move_delay_ms = move_delay_ms
        self.action_delay_ms = action_delay_ms
        self.delay_variance_percent = delay_variance_percent

        self.last_good_next = None
        self.last_held_piece = None
        self.hold_used_this_piece = False

        self.screen_image = Image.new("RGB", self.screen_resolution)
        self.refresh_screen_image()

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

    def get_tetris_board(self):
        board_image = self.screen_image.crop((
            self.board_top_left[0],
            self.board_top_left[1],
            self.board_bottom_right[0],
            self.board_bottom_right[1]
        )).convert('L')
        # board_image.save("board.png")
        board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
        block_width = board_image.width / NUM_COL
        block_height = board_image.height / NUM_ROW

        for row in reversed(range(NUM_ROW)):
            empty_row = True
            for col in range(NUM_COL):
                total_darkness = 0
                num_pixels = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        x = math.floor(col * block_width + block_width / 2) + dx
                        y = math.floor(row * block_height + block_height / 2) + dy
                        pixel_value = board_image.getpixel((x, y))
                        total_darkness += pixel_value
                        num_pixels += 1
                avg_darkness = total_darkness / num_pixels

                if avg_darkness < 30:
                    board[row][col] = 0
                else:
                    empty_row = False
                    board[row][col] = 1
            if empty_row:
                break
        return board

    def place_piece(self, best_position, rotations, need_hold):
        """Place a piece with configurable delays for more human-like inputs."""
        move_delay = lambda: get_delay_with_variance(self.move_delay_ms, self.delay_variance_percent)
        action_delay = lambda: get_delay_with_variance(self.action_delay_ms, self.delay_variance_percent)

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
            pyautogui.keyDown(self.keys["soft_drop"])
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
            pyautogui.keyUp(self.keys["soft_drop"])
        tap_key(self.keys["hard_drop"])
        time.sleep(action_delay())

    def run(self):
        combo = 0
        b2b = 0

        print("TetrioBot started. Waiting for game...", flush=True)
        self.refresh_screen_image()
        last_next_pieces = self.read_next_pieces()
        print(f"Initial next pieces detected: {last_next_pieces}", flush=True)
        expected_board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
        poll_count = 0
        empty_hold_retries = 0
        unreadable = 0
        while True:
            next_pieces = self.read_next_pieces()
            while next_pieces == last_next_pieces:
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
                time.sleep(wait_time)
                next_pieces = self.read_next_pieces()
            poll_count = 0
            self.hold_used_this_piece = False
            self.last_good_next = None  # don't merge with stale cache after shift

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
            time.sleep(SPAWN_SETTLE_SEC)

            # Falling piece = old next[0] before the queue shifted (queue inference
            # is reliable). Board color read fails on custom backgrounds — the
            # orange sky reads as L every time — so we don't use it.
            falling_piece = queue_piece
            current_piece = falling_piece

            raw_board = self.get_tetris_board()
            # The falling piece sits in the top rows — including it makes the AI
            # think the stack is taller/wrong shape and pick bad placements.
            current_board = raw_board.copy()
            current_board[0:SPAWN_MASK_ROWS, :] = 0
            if not np.all(np.equal(raw_board, expected_board)):
                expected_board = raw_board.copy()

            held_piece = self.read_held_piece()
            if held_piece is None:
                if self.last_held_piece is None:
                    empty_hold_retries += 1
                    print(f"Hold empty — pressing hold to fill slot "
                          f"(attempt {empty_hold_retries})", flush=True)
                    if empty_hold_retries >= MAX_EMPTY_HOLD_RETRIES:
                        print("Giving up on hold detection. Check held_piece_xy "
                              "with --test; aim at the coloured blocks.", flush=True)
                        return
                    tap_key(self.keys["hold"])
                    self.hold_used_this_piece = True
                    time.sleep(get_delay_with_variance(
                        self.action_delay_ms, self.delay_variance_percent) + 0.15)
                    last_next_pieces = self.read_next_pieces()
                    continue
                # Hold was used this cycle; slot looks empty but AI still
                # needs a label — use current piece (hold == current → no swap).
                held_piece = current_piece
            else:
                empty_hold_retries = 0

            t1 = time.time()
            # Screen scan uses row 0 = top; tetris_ai uses row 0 = bottom.
            ai_board = np.flipud(current_board)
            score, (position, rotations, need_hold, combo, b2b, ai_expected) = find_best_move(
                ai_board, current_piece, next_pieces, held_piece, combo, b2b,
                self.pruning_moves,
                self.pruning_breadth,
                mp_pool=self.mp_pool,
            )
            expected_board = np.flipud(ai_expected)
            t2 = time.time()

            print(f"score: {round(score):6}   b2b: {b2b:2}    time: {t2-t1}",
                  flush=True)
            if t2 - t1 < wait_time:
                time.sleep(wait_time - t2 + t1)

            if score < -50000:
                expected_board = raw_board.copy()
                continue

            place_piece = falling_piece
            if need_hold:
                place_piece = next_pieces[0] if held_piece is None else held_piece

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
                print(f"  TARGET column : {target_col}  (0=left, 9=right; ^ below)",
                      flush=True)
                print(f"  ROTATION      : {rotations}", flush=True)
                print(f"  HOLD this move: {need_hold}", flush=True)
                print(f"  AI score      : {round(score)}", flush=True)
                print(f"\n  BOARD the AI sees (#=filled, top row=0, spawn masked):",
                      flush=True)
                print(format_board_full(current_board, mark_col=target_col),
                      flush=True)
                raw_cells = int(raw_board.sum())
                ai_cells = int(current_board.sum())
                if raw_cells != ai_cells:
                    print(f"\n  RAW board ({raw_cells} cells, includes falling piece in top rows):",
                          flush=True)
                    print(format_board_full(raw_board), flush=True)
                print(f"\n  EXPECTED board after drop:", flush=True)
                print(format_board_full(expected_board), flush=True)
                print(f"  KEYS: {describe_inputs(self.keys, position, rotations, need_hold, offset)}",
                      flush=True)
            if self.pause_sec > 0:
                print(f"\n  >> dropping in {self.pause_sec:.0f}s — compare board above to screen\n",
                      flush=True)
                time.sleep(self.pause_sec)
            self.place_piece(target_col, rotations, need_hold)
            if need_hold:
                self.hold_used_this_piece = True
                self.last_held_piece = falling_piece
            time.sleep(get_delay_with_variance(
                self.action_delay_ms, self.delay_variance_percent) + LOCK_SETTLE_SEC)


def main():
    parser = argparse.ArgumentParser(description='TetrioBot - An AI player for TETR.IO')
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
    parser.add_argument('--countdown', type=int, default=STARTUP_COUNTDOWN_SEC,
                        help=f'Seconds to wait before starting so you can switch windows (default: {STARTUP_COUNTDOWN_SEC})')

    args = parser.parse_args()

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
