"""Cold Clear 2 adapter — TBP subprocess + drop-in find_best_move.

See heuristic-update.md. Speaks Tetris Bot Protocol over stdin/stdout.
All communication is local (subprocess pipes) — no network.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from constants import NUM_COL, NUM_ROW

# Guideline spawn: SRS true-rotation center x (TBP MVP).
SPAWN_CENTER_X = 4

ORIENT_INDEX = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}

# SRS true-rotation mino offsets (dx, dy) from piece center.
# dy positive = up (away from floor), matching TBP y=0 at bottom.
_SRS_MINOS = {
    "T": {
        "north": [(-1, 0), (0, 0), (1, 0), (0, 1)],
        "east": [(0, 1), (0, 0), (0, -1), (1, 0)],
        "south": [(-1, 0), (0, 0), (1, 0), (0, -1)],
        "west": [(0, 1), (0, 0), (0, -1), (-1, 0)],
    },
    "J": {
        "north": [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        "east": [(0, 1), (0, 0), (0, -1), (1, 1)],
        "south": [(-1, 0), (0, 0), (1, 0), (1, -1)],
        "west": [(0, 1), (0, 0), (0, -1), (-1, -1)],
    },
    "L": {
        "north": [(-1, 0), (0, 0), (1, 0), (1, 1)],
        "east": [(0, 1), (0, 0), (0, -1), (1, -1)],
        "south": [(-1, 0), (0, 0), (1, 0), (-1, -1)],
        "west": [(0, 1), (0, 0), (0, -1), (-1, 1)],
    },
    "S": {
        "north": [(-1, 0), (0, 0), (0, 1), (1, 1)],
        "east": [(0, 1), (0, 0), (1, 0), (1, -1)],
        "south": [(-1, -1), (0, -1), (0, 0), (1, 0)],
        "west": [(-1, 1), (-1, 0), (0, 0), (0, -1)],
    },
    "Z": {
        "north": [(-1, 1), (0, 1), (0, 0), (1, 0)],
        "east": [(1, 1), (1, 0), (0, 0), (0, -1)],
        "south": [(-1, 0), (0, 0), (0, -1), (1, -1)],
        "west": [(0, 1), (0, 0), (-1, 0), (-1, -1)],
    },
    "I": {
        "north": [(-1, 0), (0, 0), (1, 0), (2, 0)],
        "east": [(0, 1), (0, 0), (0, -1), (0, -2)],
        "south": [(-2, 0), (-1, 0), (0, 0), (1, 0)],
        "west": [(0, 2), (0, 1), (0, 0), (0, -1)],
    },
    "O": {
        "north": [(0, 0), (1, 0), (0, 1), (1, 1)],
        "east": [(0, 0), (1, 0), (0, -1), (1, -1)],
        "south": [(-1, 0), (0, 0), (-1, -1), (0, -1)],
        "west": [(-1, 0), (0, 0), (-1, 1), (0, 1)],
    },
}

DEFAULT_BINARY_CANDIDATES = [
    os.path.join("cold-clear-2", "target", "release", "cold-clear-2.exe"),
    os.path.join("cold-clear-2", "target", "release", "cold-clear-2"),
    "cold-clear-2.exe",
    "cold-clear-2",
]

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_REPO_DIR, "cc_weights.json")
DEFAULT_THINK_TIME_SEC = 0.15
BOARD_MISMATCH_CELLS = 0  # any vision mismatch → full TBP restart (avoid bad tree)

_engine_lock = threading.Lock()
_shared_engine: Optional["ColdClear2Engine"] = None
_shared_engine_key = None

LOG_PATH = os.path.join(_REPO_DIR, "ai_debug.log")


def _resolve_weights_path(config_path: Optional[str]) -> str:
    path = config_path or DEFAULT_WEIGHTS
    if path and not os.path.isabs(path):
        path = os.path.join(_REPO_DIR, path)
    return path


def reset_log():
    try:
        open(LOG_PATH, "w", encoding="utf-8").close()
    except OSError:
        pass


def log_line(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def resolve_binary(path: Optional[str] = None) -> str:
    if path and os.path.isfile(path):
        return path
    env = os.environ.get("COLD_CLEAR_BINARY")
    if env and os.path.isfile(env):
        return env
    for cand in DEFAULT_BINARY_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "Cold Clear 2 binary not found. Build it with:\n"
        "  git clone https://github.com/MinusKelvin/cold-clear-2.git\n"
        "  cd cold-clear-2 && cargo build --release\n"
        "Or set COLD_CLEAR_BINARY / profile field cc_binary."
    )


def to_tbp_board(board_20_bottom_up) -> list:
    """Convert AI board (row 0 = floor, 20x10 ints) to TBP 40x10 cells."""
    rows = []
    for r in range(NUM_ROW):
        row = []
        for c in range(NUM_COL):
            cell = board_20_bottom_up[r][c]
            row.append("X" if cell else None)
        rows.append(row)
    rows.extend([[None] * NUM_COL for _ in range(20)])
    return rows


def placement_minos(placement: dict) -> list[tuple[int, int]]:
    """Absolute (x, y) cells of a placement, y=0 at floor."""
    loc = placement["location"]
    piece = loc["type"]
    orient = loc["orientation"]
    cx, cy = int(loc["x"]), int(loc["y"])
    return [(cx + dx, cy + dy) for dx, dy in _SRS_MINOS[piece][orient]]


def apply_placement_board(board_20_bottom_up, placement: dict):
    """Return a copy of the board with the placement locked + lines cleared."""
    new_board = np.array(board_20_bottom_up, dtype=np.int32).copy()
    for x, y in placement_minos(placement):
        if 0 <= y < NUM_ROW and 0 <= x < NUM_COL:
            new_board[y][x] = 1
    survivors = [new_board[r].copy() for r in range(NUM_ROW) if not np.all(new_board[r])]
    out = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
    for i, row in enumerate(survivors):
        out[i] = row
    return out


def cleared_lines_after(board_20_bottom_up, placement: dict) -> int:
    before = int(np.sum(board_20_bottom_up)) + 4
    after = int(np.sum(apply_placement_board(board_20_bottom_up, placement)))
    return max(0, (before - after) // NUM_COL)


def placement_to_inputs(placement: dict, falling_piece: str):
    """Map a TBP Placement to (center_x, rot_index, need_hold, piece, spin)."""
    loc = placement["location"]
    piece = loc["type"]
    need_hold = piece != falling_piece
    rot_index = ORIENT_INDEX[loc["orientation"]]
    center_x = int(loc["x"])
    spin = placement.get("spin", "none")
    return center_x, rot_index, need_hold, piece, spin


def _board_mismatch(a, b) -> int:
    return int(np.sum(np.asarray(a, dtype=np.int32) != np.asarray(b, dtype=np.int32)))


def _board_diff_summary(actual, expected, max_rows=6) -> str:
    """Where actual/expected differ — scattered noise vs solid block offset."""
    a = np.asarray(actual, dtype=np.int32)
    e = np.asarray(expected, dtype=np.int32)
    extra = np.argwhere((a == 1) & (e == 0))
    missing = np.argwhere((a == 0) & (e == 1))
    rows_extra = sorted({int(r) for r, _ in extra})[:max_rows]
    rows_missing = sorted({int(r) for r, _ in missing})[:max_rows]
    parts = []
    if extra.size:
        parts.append(f"+{extra.shape[0]} cells @rows{rows_extra}")
    if missing.size:
        parts.append(f"-{missing.shape[0]} cells @rows{rows_missing}")
    return "; ".join(parts) if parts else "(no diff detail)"


class ColdClear2Engine:
    def __init__(self, binary_path: Optional[str] = None, config_path: Optional[str] = None):
        binary = resolve_binary(binary_path)
        self.binary_path = binary
        self.config_path = _resolve_weights_path(config_path)
        args = [binary]
        if self.config_path and os.path.isfile(self.config_path):
            args += ["--config", self.config_path]
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        info = self._read_line()
        if info.get("type") != "info":
            raise RuntimeError(f"Expected TBP info, got: {info}")
        self._send({"type": "rules"})
        ready = self._read_line()
        if ready.get("type") != "ready":
            raise RuntimeError(f"Expected TBP ready, got: {ready}")
        self._active = False
        self.info = info
        self.last_placement = None
        self.last_move_info = None
        self._expected_board = None
        self._last_nexts = None  # next-queue snapshot after last suggest/play
        self._last_mismatch = 0
        self._last_actual_board = None
        self.restart_count = 0
        self._pending_placement = None
        self._pending_expected_board = None
        self._pending_nexts = None

    def _send(self, msg: dict):
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            err = self._drain_stderr()
            raise RuntimeError(f"Cold Clear 2 exited unexpectedly. stderr={err!r}")
        return json.loads(line)

    def _drain_stderr(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            # Non-blocking-ish: only used on failure paths
            return self.proc.stderr.read() or ""
        except Exception:
            return ""

    def start_game(self, board_20_bottom_up, queue_pieces, hold, combo, b2b):
        if self._active:
            self.stop()
        queue = [p for p in queue_pieces if p]
        if not queue:
            raise ValueError("Cold Clear start requires a non-empty queue (current piece)")
        self._send({
            "type": "start",
            "board": to_tbp_board(board_20_bottom_up),
            "queue": queue,
            "hold": hold,
            "combo": int(combo),
            "back_to_back": bool(b2b),
        })
        self._active = True
        self._expected_board = np.array(board_20_bottom_up, dtype=np.int32).copy()
        self._last_nexts = list(queue[1:])  # preview after current

    def suggest(self, timeout_sec: float = 3.0, think_time: float = DEFAULT_THINK_TIME_SEC) -> dict:
        """Ask for a move; retry while CC2 is still warming up (empty moves)."""
        if think_time > 0:
            time.sleep(think_time)
        deadline = time.time() + timeout_sec
        last = None
        while time.time() < deadline:
            self._send({"type": "suggest"})
            while True:
                msg = self._read_line()
                if msg.get("type") == "suggestion":
                    last = msg
                    break
                if msg.get("type") == "error":
                    raise RuntimeError(f"Cold Clear error: {msg}")
            moves = (last or {}).get("moves") or []
            if moves:
                self.last_move_info = last.get("move_info")
                return moves[0]
            time.sleep(0.05)
        err = self._drain_stderr()
        raise RuntimeError(
            f"Cold Clear returned no moves within {timeout_sec}s "
            f"(last={last!r}) stderr={err!r}"
        )

    def confirm_play(self, placement: dict):
        self._send({"type": "play", "move": placement})

    def notify_new_piece(self, piece: str):
        self._send({"type": "new_piece", "piece": piece})

    def stop(self):
        if self._active:
            try:
                self._send({"type": "stop"})
            except Exception:
                pass
            self._active = False
        self._expected_board = None
        self._last_nexts = None
        self._pending_placement = None
        self._pending_expected_board = None
        self._pending_nexts = None

    def quit(self):
        try:
            if self.proc.poll() is None:
                try:
                    self._send({"type": "quit"})
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self._active = False
        self._expected_board = None

    def can_continue(self, board_20_bottom_up) -> bool:
        if not self._active or self._expected_board is None:
            return False
        if self.proc.poll() is not None:
            return False
        self._last_actual_board = np.array(board_20_bottom_up, dtype=np.int32)
        self._last_mismatch = _board_mismatch(board_20_bottom_up, self._expected_board)
        return self._last_mismatch <= BOARD_MISMATCH_CELLS


def get_shared_engine(binary_path=None, config_path=None) -> ColdClear2Engine:
    """Reuse one process; recreate if binary/weights path changes."""
    global _shared_engine, _shared_engine_key
    binary = None
    try:
        binary = resolve_binary(binary_path)
    except FileNotFoundError:
        raise
    weights = _resolve_weights_path(config_path)
    key = (os.path.abspath(binary), os.path.abspath(weights) if weights else None)
    with _engine_lock:
        if _shared_engine is not None and _shared_engine_key != key:
            try:
                _shared_engine.quit()
            except Exception:
                pass
            _shared_engine = None
        if _shared_engine is None:
            _shared_engine = ColdClear2Engine(binary_path, config_path)
            _shared_engine_key = key
        return _shared_engine


def shutdown_shared_engine():
    global _shared_engine, _shared_engine_key
    with _engine_lock:
        if _shared_engine is not None:
            _shared_engine.quit()
            _shared_engine = None
            _shared_engine_key = None


def _sync_new_pieces(engine: ColdClear2Engine, next_pieces):
    """Tell CC about newly revealed preview pieces (end of next queue)."""
    nexts = [p for p in (next_pieces or []) if p is not None]
    prev = engine._last_nexts or []
    # After a place, preview shifts: old nexts[1:] + [NEW, ...].
    # Any trailing pieces not explained by the shift are new.
    if not prev:
        for p in nexts:
            engine.notify_new_piece(p)
    else:
        # Find how many leading pieces match prev[1:], prev[2:], ...
        matched = False
        for skip in range(1, len(prev) + 1):
            head = prev[skip:]
            if nexts[: len(head)] == head:
                for p in nexts[len(head):]:
                    engine.notify_new_piece(p)
                matched = True
                break
        if not matched:
            # Desynced preview — feed full next queue as new pieces
            for p in nexts:
                engine.notify_new_piece(p)
    engine._last_nexts = list(nexts)


def find_best_move(
    current_board,
    current_piece,
    next_pieces,
    held_piece,
    combo,
    b2b,
    pruning_moves,
    pruning_breadth,
    mp_pool=None,
    binary_path=None,
    config_path=None,
    spin_ruleset=None,
    think_time_sec=DEFAULT_THINK_TIME_SEC,
):
    """Drop-in compatible with tetris_ai.find_best_move.

    Returns:
      score, (center_x, (rot_index,), need_hold, combo, b2b, expected_board)

    Uses a persistent TBP session (play + new_piece) when the vision board
    still matches the board we expected after the last play; otherwise
    restarts. pruning_*/mp are unused (CC has its own search) — kept for API
    compatibility with the legacy caller.

    spin_ruleset reclassifies the placement for local b2b tracking to match
    TETR.IO's room setting (see spin_path.RULESETS).
    """
    _ = (pruning_moves, pruning_breadth, mp_pool)  # legacy API only
    from spin_path import classify_spin, normalize_ruleset, pad_board, CW, CCW, ROT180, path_for_placement

    ruleset = normalize_ruleset(spin_ruleset)
    engine = get_shared_engine(binary_path, config_path)
    nexts = [p for p in (next_pieces or []) if p is not None]
    queue = [current_piece] + nexts
    # CC2: pass null hold when slot is empty — do not fake hold=current.
    hold_for_cc = held_piece

    was_active = engine._active
    continued = False
    if engine.can_continue(current_board):
        try:
            _sync_new_pieces(engine, nexts)
            placement = engine.suggest(think_time=think_time_sec)
            continued = True
            log_line("Cold Clear continue (tree kept)")
        except Exception as e:
            log_line(f"Cold Clear continue failed ({e}); restarting session")
            engine.stop()

    if not continued:
        if was_active:
            diff = ""
            if engine._last_actual_board is not None and engine._expected_board is not None:
                diff = " " + _board_diff_summary(
                    engine._last_actual_board, engine._expected_board
                )
            log_line(
                f"Cold Clear: board mismatch ({engine._last_mismatch} cells) — "
                f"restarting session (tree discarded){diff}"
            )
            engine.restart_count += 1
        engine.start_game(current_board, queue, hold_for_cc, combo, b2b)
        placement = engine.suggest(think_time=think_time_sec)

    engine.last_placement = placement
    center_x, rot_index, need_hold, piece, cc_spin = placement_to_inputs(
        placement, current_piece
    )
    expected = apply_placement_board(current_board, placement)
    cleared = cleared_lines_after(current_board, placement)

    # Reclassify under the user's TETR.IO ruleset (path ending in rotate ⇒ spin).
    last_was_rotate = True
    try:
        path = path_for_placement(current_board, placement, ruleset)
        if path and len(path) >= 2:
            last_was_rotate = path[-2] in (CW, CCW, ROT180)
        elif path is None:
            last_was_rotate = (cc_spin or "none") != "none"
    except Exception:
        last_was_rotate = (cc_spin or "none") != "none"
    loc = placement["location"]
    spin = classify_spin(
        ruleset,
        loc["type"],
        pad_board(current_board),
        int(loc["x"]),
        int(loc["y"]),
        loc["orientation"],
        kick_index=4 if last_was_rotate else 0,
        last_was_rotate=last_was_rotate,
    )

    if cleared <= 0:
        new_combo = 0
        new_b2b = b2b
    elif cleared == 4 or spin in ("mini", "full"):
        new_combo = combo + 1
        new_b2b = (b2b + 1) if b2b else 1
    else:
        new_combo = combo + 1
        new_b2b = 0

    # Stash for confirm after keys are sent — don't tell CC the piece landed
    # until the bot has actually executed the placement.
    engine._pending_placement = placement
    engine._pending_expected_board = expected
    engine._pending_nexts = list(nexts)

    info = engine.last_move_info or {}
    score = float(info.get("nodes") or 0)

    return score, (center_x, (rot_index,), need_hold, new_combo, new_b2b, expected)


def cancel_pending_placement(binary_path=None, config_path=None):
    """Drop a stashed placement when the bot decides not to execute it."""
    engine = get_shared_engine(binary_path, config_path)
    engine._pending_placement = None
    engine._pending_expected_board = None
    engine._pending_nexts = None


def confirm_pending_placement(binary_path=None, config_path=None):
    """Tell CC the last suggested move actually landed (call after keypresses)."""
    engine = get_shared_engine(binary_path, config_path)
    placement = engine._pending_placement
    if placement is None:
        return
    expected = engine._pending_expected_board
    nexts = engine._pending_nexts
    engine._pending_placement = None
    engine._pending_expected_board = None
    engine._pending_nexts = None
    try:
        engine.confirm_play(placement)
        if expected is not None:
            engine._expected_board = expected
        if nexts is not None:
            engine._last_nexts = list(nexts)
    except Exception as e:
        log_line(f"Cold Clear play failed ({e}); stopping session")
        engine.stop()


def assert_srs_tables():
    """Sanity: each orientation has 4 minos; centers produce in-bounds spawn."""
    for piece, orients in _SRS_MINOS.items():
        assert set(orients) == {"north", "east", "south", "west"}, piece
        for orient, cells in orients.items():
            assert len(cells) == 4, (piece, orient)
            assert (0, 0) in cells or piece in "IO", (piece, orient, cells)


def smoke_test(binary_path=None, config_path=None):
    """rules → start empty → suggest. Returns the best placement."""
    assert_srs_tables()
    engine = ColdClear2Engine(binary_path, config_path)
    board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
    engine.start_game(board, ["T", "I", "O", "S", "Z"], "T", 0, False)
    move = engine.suggest()
    assert move["location"]["type"] == "T", move
    engine.confirm_play(move)
    engine.notify_new_piece("J")
    move2 = engine.suggest()
    engine.quit()
    return move, move2
