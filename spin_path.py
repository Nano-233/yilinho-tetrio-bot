"""SRS pathfinding + TETR.IO-style spin ruleset classification.

Pathfinder is a Zero-G BFS (CW/CCW/left/right/sonic-drop) matching Cold Clear
v1's approach: find an input sequence that locks the piece at a target
placement, preferring paths whose last action is a rotation when the ruleset
would credit a spin.

Rulesets (common TETR.IO presets):
  t_spins       — guideline 3-corner T only
  all_mini      — T 3-corner; non-T immobile → mini (T immobile alone → none)
  all_mini_plus — default MP/ZEN: all_mini + T immobile → mini
  all_spin      — T 3-corner; immobile (any piece) → full
  none          — never credit spins / never force rotate-last
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from constants import NUM_COL, NUM_ROW

# y=0 floor, y positive up. Matches TBP / cold_clear.
ORIENTS = ("north", "east", "south", "west")
ORIENT_INDEX = {o: i for i, o in enumerate(ORIENTS)}

SPAWN_X = 4
SPAWN_Y = 19  # guideline; board is padded to 40 rows for pathing
BOARD_H = 40

RULESETS = (
    "t_spins",
    "all_mini",
    "all_mini_plus",
    "all_spin",
    "none",
)
DEFAULT_RULESET = "all_mini_plus"

RULESET_LABELS = {
    "t_spins": "T-spins (3-corner)",
    "all_mini": "All-Mini",
    "all_mini_plus": "All-Mini+ (default)",
    "all_spin": "All-Spin",
    "none": "None",
}

# SRS true-rotation minos (dx, dy) from center — same as cold_clear.
SRS_MINOS = {
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

# SRS offset tables (TTC); kick = from_offset[i] - to_offset[i]. y up.
_JLSTZ_OFFSETS = (
    ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),  # north
    ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),  # east
    ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),  # south
    ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),  # west
)
_I_OFFSETS = (
    ((0, 0), (-1, 0), (2, 0), (-1, 0), (2, 0)),
    ((-1, 0), (0, 0), (0, 0), (0, 1), (0, -2)),
    ((-1, 1), (1, 1), (-2, 1), (1, 0), (-2, 0)),
    ((0, 1), (0, 1), (0, 1), (0, -1), (0, 2)),
)
# TETR.IO SRS+ 180 kicks (community-documented; CW/CCW still cover most spins).
_JLSTZ_180 = {
    (0, 2): ((0, 0), (1, 0), (2, 0), (1, 1), (2, -1)),
    (2, 0): ((0, 0), (-1, 0), (-2, 0), (-1, -1), (-2, 1)),
    (1, 3): ((0, 0), (0, 1), (0, 2), (-1, 1), (1, 1)),
    (3, 1): ((0, 0), (0, -1), (0, -2), (1, -1), (-1, -1)),
}
_I_180 = {
    (0, 2): ((0, 0), (-1, 0), (-2, 0), (1, 0), (2, 0)),
    (2, 0): ((0, 0), (1, 0), (2, 0), (-1, 0), (-2, 0)),
    (1, 3): ((0, 0), (0, -1), (0, -2), (0, 1), (0, 2)),
    (3, 1): ((0, 0), (0, 1), (0, 2), (0, -1), (0, -2)),
}

# Inputs executed by bot.py
LEFT, RIGHT, CW, CCW, ROT180, SD, HD = (
    "left", "right", "cw", "ccw", "rot180", "sd", "hd",
)


def normalize_ruleset(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_RULESET
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tspin": "t_spins",
        "tspins": "t_spins",
        "t_spin": "t_spins",
        "allmini": "all_mini",
        "all_mini+": "all_mini_plus",
        "allmini+": "all_mini_plus",
        "allmini_plus": "all_mini_plus",
        "allspin": "all_spin",
        "all": "all_spin",
    }
    key = aliases.get(key, key)
    return key if key in RULESETS else DEFAULT_RULESET


def pad_board(board_20) -> np.ndarray:
    """20x10 bottom-up → 40x10 with empty sky."""
    b = np.zeros((BOARD_H, NUM_COL), dtype=np.int8)
    src = np.asarray(board_20, dtype=np.int8)
    h = min(NUM_ROW, src.shape[0])
    b[:h, :NUM_COL] = src[:h, :NUM_COL]
    return b


def occupied(board, x: int, y: int) -> bool:
    if x < 0 or x >= NUM_COL or y < 0 or y >= BOARD_H:
        return True
    return bool(board[y, x])


def cells(piece: str, x: int, y: int, orient: str):
    return [(x + dx, y + dy) for dx, dy in SRS_MINOS[piece][orient]]


def valid(board, piece: str, x: int, y: int, orient: str) -> bool:
    return all(not occupied(board, cx, cy) for cx, cy in cells(piece, x, y, orient))


def _offsets(piece: str, rot: int):
    if piece == "O":
        return ((0, 0),) * 5
    if piece == "I":
        return _I_OFFSETS[rot]
    return _JLSTZ_OFFSETS[rot]


def try_rotate(board, piece, x, y, rot, new_rot):
    """Apply SRS (+ optional 180 table). Returns (nx, ny, kick_index) or None."""
    if piece == "O":
        return x, y, 0
    delta = (new_rot - rot) % 4
    if delta == 2:
        table = _I_180 if piece == "I" else _JLSTZ_180
        kicks = table.get((rot, new_rot), ((0, 0),))
        for i, (dx, dy) in enumerate(kicks):
            nx, ny = x + dx, y + dy
            if valid(board, piece, nx, ny, ORIENTS[new_rot]):
                return nx, ny, i
        return None
    from_off = _offsets(piece, rot)
    to_off = _offsets(piece, new_rot)
    for i in range(5):
        dx = from_off[i][0] - to_off[i][0]
        dy = from_off[i][1] - to_off[i][1]
        nx, ny = x + dx, y + dy
        if valid(board, piece, nx, ny, ORIENTS[new_rot]):
            return nx, ny, i
    return None


def sonic_drop_y(board, piece, x, y, orient) -> int:
    while valid(board, piece, x, y - 1, orient):
        y -= 1
    return y


def is_immobile(board, piece, x, y, orient) -> bool:
    """Cannot shift L/R/U/D — TETR.IO immobile spin test."""
    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        if valid(board, piece, x + dx, y + dy, orient):
            return False
    return True


def t_spin_corners(board, x, y, orient: str, kick_index: int) -> str:
    """Guideline 3-corner T-spin → 'full' | 'mini' | 'none'."""
    # Diagonal corners around center; walls/floor count filled.
    corners = [
        (x - 1, y + 1),  # NW
        (x + 1, y + 1),  # NE
        (x - 1, y - 1),  # SW
        (x + 1, y - 1),  # SE
    ]
    filled = [occupied(board, cx, cy) for cx, cy in corners]
    if sum(filled) < 3:
        return "none"
    # Front (pointy) corners by orientation
    if orient == "north":
        front = (filled[0], filled[1])  # NW, NE
    elif orient == "east":
        front = (filled[1], filled[3])  # NE, SE
    elif orient == "south":
        front = (filled[2], filled[3])  # SW, SE
    else:
        front = (filled[0], filled[2])  # NW, SW
    if all(front) or kick_index == 4:
        return "full"
    return "mini"


def classify_spin(
    ruleset: str,
    piece: str,
    board,
    x: int,
    y: int,
    orient: str,
    *,
    kick_index: int = 0,
    last_was_rotate: bool = True,
) -> str:
    """Return 'none' | 'mini' | 'full' under the given ruleset."""
    ruleset = normalize_ruleset(ruleset)
    if ruleset == "none" or not last_was_rotate:
        return "none"

    t_kind = "none"
    if piece == "T":
        t_kind = t_spin_corners(board, x, y, orient, kick_index)
    immobile = is_immobile(board, piece, x, y, orient)

    if ruleset == "t_spins":
        return t_kind if piece == "T" else "none"

    if ruleset == "all_mini":
        if piece == "T":
            return t_kind  # immobile alone does not mini-T under All-Mini
        return "mini" if immobile else "none"

    if ruleset == "all_mini_plus":
        if piece == "T":
            if t_kind != "none":
                return t_kind
            return "mini" if immobile else "none"
        return "mini" if immobile else "none"

    if ruleset == "all_spin":
        if piece == "T":
            if t_kind != "none":
                return t_kind
            return "full" if immobile else "none"
        return "full" if immobile else "none"

    return "none"


def would_spin(ruleset, piece, board, x, y, orient) -> bool:
    """True if a rotate-last lock here would score any spin under ruleset."""
    # Check both kick 0 and kick 4 for T (TST).
    for ki in (0, 4):
        if classify_spin(ruleset, piece, board, x, y, orient, kick_index=ki) != "none":
            return True
    return False


def find_path(
    board_20,
    piece: str,
    target_x: int,
    target_y: int,
    target_orient: str,
    *,
    require_rotate_last: bool = False,
    allow_180: bool = True,
    allow_soft_drop: bool = True,
) -> Optional[list]:
    """BFS to target lock. Returns input list ending with 'hd', or None.

    Inputs: left/right/cw/ccw/rot180/sd/hd

    A rotate-last path means the piece arrives at the exact resting target via
    a rotation (SRS kick into the pocket) — required for spin credit.

    allow_soft_drop=False: only rotates/shifts then hard-drop (no mid-air tucks).
    Prefer this first — BFS otherwise treats sonic-SD as one cheap edge and
    picks tucks over longer kick sequences.
    """
    board = pad_board(board_20)
    if target_orient not in ORIENT_INDEX:
        return None
    tgt_rot = ORIENT_INDEX[target_orient]

    sx, sy, srot = SPAWN_X, SPAWN_Y, 0
    if not valid(board, piece, sx, sy, ORIENTS[srot]):
        sy = SPAWN_Y + 1
        if not valid(board, piece, sx, sy, ORIENTS[srot]):
            return None

    start = (sx, sy, srot)
    parent = {start: (None, None, 0)}  # state -> (prev, action, kick)
    q = deque([start])
    found_spin = None
    found_any = None

    def try_enqueue(nx, ny, nrot, action, kick, cur):
        st = (nx, ny, nrot)
        if st in parent:
            return
        parent[st] = (cur, action, kick)
        q.append(st)

    def consider_lock(x, y, rot):
        """Lock if hard-drop from here lands on the target.

        TETR.IO hard-drop sonics: reaching (target_x, target_rot) at *any*
        height is enough — no soft-drop needed. Requiring y==target_y made
        BFS soft-drop early then rotate on the floor (I/S/Z left/right miss).
        """
        nonlocal found_spin, found_any
        if x != target_x or rot != tgt_rot:
            return
        orient = ORIENTS[rot]
        if sonic_drop_y(board, piece, x, y, orient) != target_y:
            return
        _, last_action, last_kick = parent[(x, y, rot)]
        # Spin credit needs the rotate to happen at rest (kick into pocket).
        at_rest = not valid(board, piece, x, y - 1, orient)
        ends_rot = last_action in (CW, CCW, ROT180)
        inputs = _reconstruct(parent, (x, y, rot))
        inputs.append(HD)
        if ends_rot and at_rest:
            if found_spin is None:
                found_spin = (inputs, last_kick)
        elif found_any is None:
            # SD-at-rest is dominated by HD from the pre-SD air state
            if last_action == SD:
                return
            found_any = (inputs, last_kick)

    while q:
        # BFS: first hit is shortest. Do NOT keep searching for a spin ending
        # when one isn't required — that produced 40-step rot180 spam paths.
        if require_rotate_last:
            if found_spin is not None:
                break
        elif found_any is not None or found_spin is not None:
            break

        x, y, rot = q.popleft()
        orient = ORIENTS[rot]
        consider_lock(x, y, rot)
        if require_rotate_last:
            if found_spin is not None:
                break
        elif found_any is not None or found_spin is not None:
            break

        if piece != "O":
            for drot, action in ((1, CW), (3, CCW)):
                nr = (rot + drot) % 4
                res = try_rotate(board, piece, x, y, rot, nr)
                if res:
                    nx, ny, ki = res
                    try_enqueue(nx, ny, nr, action, ki, (x, y, rot))
            if allow_180:
                nr = (rot + 2) % 4
                res = try_rotate(board, piece, x, y, rot, nr)
                if res:
                    nx, ny, ki = res
                    try_enqueue(nx, ny, nr, ROT180, ki, (x, y, rot))

        if valid(board, piece, x - 1, y, orient):
            try_enqueue(x - 1, y, rot, LEFT, 0, (x, y, rot))
        if valid(board, piece, x + 1, y, orient):
            try_enqueue(x + 1, y, rot, RIGHT, 0, (x, y, rot))

        if allow_soft_drop:
            # Sonic soft-drop to rest (one compressed SD edge)
            ly = sonic_drop_y(board, piece, x, y, orient)
            if ly != y:
                try_enqueue(x, ly, rot, SD, 0, (x, y, rot))
            elif valid(board, piece, x, y - 1, orient):
                try_enqueue(x, y - 1, rot, SD, 0, (x, y, rot))

    if require_rotate_last:
        hit = found_spin or found_any
    else:
        # Prefer shortest non-spin; spin only if that's all we found
        hit = found_any or found_spin
    if hit is None:
        return None
    path = _compress_for_execution(_compress_sd(hit[0]))
    if len(path) > MAX_PATH_LEN:
        return None
    return path


def _reconstruct(parent, state) -> list:
    actions = []
    while True:
        prev, action, _ = parent[state]
        if prev is None:
            break
        actions.append(action)
        state = prev
    actions.reverse()
    return actions


def _compress_sd(inputs: list) -> list:
    """Collapse runs of sd into a single sd (sonic)."""
    out = []
    for a in inputs:
        if a == SD and out and out[-1] == SD:
            continue
        out.append(a)
    return out


def _compress_for_execution(inputs: list) -> list:
    """Trailing sd+hd → hd only (hard-drop sonics in TETR.IO; avoids soft-drop key)."""
    if len(inputs) >= 2 and inputs[-2] == SD and inputs[-1] == HD:
        return inputs[:-2] + [HD]
    return inputs


MAX_PATH_LEN = 14  # longer = unusable within lock delay / pathfinder bug


def path_ok(path: Optional[list]) -> bool:
    return bool(path) and len(path) <= MAX_PATH_LEN


def path_needs_tuck(path: Optional[list]) -> bool:
    """True if path soft-drops then keeps moving/rotating (not just SD→HD)."""
    if not path or SD not in path:
        return False
    seen_sd = False
    for a in path:
        if a == SD:
            seen_sd = True
            continue
        if seen_sd and a != HD:
            return True
    return False


def path_for_placement(
    board_20,
    placement: dict,
    ruleset: str,
    *,
    simple_only: bool = False,
) -> Optional[list]:
    """Path to a TBP placement; prefer kick/rotate paths over soft-drop tucks.

    Tries allow_soft_drop=False first (hard-drop / kicks only). Falls back to
    soft-drop tucks unless simple_only=True.
    """
    loc = placement["location"]
    piece = loc["type"]
    orient = loc["orientation"]
    x, y = int(loc["x"]), int(loc["y"])
    board = pad_board(board_20)
    # Drive off our ruleset, not CC's tag — CC may label spins that TETR.IO
    # won't credit under t_spins / none.
    need = would_spin(ruleset, piece, board, x, y, orient)

    def _try(allow_sd: bool, require_rot: bool):
        return find_path(
            board_20,
            piece,
            x,
            y,
            orient,
            require_rotate_last=require_rot,
            allow_soft_drop=allow_sd,
        )

    # 1) Kick/rotate into place without soft-drop (spins / kicks)
    path = _try(False, need)
    if path is None and need:
        path = _try(False, False)
    if path is not None and path_ok(path):
        return path
    if simple_only:
        return None
    # 2) Allow soft-drop (normal gravity placements + tucks)
    path = _try(True, need)
    if path is None and need:
        path = _try(True, False)
    if path is not None and path_ok(path):
        return path
    return None


def classify_placement(board_20, placement: dict, ruleset: str, *, last_was_rotate: bool) -> str:
    loc = placement["location"]
    board = pad_board(board_20)
    return classify_spin(
        ruleset,
        loc["type"],
        board,
        int(loc["x"]),
        int(loc["y"]),
        loc["orientation"],
        kick_index=4 if last_was_rotate else 0,  # optimistic for TST
        last_was_rotate=last_was_rotate,
    )


def _self_check():
    """TSD pocket: path must end with rotate before hard-drop."""
    # Open shaft (cols 3–4) into a 3-corner T south slot @ (4,2).
    board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
    for c in range(NUM_COL):
        board[0][c] = 1
    for c in range(NUM_COL):
        if c != 4:
            board[1][c] = 1
    for c in range(NUM_COL):
        if c not in (3, 4, 5):
            board[2][c] = 1
    for c in range(NUM_COL):
        if c not in (3, 4):
            board[3][c] = 1

    tx, ty, to = 4, 2, "south"
    b40 = pad_board(board)
    assert valid(b40, "T", tx, ty, to), cells("T", tx, ty, to)
    assert classify_spin("t_spins", "T", b40, tx, ty, to) == "full"

    path = find_path(board, "T", tx, ty, to, require_rotate_last=True)
    assert path is not None, "no spin path for TSD"
    assert path[-1] == HD
    assert path[-2] in (CW, CCW, ROT180), path

    assert classify_spin("none", "T", b40, tx, ty, to) == "none"
    assert normalize_ruleset("All-Mini+") == "all_mini_plus"
    assert normalize_ruleset("T-spins") == "t_spins"
    # Immobile L under overhang → mini vs full by ruleset
    ib = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
    for c in range(NUM_COL):
        ib[0][c] = 1
    # Stuff an immobile-ish cavity for L — just verify ruleset dispatch on T immobile
    assert would_spin("all_mini_plus", "T", b40, tx, ty, to)
    print("spin_path self-check ok", path)



if __name__ == "__main__":
    _self_check()
