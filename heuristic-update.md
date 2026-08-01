# Spec: Swapping Cold Clear 2 into the yilinho tetrio-bot pipeline

## 0. Goal

Replace `tetris_ai.py` / `spin_*.py` (the decision engine) with **Cold Clear 2**,
while keeping everything else from the working yilinho pipeline: screen
calibration, color-based board reading (`bot.py`), and the pyautogui
keypress/timing logic.

Cold Clear 2 is not a library you `import` — it's a **separate Rust process**
that speaks a JSON protocol (TBP, the Tetris Bot Protocol) over stdin/stdout.
Your Python code will spawn it as a subprocess and talk to it like a
client talks to a server. This is confirmed directly from its source
(`src/main.rs`): it reads one JSON object per line from stdin and writes one
JSON object per line to stdout — nothing more exotic than that.

```
┌───────────────┐   board/queue/hold state    ┌──────────────────────┐
│ Python: your   │ ───────────────────────────▶│ cold-clear-2.exe     │
│ pipeline       │                              │ (Rust subprocess)    │
│ (state reader, │◀─────────────────────────── │ speaks TBP over      │
│  overlay,      │   suggested placement        │ stdin/stdout          │
│  autoplay)     │                              └──────────────────────┘
└───────────────┘
```

---

## 1. Prerequisites

- Rust toolchain (`rustup`, stable channel) — there are no prebuilt binaries/releases for cold-clear-2, so you must build it yourself.
- `git`.
- No C API / DLL exists for this version (unlike Cold Clear v1) — you cannot call it with `ctypes`. It must be run as a subprocess.

## 2. Build

```bash
git clone https://github.com/MinusKelvin/cold-clear-2.git
cd cold-clear-2
cargo build --release
# binary lands at target/release/cold-clear-2 (or cold-clear-2.exe on Windows)
```

Optional: `cargo build --release -p ... --config <path>` isn't a thing here —
config is passed at *runtime* via a CLI flag, not compiled in (see §6).

Sanity check it runs and speaks TBP:
```bash
./target/release/cold-clear-2
# should immediately print one line of JSON: {"type":"info","name":"Cold Clear 2",...}
```
If you see that line, the binary works. Ctrl+C to exit (there's no clean
prompt — it just blocks reading stdin).

---

## 3. Protocol reference (TBP)

Source of truth: `tetris-bot-protocol/tbp-spec` (`text/0000-mvp.md`).
Confirmed against cold-clear-2's actual implementation (`src/tbp.rs`,
`src/lib.rs`) — it implements the MVP faithfully with one addition
(`move_info` on suggestions).

### Lifecycle

```
Bot   → info                      (sent immediately on launch)
You   → rules                     (send an empty {"type":"rules"})
Bot   → ready
You   → start                     (board, queue, hold, combo, b2b)
You   → suggest                   (ask for a move)
Bot   → suggestion                (list of candidate placements)
You   → play  {move: <chosen>}    (tell it what was actually placed)
You   → new_piece {piece: "X"}    (tell it the queue grew by one)
        (repeat suggest → play → new_piece per piece)
You   → stop                      (end of game)
You   → quit                      (kill it)
```

### Messages you send

| Type | Fields | Notes |
|---|---|---|
| `rules` | *(empty)* | Just `{"type": "rules"}` |
| `start` | `board`, `queue`, `hold`, `combo`, `back_to_back` | See §4 for board format. **Important cold-clear-2 quirk**: if `hold` is null AND `queue` is empty, it silently buffers and waits for a `new_piece` message before actually starting — always send at least the current piece in `queue`. |
| `suggest` | *(empty)* | Ask for the current best move(s) |
| `play` | `move` | Echo back exactly one of the `Placement` objects the bot suggested (see §5), so it can advance its internal search tree |
| `new_piece` | `piece` | Tell it the next piece revealed in your queue preview, one at a time |
| `stop` | *(empty)* | |
| `quit` | *(empty)* | |

### Messages you receive

| Type | Fields | Notes |
|---|---|---|
| `info` | `name`, `version`, `author`, `features` | Sent once on launch, before you send anything |
| `ready` | — | Response to `rules` |
| `suggestion` | `moves` (list of `Placement`), `move_info` | Take `moves[0]` as the best move |

---

## 4. Board format — the critical gotcha

**TBP wants a 40-row board, ordered bottom-to-top.** Your yilinho pipeline
(`constants.py`: `NUM_ROW = 20`) reads only the 20 visible rows, most likely
top-to-bottom (row 0 = near the top of the screen, matching how pieces are
sliced/placed in `tetris_ai.py`).

*Why 40 and not 20?* This isn't TETR.IO-specific — it's the standard Tetris
Guideline that virtually all modern Tetris games (TETR.IO included) follow:
a 20-row visible playfield plus a 20-row hidden "buffer zone" above it,
where pieces spawn and where your stack can briefly extend into if you take
a big garbage hit without topping out. TBP just adopted that same 40-row
convention as its wire format. In practice this is simple for you: your
color-reader only ever sees the visible 20 rows, so **the top 20 rows you
send are always empty** — you're never reading blocks that live in the
hidden zone anyway (if you were, you'd be about to top out regardless).

Confirmed from cold-clear-2's own board decoder (`src/tbp.rs`):
```rust
for x in 0..10 {
    for y in 0..40 {
        if v[y][x].is_some() {
            cols[x] |= 1 << y;   // bit 0 = bottom row
        }
    }
}
```
and the TBP spec's placement coordinates explicitly say `y=0` is "the
bottommost row." So:

- `board` must be a **list of 40 lists of 10 cells**.
- Row index `0` = the floor. Row index `39` = the top of the (mostly hidden)
  buffer zone above the visible playfield.
- Each cell is `null` (empty), a piece-letter string like `"T"`, or `"G"`
  for garbage. Cold-clear-2 only actually checks "is this cell non-null,"
  so it's safe to just use `"X"` for anything you can't confidently
  classify by color — but pass real letters where you have them, in case
  a future cold-clear-2 version starts using them.

**Your conversion function needs to do two things:**
1. Flip your 20-row top-down grid vertically (row 0 becomes row 19, etc.).
2. Pad 20 empty rows on top (indices 20–39) to reach the required 40.

```python
def to_tbp_board(board_20_topdown):
    """board_20_topdown: 20x10, row 0 = top of visible playfield, as read by yilinho's color scanner"""
    flipped = board_20_topdown[::-1]  # now row 0 = floor
    tbp_rows = []
    for row in flipped:
        tbp_rows.append([PIECE_LETTERS.get(cell) for cell in row])  # None for empty
    tbp_rows += [[None] * 10 for _ in range(20)]  # pad hidden buffer rows
    return tbp_rows
```
**Verify this empirically before trusting anything downstream** — print
the converted board next to a known game state and eyeball it. An
off-by-one or unflipped board will make Cold Clear 2 place pieces as if
the board were upside down or shifted, which will look like "the AI is
insane" when really it's a coordinate bug.

---

## 5. Placement / move format

A `Placement` (used both in `suggestion.moves` and in what you echo back
via `play`) looks like:

```json
{
  "location": {
    "type": "T",
    "orientation": "north" | "east" | "south" | "west",
    "x": 4,
    "y": 18
  },
  "spin": "none" | "mini" | "full"
}
```

- `x`/`y` are the coordinates of the piece's **SRS true-rotation center**,
  not a corner — see the spec's diagram reference for exact per-piece
  center definitions (they differ for O/I vs the rest). Get this wrong
  and every placement will be off by a cell or two.
- `y=0` is the floor (see §4).

**This is the important structural difference from Cold Clear v1**: the v1
C API returns an actual **input sequence** (left/right/rotate/soft-drop/hard-drop
actions) ready to execute. TBP/Cold Clear 2 gives you only the **final
resting placement** — you are responsible for turning "put a T piece at
(x=4, y=18, orientation=south)" into an actual sequence of keypresses,
including getting the rotation kick right for spins. This is the main
extra engineering work in this integration.

---

## 6. Translating a placement into keypresses

You need a small **finesse/pathing module** between "target placement" and
"keys to press." Two viable approaches, roughly in order of effort:

**A. Simple greedy approach (do this first):**
1. Rotate toward the target orientation (0–3 presses of rotate-CW/CCW,
   whichever is fewer).
2. Shift left/right until the piece's x matches the target x.
3. Hard-drop.

This works for the large majority of non-spin placements, since most
Cold Clear 2 suggestions land via straightforward drop paths.

**B. Spin-aware pathing (needed for T-spins/mini-spins to actually register):**
For any placement with `"spin": "mini"` or `"spin": "full"`, a hard-drop
after simple shifting will **not** trigger the spin — TETR.IO (like most
modern guideline implementations) uses the "last action was a rotation"
rule (plus the T-spin corner test) to decide whether a placement counts as
a spin. You need to reproduce the actual rotate-into-pocket input sequence
(usually: drop the piece adjacent to the slot, then rotate into place as
the final input, using the correct SRS kick table), not just teleport it
there. This is exactly the kind of logic Cold Clear v1's own frontend
integration code (and StackRabbit's finesse solver) already implements —
worth reading those as reference implementations rather than deriving SRS
kick tables from scratch.

Start with (A), get t-spins landing "close enough" via simple rotate+shift
first, and only invest in (B) once the rest of the pipeline is validated —
most non-spin placements (the majority of moves in a game) don't need it.

---

## 7. Tuning for PvP/attack (the weights file)

Cold Clear 2 ships a default weights file (`src/default.json`) that is
already tuned for attack-oriented, versus-style play — confirms your
assumption that this isn't a marathon-style "just clear lines" bot:

```json
{
  "freestyle_weights": {
    "cell_coveredness": -0.2,
    "holes": -1.5,
    "row_transitions": -0.2,
    "height": -0.4,
    "height_upper_half": -1.5,
    "height_upper_quarter": -5.0,
    "tetris_well_depth": 0.3,
    "tslot": [0.1, 1.5, 2.0, 4.0],
    "has_back_to_back": 0.5,
    "wasted_t": -1.5,
    "normal_clears": [0.0, -2.0, -1.5, -1.0, 3.5],
    "spin_clears": [0.0, 1.0, 4.0, 6.0],
    "back_to_back_clear": 1.0,
    "combo_attack": 1.5,
    "perfect_clear": 15.0,
    "perfect_clear_override": true
  }
}
```
Notice `tetris_well_depth`, `tslot`, `spin_clears`, and `combo_attack` are
all explicitly attack-value features, not just "board cleanliness." Pass
your own config at launch:
```bash
./cold-clear-2 --config my_weights.json
```
Copy `default.json`, tweak values, and rerun to change play style (e.g.
raise `tetris_well_depth`/`spin_clears` further to bias toward T-spin/tetris
stacking over safer flat clears).

---

## 8. Python adapter — interface it should expose

Build this as a drop-in replacement matching `find_best_move()`'s call
signature from `tetris_ai.py`, so the rest of the yilinho pipeline barely
has to change:

```python
import json, subprocess, threading, queue

class ColdClear2Engine:
    def __init__(self, binary_path, config_path=None):
        args = [binary_path]
        if config_path:
            args += ["--config", config_path]
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._read_line()  # consume the initial `info` message
        self._send({"type": "rules"})
        assert self._read_line()["type"] == "ready"

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _read_line(self):
        line = self.proc.stdout.readline()
        return json.loads(line)

    def start_game(self, board_20_topdown, queue_pieces, hold, combo, b2b):
        self._send({
            "type": "start",
            "board": to_tbp_board(board_20_topdown),
            "queue": queue_pieces,
            "hold": hold,
            "combo": combo,
            "back_to_back": b2b,
        })

    def suggest(self):
        self._send({"type": "suggest"})
        msg = self._read_line()
        assert msg["type"] == "suggestion"
        return msg["moves"][0]  # best placement

    def confirm_play(self, placement):
        self._send({"type": "play", "move": placement})

    def notify_new_piece(self, piece):
        self._send({"type": "new_piece", "piece": piece})

    def quit(self):
        self._send({"type": "quit"})
        self.proc.wait(timeout=2)
```

Wire it into your main loop roughly as:
```python
engine = ColdClear2Engine("cold-clear-2/target/release/cold-clear-2", "my_weights.json")
engine.start_game(board, queue, hold, combo, b2b)

while playing:
    move = engine.suggest()
    draw_overlay(move)                # step 3 from earlier
    if demonstrate_mode:
        execute_keypresses(move)      # step 6 above
    engine.confirm_play(move)
    engine.notify_new_piece(next_revealed_piece)
```

---

## 9. Validation plan (do these in order)

1. **Protocol smoke test**: hand-write a tiny script that sends a hardcoded
   `rules` → `start` (empty board, 5-piece queue) → `suggest` and confirms
   you get a sane-looking `suggestion` back. Do this *before* wiring in
   your real screen reader.
2. **Board conversion test**: feed a board with a known, distinctive shape
   (e.g. a single hole in a specific column) through `to_tbp_board()` and
   manually verify the JSON matches what you'd expect bottom-up.
3. **End-to-end, overlay-only**: run your real state reader → Cold Clear 2 →
   overlay, no autoplay yet. Watch a few pieces and sanity-check the
   suggested placements against your own judgment.
4. **Enable autoplay** last, once you trust the placements and the finesse
   module lands pieces where intended (watch specifically for spin
   placements landing as flat drops instead of spins — a strong sign your
   finesse module needs the spin-aware path from §6B).

---

## 10. Known risks / gotchas summary

- **No prebuilt binaries** — you own the Rust build step and any future
  rebuilds.
- **Board orientation bug is the #1 likely failure mode** — verify §4
  before debugging anything else.
- **Spin placements need real input sequences**, not teleportation — flat
  hard-drops will silently fail to register as T-spins even if the piece
  ends up in the right cells.
- **`start` won't actually begin calculating** until it has a non-empty
  queue or gets a `new_piece` — don't send `start` with both `hold: null`
  and `queue: []` and then wonder why `suggest` hangs.
- This is unreleased/experimental software (no version tags) — pin the
  exact git commit you built against so behavior doesn't shift under you
  mid-project.
