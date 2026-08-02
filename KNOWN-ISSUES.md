# Known Issues — yilinho-tetrio-bot (Cold Clear 2 / SRS+)

Last updated after reverting uncommitted rabbit-hole patches and re-applying a **minimal curated fix set** on top of commit `8c3b0a0`.

This document is the starting point for a clean redesign. It catalogs symptoms, root causes, what was tried, and what is still open.

---

## Environment

| Item | Value |
|------|-------|
| OS | Windows 10/11 |
| GPU / capture | NVIDIA RTX 5070 Ti, `bettercam` preferred, `mss` fallback |
| Monitor | Secondary @ offset `[2560, 0]`, 1920×1080 |
| Game | TETR.IO, SRS+, WASD move, arrows rotate, Shift hold, `S` hard drop |
| AI | Cold Clear 2 via TBP subprocess |
| Run | `python ui.py` |

---

## What works (committed baseline + curated fixes)

These are **in the tree now** and worth keeping:

| Fix | Why |
|-----|-----|
| `cc_think_ms` (default 150 ms) + sleep before `suggest()` | CC2 searches in the background; immediate `suggest` returns shallow moves |
| `hold_for_cc = held_piece` (null when empty) | Fake `hold=current` made CC plan impossible hold swaps at startup |
| Deferred `confirm_play` (`confirm_pending_placement` after keys) | Optimistic confirm put CC's expected board ahead of the real game |
| Remove auto-press hold on empty slot | Empty hold at start is normal; auto-hold skipped first cycles |
| `pydirectinput` for game keys | Browser/Electron often ignores `pyautogui` |
| No `mp.Pool` when `ai_engine == cold_clear` | Idle worker processes caused AppHang on shutdown |
| Ghost overlay state only (no Qt from worker thread) | Prevents UI thread deadlock |
| `importlib.reload` on bot Start in UI | Stop/Start picks up code edits without restarting `ui.py` |
| Absolute path for `cc_weights.json` | Relative cwd broke CC config loading from UI |
| `ai_debug.log` + mismatch diff summaries | Post-mortem without pasting terminal |
| Spin path only when CC tags `spin=mini/full` | `would_spin` heuristic caused bad BFS paths on non-spins |
| Greedy rotate → shift → hard drop for non-spins | Reliable execution for normal placements |

---

## What was reverted (do not re-apply blindly)

These patches made debugging harder or masked real bugs:

- `play_next_immediately` + complex queue-advance algebra
- `board_sample_row_offset` (vision row shift)
- Always `start_game` / silencing CC session restarts
- Layered queue dedupe / 40-retry sync loops
- Hold validation skip/restart spaghetti
- Forced `focus_game_window()` clicks during play

---

## Issue catalog

### 1. First piece not bot-controlled

**Symptom:** Bot waits until you manually drop one piece before acting.

**Cause:** Falling piece is inferred from **queue shift** (`prev next[0]`), not from reading the active piece on the board. Board colour classification fails on custom backgrounds (e.g. orange sky reads as L).

**Status:** By design for now. Workflow: start game → manually drop once → bot takes over on next queue change.

**Proper fix (open):** Reliable active-piece detection (ghost silhouette, piece bbox, or queue-only state machine with explicit spawn event).

---

### 2. Stuck after one bot move

**Symptom:** Bot places once, then polls forever (`Polling for piece change...`).

**Causes seen:**
1. Keys never reached the game (`pyautogui` → queue unchanged) — **mitigated** with `pydirectinput`.
2. Loop waits for queue shift even when the next piece is already active — **not fixed** in curated set (rabbit-hole `play_next_immediately` was reverted).

**Status:** Partially mitigated. If still stuck, check `ai_debug.log` for unchanged queue after `greedy:` / spin path lines.

---

### 3. Cold Clear session restarts every move

**Symptom:** Log spam: `board mismatch (N cells) — restarting session`.

**Cause:** Vision board ≠ CC `_expected_board` after last placement. Common patterns:
- `+4 @rows[0-3] / -4 @rows[1-4]` — systematic vertical offset (falling piece still in vision, or confirm timing)
- `+115 cells` — bogus full-board read (overlay, unfocused window, capture black frame)

**Mitigations applied:** Deferred confirm, spawn-row mask (`SPAWN_MASK_ROWS`), `BOARD_MISMATCH_CELLS = 0` (strict restart).

**Status:** Restarts are **correct** when mismatched — do not disable without fixing vision/sync. Investigate mismatch dumps (`ai_debug_mismatch_board.*` if enabled).

---

### 4. Queue / piece identity desync (bad stacking)

**Symptom:** CC chooses garbage placements; log shows duplicate pieces in queue (e.g. `[Z, Z, ...]`) or wrong piece after first hold.

**Root causes:**
| Bug | Detail |
|-----|--------|
| Preview vs active | After hold, active piece is `preview[1]`, not `preview[0]` |
| Duplicate in CC queue | Passing `next_pieces` that still include the active piece |
| First-hold preview shift | Post-lock preview should drop **two** slots, not one |
| Hold execution | When `need_hold=True`, held piece must match CC placement type |

**Status:** **Open — highest priority.** Curated fixes help CC hold modelling but do not fully solve queue algebra. Needs one coherent queue state machine documented against TETR.IO preview semantics.

---

### 5. Hold at startup

**Symptom (old):** `skip: hold is empty`, bot never acts until 2+ manual pieces.

**Cause:** Auto-press hold on empty slot + fake `hold=current` in CC.

**Status:** **Fixed** in curated set. Empty hold at start is valid.

---

### 6. CC told piece landed before keys sent

**Symptom:** Mismatch immediately after every move even when placement looked correct.

**Cause:** `confirm_play` in `find_best_move` before `place_cc`.

**Status:** **Fixed** — pending placement confirmed after key execution.

---

### 7. Shallow / bad CC moves

**Symptom:** CC returns weak placements, low node counts.

**Cause:** `suggest()` called with no think budget.

**Status:** **Fixed** — `cc_think_ms` configurable (UI + profile).

---

### 8. Keys not reaching TETR.IO

**Symptom:** Queue unchanged after autodrop; infinite poll.

**Cause:** `pyautogui` ignored by browser.

**Status:** **Mitigated** — `pydirectinput` on Windows. Startup log shows `input=pydirectinput`.

---

### 9. Vision / capture failures

**Symptoms:**
- Black next-piece samples → `bettercam` None frame or wrong monitor
- `Board looks obscured` → window covered/unfocused
- `locked (+131 cells)` → full bottom rows filled in one frame (overlay or bad capture)

**Mitigations in baseline:** bettercam retry, black-frame fallback warning, obscured-board guard.

**Status:** Calibration-sensitive. Re-run calibration if monitor layout changes. Verify `screen_offset` matches secondary monitor.

---

### 10. Execution: wrong column / rotation

**Symptom:** Piece locks with correct cell count but wrong shape (+4/-4 scattered mismatch).

**Possible causes (unconfirmed):**
- Kick table mismatch (SRS+ vs guideline) for specific pieces
- Greedy path uses spawn-centered shift math vs kick-adjusted center
- I-piece vertical kicks need BFS path, not greedy rotate-first

**Status:** **Open.** `spin_path.py` has SRS+ 180 kicks; I-piece CW/CCW may still need path-first execution for some orientations.

---

### 11. UI / process stability

| Issue | Status |
|-------|--------|
| mp Pool hang with CC | Fixed (no pool for cold_clear) |
| Ghost Qt calls from worker | Fixed (state-only publish) |
| Stale bot.py on Stop/Start | Fixed (importlib.reload) |
| CC subprocess left running | `shutdown_shared_engine()` on bot close |

---

## Recommended debug workflow

1. **Restart UI** after code changes (or Stop → Start; reload is automatic).
2. **Manual drop once** after board is empty / new game.
3. Watch **`ai_debug.log`** for:
   - `nodes:` / `CC place:` — piece type, hold flag, spin
   - `Cold Clear continue (tree kept)` vs `board mismatch`
   - `greedy:` / spin path lines
4. On mismatch: compare vision vs expected; check capture (focus, monitor offset).
5. Do **not** patch queue/hold with ad-hoc dedupe — fix the state model once.

---

## Open design questions

1. **Queue model:** Does TETR.IO preview exclude the active piece? Document exact mapping to CC TBP `queue` + `new_piece` events.
2. **Hold sync:** When CC returns `need_hold` with empty hold, how do preview indices shift before and after lock?
3. **Continue vs restart:** When is vision mismatch acceptable noise vs fatal desync?
4. **Active piece:** Can queue-only inference ever be enough, or do we need spawn-area vision?

---

## File map

| File | Role |
|------|------|
| `bot.py` | Main loop, vision, placement, queue inference |
| `cold_clear.py` | CC2 TBP adapter, session continue/restart, think budget |
| `spin_path.py` | SRS BFS pathing, spin rulesets |
| `ui.py` | PyQt control panel, worker thread |
| `profiles.py` / `profiles/*.json` | Saved layouts + AI settings |
| `cc_weights.json` | CC heuristic weights |
| `ai_debug.log` | Runtime CC diagnostics (gitignored) |

---

## Git state note

Only commit on branch: `8c3b0a0` — CC vision/capture pipeline. All debugging after that was uncommitted; rabbit-hole changes were discarded. Curated fixes above are **uncommitted working changes** on top of that commit.
