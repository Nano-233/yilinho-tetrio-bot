"""PyQt5 control panel + PythonOverlayLib calibration overlay.

Entry: python ui.py
"""
from __future__ import annotations

import importlib
import math
import sys
import traceback

import pyautogui
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal

import overlay_lib
from overlay_lib import (
    DrawLine,
    DrawText,
    FlDrawRect,
    RgbaColor,
    SkDrawRect,
    Vector2D,
)

from bot import (
    CALIBRATION_COUNTDOWN_SEC,
    DEFAULT_KEYBINDS,
    bot_from_config,
    ensure_dpi_awareness,
    monitor_containing,
)
import bot as bot_module
from constants import NUM_COL, NUM_ROW
from profiles import (
    DEFAULT_APP_HOTKEYS,
    create_profile,
    delete_profile,
    duplicate_profile,
    list_profiles,
    load_app_settings,
    load_profile,
    migrate_legacy_config,
    repair_monitor_if_stale,
    save_app_settings,
    save_profile,
    sync_legacy_config,
)

# Shared with overlay drawlist callback (OverlayLib owns the Qt app).
_STATE = {
    "config": None,
    "overlay_enabled": False,
    "panel": None,
    "overlay_win": None,
    # Overlay window origin (virtual desktop). Draw coords are window-local.
    "virtual_origin": (0, 0),
    "play_mode": "autodrop",  # "autodrop" | "suggest"
    "ghost_cells": [],  # [(col, row_top), ...]
    "ghost_label": "",
}

YELLOW = RgbaColor(255, 255, 0, 220)
GREEN = RgbaColor(0, 255, 0, 220)
PURPLE = RgbaColor(255, 0, 255, 220)
GHOST_FILL = RgbaColor(0, 220, 255, 110)
GHOST_OUTLINE = RgbaColor(0, 255, 255, 255)

GAME_BIND_LABELS = [
    ("move_left", "Move left"),
    ("move_right", "Move right"),
    ("soft_drop", "Soft drop"),
    ("hard_drop", "Hard drop"),
    ("rotate_cw", "Rotate CW"),
    ("rotate_ccw", "Rotate CCW"),
    ("rotate_180", "Rotate 180"),
    ("hold", "Hold"),
]

APP_HOTKEY_LABELS = [
    ("start_bot", "Start bot"),
    ("stop_bot", "Stop bot"),
    ("toggle_overlay", "Toggle overlay"),
    ("toggle_play_mode", "Toggle autodrop/suggest"),
]

CALIB_FIELDS = [
    ("board_top_left", "Board top-left"),
    ("board_bottom_right", "Board bottom-right"),
    ("next_piece_xy_0", "Next piece #1"),
    ("next_piece_xy_4", "Next piece #5"),
    ("held_piece_xy", "Held piece"),
]


def _next_slots(config):
    x0, y0 = config["next_piece_xy_0"]
    x4, y4 = config["next_piece_xy_4"]
    return [
        (x0, y0),
        ((x0 + x4) // 2, y0 + math.floor(((y4 - y0) / 4) * 1)),
        ((x0 + x4) // 2, y0 + math.floor(((y4 - y0) / 4) * 2)),
        ((x0 + x4) // 2, y0 + math.floor(((y4 - y0) / 4) * 3)),
        (x4, y4),
    ]


def _pixel_half(config):
    y0 = config["next_piece_xy_0"][1]
    y4 = config["next_piece_xy_4"][1]
    return max(1, ((y4 - y0) // NUM_COL) // 2)


def virtual_screen_rect():
    """Full virtual desktop (x, y, w, h) — must run after QApplication exists.

    OverlayLib only covers the primary monitor; we stretch to the virtual
    desktop so shapes on a second monitor are visible. Prefer Qt's geometry
    (same DPI space as mss once Qt has started).
    """
    app = QtWidgets.QApplication.instance()
    if app is not None and app.primaryScreen() is not None:
        g = app.primaryScreen().virtualGeometry()
        return int(g.x()), int(g.y()), int(g.width()), int(g.height())
    # Fallback before Qt is ready
    import ctypes
    user32 = ctypes.windll.user32
    return (
        int(user32.GetSystemMetrics(76)),
        int(user32.GetSystemMetrics(77)),
        int(user32.GetSystemMetrics(78)),
        int(user32.GetSystemMetrics(79)),
    )


def sync_overlay_geometry():
    """Place overlay on the profile's monitor (fallback: full virtual desktop).

    OverlayLib defaults to primary-only. Covering only the game monitor is more
    reliable on Windows than one huge multi-monitor translucent window.
    """
    win = _STATE.get("overlay_win")
    if win is None:
        return
    cfg = _STATE.get("config") or {}
    ox = oy = None
    if "screen_offset" in cfg and "screen_resolution" in cfg:
        ox, oy = cfg["screen_offset"]
        sw, sh = cfg["screen_resolution"]
        if sw > 0 and sh > 0:
            _STATE["virtual_origin"] = (int(ox), int(oy))
            win.setGeometry(int(ox), int(oy), int(sw), int(sh))
            return
    vx, vy, vw, vh = virtual_screen_rect()
    _STATE["virtual_origin"] = (vx, vy)
    win.setGeometry(vx, vy, max(1, vw), max(1, vh))


def _abs(config, xy):
    ox, oy = config["screen_offset"]
    return int(ox + xy[0]), int(oy + xy[1])


def _local(abs_xy):
    """Absolute screen → overlay window coords."""
    vx, vy = _STATE.get("virtual_origin", (0, 0))
    return abs_xy[0] - vx, abs_xy[1] - vy


def build_ghost_shapes(config, cells, label=""):
    """Cyan cells for the suggested landing position (col, row_top)."""
    if not config or not cells:
        return []
    left, top = _local(_abs(config, config["board_top_left"]))
    right, bottom = _local(_abs(config, config["board_bottom_right"]))
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return []
    cell_w = w / NUM_COL
    cell_h = h / NUM_ROW
    shapes = []
    for col, row in cells:
        if not (0 <= col < NUM_COL and 0 <= row < NUM_ROW):
            continue
        x = int(left + col * cell_w)
        y = int(top + row * cell_h)
        cw = max(1, int(cell_w) - 1)
        ch = max(1, int(cell_h) - 1)
        shapes.append(FlDrawRect(
            Vector2D(x, y), cw, ch, GHOST_FILL, GHOST_OUTLINE, 2
        ))
    if label:
        shapes.append(DrawText(
            Vector2D(left, min(bottom + 16, top + h + 16)),
            12, label, "Arial", GHOST_OUTLINE, 1
        ))
    return shapes


def build_overlay_shapes(config):
    """Calibration regions + optional suggestion ghost."""
    if not config:
        return []
    shapes = []
    left, top = _local(_abs(config, config["board_top_left"]))
    right, bottom = _local(_abs(config, config["board_bottom_right"]))
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return shapes
    shapes.append(SkDrawRect(Vector2D(left, top), w, h, YELLOW, 2))
    cell_w = w / NUM_COL
    cell_h = h / NUM_ROW
    for col in range(1, NUM_COL):
        x = int(left + col * cell_w)
        shapes.append(DrawLine(Vector2D(x, top), Vector2D(x, bottom), YELLOW, 1))
    for row in range(1, NUM_ROW):
        y = int(top + row * cell_h)
        shapes.append(DrawLine(Vector2D(left, y), Vector2D(right, y), YELLOW, 1))
    shapes.append(DrawText(
        Vector2D(left, max(12, top - 4)), 12, "board", "Arial", YELLOW, 1
    ))

    half = _pixel_half(config)
    for i, (rx, ry) in enumerate(_next_slots(config)):
        ax, ay = _local(_abs(config, (rx, ry)))
        shapes.append(SkDrawRect(
            Vector2D(ax - half, ay - half), half * 2, half * 2, GREEN, 2
        ))
        shapes.append(DrawText(
            Vector2D(ax + half + 4, ay), 11, f"next{i}", "Arial", GREEN, 1
        ))

    hx, hy = _local(_abs(config, config["held_piece_xy"]))
    shapes.append(SkDrawRect(
        Vector2D(hx - half, hy - half), half * 2, half * 2, PURPLE, 2
    ))
    shapes.append(DrawText(
        Vector2D(hx + half + 4, hy), 11, "held", "Arial", PURPLE, 1
    ))

    shapes.extend(build_ghost_shapes(
        config, _STATE.get("ghost_cells") or [], _STATE.get("ghost_label") or ""
    ))
    return shapes


def drawlist_callback():
    # Suggest ghost needs the overlay window; calib overlay is optional toggle.
    has_ghost = bool(_STATE.get("ghost_cells"))
    if not _STATE["overlay_enabled"] and not has_ghost:
        return []
    if not _STATE["overlay_enabled"] and has_ghost:
        # Draw only the ghost (no calib chrome) when overlay checkbox is off
        return build_ghost_shapes(
            _STATE["config"],
            _STATE.get("ghost_cells") or [],
            _STATE.get("ghost_label") or "",
        )
    return build_overlay_shapes(_STATE["config"])


class BotWorker(QtCore.QThread):
    errored = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self.bot = None

    def run(self):
        try:
            # ponytail: ui.py imports bot once at process start — reload so
            # Stop/Start picks up edits without restarting ui.py.
            import cold_clear
            importlib.reload(cold_clear)
            importlib.reload(bot_module)
            self.bot = bot_module.bot_from_config(self._config)
            self.bot.play_mode = _STATE.get("play_mode", "autodrop")
            self.bot.publish_ghost = _publish_ghost_from_bot
            self.bot.run()
        except Exception:
            # stop() kills the CC process to unblock — don't treat that as an error
            if not (self.bot is not None and getattr(self.bot, "_stop", False)):
                self.errored.emit(traceback.format_exc())
        finally:
            _publish_ghost_from_bot([], "")
            if self.bot is not None:
                try:
                    self.bot.close()
                except Exception:
                    pass
                self.bot = None
            self.stopped.emit()

    def request_stop(self):
        if self.bot is not None:
            self.bot.stop()

    def set_play_mode(self, mode):
        if self.bot is not None:
            self.bot.play_mode = mode


def _publish_ghost_from_bot(cells, label=""):
    # State only — OverlayLib's drawlist reads this. Never touch Qt widgets
    # from the bot worker thread (that AppHangs python.exe under Windows).
    _STATE["ghost_cells"] = list(cells or [])
    _STATE["ghost_label"] = label or ""


class PointEdit(QtWidgets.QWidget):
    """Two spinboxes for an (x, y) pair + Capture button."""

    capture_requested = pyqtSignal(str)

    def __init__(self, field_key, label, parent=None):
        super().__init__(parent)
        self.field_key = field_key
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label), 2)
        self.x = QtWidgets.QSpinBox()
        self.y = QtWidgets.QSpinBox()
        for spin in (self.x, self.y):
            spin.setRange(-10000, 10000)
            spin.setMaximumWidth(90)
            layout.addWidget(spin)
        btn = QtWidgets.QPushButton("Capture")
        btn.clicked.connect(lambda: self.capture_requested.emit(self.field_key))
        layout.addWidget(btn)

    def get(self):
        return [self.x.value(), self.y.value()]

    def set(self, xy):
        self.x.setValue(int(xy[0]))
        self.y.setValue(int(xy[1]))


class ControlPanel(QtWidgets.QMainWindow):
    # Cross-thread hotkey → UI
    hotkey_start = pyqtSignal()
    hotkey_stop = pyqtSignal()
    hotkey_overlay = pyqtSignal()
    hotkey_play_mode = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TetrioBot")
        self.resize(520, 640)

        migrate_legacy_config()
        self.settings = load_app_settings()
        self.profile_name = self.settings["active_profile"]
        self.config = load_profile(self.profile_name) or {}
        _STATE["config"] = self.config
        _STATE["panel"] = self
        _STATE["play_mode"] = self.settings.get("play_mode", "autodrop")

        self.worker = None
        self._capturing = False
        self._capture_field = None
        self._countdown = 0
        self._hotkey_hooks = []

        self.hotkey_start.connect(self.start_bot)
        self.hotkey_stop.connect(self.stop_bot)
        self.hotkey_overlay.connect(self.toggle_overlay)
        self.hotkey_play_mode.connect(self.toggle_play_mode)

        self._build_ui()
        self._load_profile_into_form(self.profile_name)
        self._register_hotkeys()

        self._capture_timer = QtCore.QTimer(self)
        self._capture_timer.setInterval(1000)
        self._capture_timer.timeout.connect(self._capture_tick)

        QtCore.QTimer.singleShot(0, self._after_show)

    def _after_show(self):
        # OverlayLib only sizes to the primary monitor — expand to virtual desktop.
        sync_overlay_geometry()
        self._update_monitor_hint()
        win = _STATE.get("overlay_win")
        if win is not None and not _STATE["overlay_enabled"]:
            win.hide()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)

        # Profile row
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Profile"))
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        row.addWidget(self.profile_combo, 1)
        for text, slot in (
            ("New", self._new_profile),
            ("Duplicate", self._dup_profile),
            ("Delete", self._del_profile),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)

        self.status = QtWidgets.QLabel("Idle")
        v.addWidget(self.status)

        # Run controls
        run = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start bot")
        self.start_btn.clicked.connect(self.start_bot)
        self.stop_btn = QtWidgets.QPushButton("Stop bot")
        self.stop_btn.clicked.connect(self.stop_bot)
        self.stop_btn.setEnabled(False)
        self.overlay_check = QtWidgets.QCheckBox("Overlay")
        self.overlay_check.toggled.connect(self._set_overlay)
        run.addWidget(self.start_btn)
        run.addWidget(self.stop_btn)
        run.addWidget(self.overlay_check)
        run.addStretch(1)
        v.addLayout(run)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Play mode"))
        self.mode_autodrop = QtWidgets.QRadioButton("Autodrop")
        self.mode_suggest = QtWidgets.QRadioButton("Suggest")
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.addButton(self.mode_autodrop)
        self.mode_group.addButton(self.mode_suggest)
        if _STATE.get("play_mode") == "suggest":
            self.mode_suggest.setChecked(True)
        else:
            self.mode_autodrop.setChecked(True)
        self.mode_autodrop.toggled.connect(self._on_play_mode_radio)
        mode_row.addWidget(self.mode_autodrop)
        mode_row.addWidget(self.mode_suggest)
        mode_row.addStretch(1)
        v.addLayout(mode_row)

        tabs = QtWidgets.QTabWidget()
        v.addWidget(tabs, 1)

        # --- Calibration ---
        cal = QtWidgets.QWidget()
        cal_l = QtWidgets.QVBoxLayout(cal)
        self.screen_label = QtWidgets.QLabel("Screen: —")
        cal_l.addWidget(self.screen_label)
        self.monitor_hint = QtWidgets.QLabel("")
        self.monitor_hint.setWordWrap(True)
        cal_l.addWidget(self.monitor_hint)
        self.point_edits = {}
        for key, label in CALIB_FIELDS:
            edit = PointEdit(key, label)
            edit.capture_requested.connect(self._begin_capture)
            self.point_edits[key] = edit
            cal_l.addWidget(edit)
        tip = QtWidgets.QLabel(
            "Capture: click Capture, switch to TETR.IO, hover the target, wait.\n"
            "Capture Board top-left first (sets which monitor). Overlay must cover "
            "the game monitor — toggle Overlay after capturing to verify."
        )
        tip.setWordWrap(True)
        cal_l.addWidget(tip)
        cal_l.addStretch(1)
        tabs.addTab(cal, "Calibration")

        # --- Delays / AI ---
        ai = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(ai)
        self.move_delay = QtWidgets.QSpinBox()
        self.move_delay.setRange(0, 500)
        self.action_delay = QtWidgets.QSpinBox()
        self.action_delay.setRange(0, 500)
        self.variance = QtWidgets.QSpinBox()
        self.variance.setRange(0, 100)
        self.mp = QtWidgets.QSpinBox()
        self.mp.setRange(1, 64)
        self.pruning_moves = QtWidgets.QSpinBox()
        self.pruning_moves.setRange(1, 20)
        self.pruning_breadth = QtWidgets.QSpinBox()
        self.pruning_breadth.setRange(1, 20)
        self.label_edit = QtWidgets.QLineEdit()
        form.addRow("Label", self.label_edit)
        form.addRow("Move delay (ms)", self.move_delay)
        form.addRow("Action delay (ms)", self.action_delay)
        form.addRow("Variance (%)", self.variance)
        form.addRow("Workers (mp)", self.mp)
        form.addRow("Pruning moves", self.pruning_moves)
        form.addRow("Pruning breadth", self.pruning_breadth)
        self.ai_engine_group = QtWidgets.QButtonGroup(self)
        self.ai_legacy = QtWidgets.QRadioButton("Legacy (yilinho)")
        self.ai_cc = QtWidgets.QRadioButton("Cold Clear 2")
        self.ai_engine_group.addButton(self.ai_legacy)
        self.ai_engine_group.addButton(self.ai_cc)
        eng_row = QtWidgets.QHBoxLayout()
        eng_row.addWidget(self.ai_legacy)
        eng_row.addWidget(self.ai_cc)
        eng_row.addStretch(1)
        form.addRow("Heuristics", eng_row)
        self.spin_ruleset = QtWidgets.QComboBox()
        try:
            from spin_path import RULESETS, RULESET_LABELS, DEFAULT_RULESET
            for key in RULESETS:
                self.spin_ruleset.addItem(RULESET_LABELS.get(key, key), key)
            self._default_spin_ruleset = DEFAULT_RULESET
        except Exception:
            for key, label in (
                ("t_spins", "T-spins (3-corner)"),
                ("all_mini", "All-Mini"),
                ("all_mini_plus", "All-Mini+ (default)"),
                ("all_spin", "All-Spin"),
                ("none", "None"),
            ):
                self.spin_ruleset.addItem(label, key)
            self._default_spin_ruleset = "all_mini_plus"
        form.addRow("Spin ruleset", self.spin_ruleset)
        self.cc_think_ms = QtWidgets.QSpinBox()
        self.cc_think_ms.setRange(0, 2000)
        self.cc_think_ms.setSuffix(" ms")
        self.cc_think_ms.setValue(150)
        self.cc_think_ms.setToolTip(
            "Sleep before each Cold Clear suggest() so CC accumulates search nodes."
        )
        form.addRow("CC think time", self.cc_think_ms)
        self.ai_legacy.toggled.connect(self._sync_legacy_ai_controls)
        self.ai_cc.toggled.connect(self._sync_legacy_ai_controls)
        self._legacy_ai_widgets = (self.mp, self.pruning_moves, self.pruning_breadth)
        self._sync_legacy_ai_controls()
        tabs.addTab(ai, "Delays / AI")

        # --- Keybinds ---
        keys = QtWidgets.QWidget()
        keys_l = QtWidgets.QVBoxLayout(keys)
        keys_l.addWidget(QtWidgets.QLabel("Game keybinds (must match TETR.IO)"))
        self.game_bind_edits = {}
        gform = QtWidgets.QFormLayout()
        for key, label in GAME_BIND_LABELS:
            edit = QtWidgets.QLineEdit()
            self.game_bind_edits[key] = edit
            gform.addRow(label, edit)
        keys_l.addLayout(gform)
        keys_l.addWidget(QtWidgets.QLabel("App hotkeys (global)"))
        self.app_hotkey_edits = {}
        aform = QtWidgets.QFormLayout()
        for key, label in APP_HOTKEY_LABELS:
            edit = QtWidgets.QLineEdit()
            self.app_hotkey_edits[key] = edit
            aform.addRow(label, edit)
        keys_l.addLayout(aform)
        keys_l.addStretch(1)
        tabs.addTab(keys, "Keybinds")

        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self.save_all)
        v.addWidget(save_btn)

        self._refresh_profile_combo()

    def _sync_legacy_ai_controls(self, *_):
        """Workers/pruning only apply to legacy AI — grey them out for CC."""
        legacy = self.ai_legacy.isChecked()
        for w in getattr(self, "_legacy_ai_widgets", ()):
            w.setEnabled(legacy)
        if hasattr(self, "spin_ruleset"):
            self.spin_ruleset.setEnabled(not legacy)
        if hasattr(self, "cc_think_ms"):
            self.cc_think_ms.setEnabled(not legacy)

    def _refresh_profile_combo(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(list_profiles())
        idx = self.profile_combo.findText(self.profile_name)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def _set_status(self, text):
        self.status.setText(text)

    def _update_monitor_hint(self):
        """Show live monitor layout + warn if profile offset is stale."""
        try:
            ensure_dpi_awareness()
            import mss
            lines = []
            with mss.MSS() as sct:
                for i, mon in enumerate(sct.monitors[1:], start=1):
                    tag = " primary" if mon.get("is_primary") else ""
                    lines.append(
                        f"  M{i}: ({mon['left']},{mon['top']}) "
                        f"{mon['width']}x{mon['height']}{tag}"
                    )
            ox, oy = self.config.get("screen_offset", [None, None])
            sw, sh = self.config.get("screen_resolution", [None, None])
            match = False
            with mss.MSS() as sct:
                for mon in sct.monitors[1:]:
                    if (mon["left"] == ox and mon["top"] == oy
                            and mon["width"] == sw and mon["height"] == sh):
                        match = True
                        break
            status = "OK" if match else (
                "MISMATCH — re-capture Board top-left on the TETR.IO monitor"
            )
            vx, vy, vw, vh = virtual_screen_rect()
            self.monitor_hint.setText(
                "Monitors now:\n" + "\n".join(lines)
                + f"\nVirtual desktop: ({vx},{vy}) {vw}x{vh}"
                + f"\nProfile screen: ({ox},{oy}) {sw}x{sh} → {status}"
            )
        except Exception as e:
            self.monitor_hint.setText(f"Monitor probe failed: {e}")

    def _is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def _collect_form(self):
        cfg = dict(self.config)
        for key, edit in self.point_edits.items():
            cfg[key] = edit.get()
        cfg["label"] = self.label_edit.text().strip() or self.profile_name
        cfg["move_delay_ms"] = self.move_delay.value()
        cfg["action_delay_ms"] = self.action_delay.value()
        cfg["delay_variance_percent"] = self.variance.value()
        cfg["mp"] = self.mp.value()
        cfg["pruning_moves"] = self.pruning_moves.value()
        cfg["pruning_breadth"] = self.pruning_breadth.value()
        cfg["ai_engine"] = "cold_clear" if self.ai_cc.isChecked() else "legacy"
        if hasattr(self, "spin_ruleset"):
            cfg["spin_ruleset"] = self.spin_ruleset.currentData() or "all_mini_plus"
        if hasattr(self, "cc_think_ms"):
            cfg["cc_think_ms"] = self.cc_think_ms.value()
        binds = dict(DEFAULT_KEYBINDS)
        for key, edit in self.game_bind_edits.items():
            val = edit.text().strip().lower()
            binds[key] = val or DEFAULT_KEYBINDS[key]
        cfg["keybinds"] = binds
        # Keep screen fields from current config (updated on board TL capture)
        cfg.setdefault("screen_offset", self.config.get("screen_offset", [0, 0]))
        cfg.setdefault(
            "screen_resolution", self.config.get("screen_resolution", [1920, 1080])
        )
        return cfg

    def _collect_app_hotkeys(self):
        hotkeys = dict(DEFAULT_APP_HOTKEYS)
        for key, edit in self.app_hotkey_edits.items():
            val = edit.text().strip().lower()
            hotkeys[key] = val or DEFAULT_APP_HOTKEYS[key]
        return hotkeys

    def _load_profile_into_form(self, name):
        cfg = load_profile(name)
        if cfg is None:
            QtWidgets.QMessageBox.warning(self, "Profile", f"Missing profile '{name}'")
            return
        cfg, repaired = repair_monitor_if_stale(cfg)
        if repaired:
            save_profile(name, cfg)
            sync_legacy_config(name)
        self.profile_name = name
        self.config = cfg
        _STATE["config"] = cfg
        for key, edit in self.point_edits.items():
            edit.set(cfg.get(key, [0, 0]))
        ox, oy = cfg.get("screen_offset", [0, 0])
        sw, sh = cfg.get("screen_resolution", [0, 0])
        self.screen_label.setText(f"Screen: offset=({ox},{oy}) size={sw}x{sh}")
        self._update_monitor_hint()
        self.label_edit.setText(cfg.get("label") or name)
        self.move_delay.setValue(int(cfg.get("move_delay_ms", 30)))
        self.action_delay.setValue(int(cfg.get("action_delay_ms", 50)))
        self.variance.setValue(int(cfg.get("delay_variance_percent", 20)))
        self.mp.setValue(int(cfg.get("mp", 16)))
        self.pruning_moves.setValue(int(cfg.get("pruning_moves", 5)))
        self.pruning_breadth.setValue(int(cfg.get("pruning_breadth", 5)))
        if (cfg.get("ai_engine") or "legacy") == "cold_clear":
            self.ai_cc.setChecked(True)
        else:
            self.ai_legacy.setChecked(True)
        if hasattr(self, "spin_ruleset"):
            want = cfg.get("spin_ruleset") or getattr(
                self, "_default_spin_ruleset", "all_mini_plus"
            )
            idx = self.spin_ruleset.findData(want)
            if idx < 0:
                # try normalize
                try:
                    from spin_path import normalize_ruleset
                    want = normalize_ruleset(want)
                    idx = self.spin_ruleset.findData(want)
                except Exception:
                    idx = -1
            if idx >= 0:
                self.spin_ruleset.setCurrentIndex(idx)
        if hasattr(self, "cc_think_ms"):
            self.cc_think_ms.setValue(int(cfg.get("cc_think_ms", 150)))
        self._sync_legacy_ai_controls()
        binds = cfg.get("keybinds") or {}
        for key, edit in self.game_bind_edits.items():
            edit.setText(binds.get(key, DEFAULT_KEYBINDS[key]))
        hotkeys = self.settings.get("hotkeys") or DEFAULT_APP_HOTKEYS
        for key, edit in self.app_hotkey_edits.items():
            edit.setText(hotkeys.get(key, DEFAULT_APP_HOTKEYS[key]))
        if repaired:
            self._set_status(
                "Profile monitor remapped to current display layout — "
                "toggle Overlay to verify, re-capture if misaligned"
            )

    def _on_profile_changed(self, name):
        if not name or name == self.profile_name:
            return
        if self._is_running():
            self.stop_bot()
        self._load_profile_into_form(name)
        self.settings["active_profile"] = name
        save_app_settings(self.settings)
        sync_legacy_config(name)
        self._set_status(f"Loaded profile '{name}'")

    def _new_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New profile", "Name:")
        if not ok or not name.strip():
            return
        try:
            create_profile(name.strip())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "New profile", str(e))
            return
        self._refresh_profile_combo()
        self.profile_combo.setCurrentText(name.strip())

    def _dup_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Duplicate profile", "New name:", text=f"{self.profile_name}_copy"
        )
        if not ok or not name.strip():
            return
        try:
            duplicate_profile(self.profile_name, name.strip())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Duplicate", str(e))
            return
        self._refresh_profile_combo()
        self.profile_combo.setCurrentText(name.strip())

    def _del_profile(self):
        if len(list_profiles()) <= 1:
            QtWidgets.QMessageBox.warning(self, "Delete", "Cannot delete the last profile")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Delete", f"Delete profile '{self.profile_name}'?"
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            delete_profile(self.profile_name)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Delete", str(e))
            return
        remaining = list_profiles()
        self.profile_name = remaining[0]
        self._refresh_profile_combo()
        self._load_profile_into_form(self.profile_name)
        self.settings["active_profile"] = self.profile_name
        save_app_settings(self.settings)
        sync_legacy_config(self.profile_name)
        self._set_status(f"Loaded profile '{self.profile_name}'")

    def save_all(self):
        self.config = self._collect_form()
        _STATE["config"] = self.config
        save_profile(self.profile_name, self.config)
        self.settings["active_profile"] = self.profile_name
        self.settings["hotkeys"] = self._collect_app_hotkeys()
        self.settings["play_mode"] = _STATE.get("play_mode", "autodrop")
        save_app_settings(self.settings)
        sync_legacy_config(self.profile_name)
        self._register_hotkeys()
        self._set_status(f"Saved '{self.profile_name}'")

    # --- Overlay ---
    def _set_overlay(self, enabled):
        _STATE["overlay_enabled"] = bool(enabled)
        self.overlay_check.blockSignals(True)
        self.overlay_check.setChecked(bool(enabled))
        self.overlay_check.blockSignals(False)
        win = _STATE.get("overlay_win")
        if win is not None:
            if enabled:
                self.config = self._collect_form()
                _STATE["config"] = self.config
                sync_overlay_geometry()
                win.show()
                win.raise_()
            elif not _STATE.get("ghost_cells"):
                win.hide()

    def toggle_overlay(self):
        self._set_overlay(not _STATE["overlay_enabled"])

    def _on_play_mode_radio(self, checked):
        if not checked:
            return
        mode = "suggest" if self.mode_suggest.isChecked() else "autodrop"
        self._set_play_mode(mode)

    def toggle_play_mode(self):
        mode = "autodrop" if _STATE.get("play_mode") == "suggest" else "suggest"
        self._set_play_mode(mode)

    def _set_play_mode(self, mode):
        mode = "suggest" if mode == "suggest" else "autodrop"
        _STATE["play_mode"] = mode
        self.settings["play_mode"] = mode
        save_app_settings(self.settings)
        self.mode_autodrop.blockSignals(True)
        self.mode_suggest.blockSignals(True)
        if mode == "suggest":
            self.mode_suggest.setChecked(True)
            # Suggest needs a visible overlay surface for the ghost
            if not _STATE["overlay_enabled"]:
                self._set_overlay(True)
        else:
            self.mode_autodrop.setChecked(True)
            _STATE["ghost_cells"] = []
            _STATE["ghost_label"] = ""
        self.mode_autodrop.blockSignals(False)
        self.mode_suggest.blockSignals(False)
        if self.worker is not None:
            self.worker.set_play_mode(mode)
        self._set_status(f"Play mode: {mode}")

    # --- Calibration capture ---
    def _begin_capture(self, field_key):
        if self._capturing:
            return
        if self._is_running():
            QtWidgets.QMessageBox.information(
                self, "Capture", "Stop the bot before calibrating."
            )
            return
        self._capturing = True
        self._capture_field = field_key
        self._countdown = CALIBRATION_COUNTDOWN_SEC
        self._set_status(
            f"Capturing {field_key} in {self._countdown}s — hover the target"
        )
        self._capture_timer.start()

    def _capture_tick(self):
        self._countdown -= 1
        if self._countdown > 0:
            self._set_status(
                f"Capturing {self._capture_field} in {self._countdown}s — hover the target"
            )
            return
        self._capture_timer.stop()
        pos = pyautogui.position()
        ax, ay = int(pos[0]), int(pos[1])
        field = self._capture_field
        self._capturing = False
        self._capture_field = None

        mon = monitor_containing(ax, ay)
        # Board TL defines the capture monitor; other points stay relative to it.
        warn = ""
        if field == "board_top_left":
            self.config["screen_offset"] = [mon["left"], mon["top"]]
            self.config["screen_resolution"] = [mon["width"], mon["height"]]
            ox, oy = mon["left"], mon["top"]
            self.screen_label.setText(
                f"Screen: offset=({ox},{oy}) size={mon['width']}x{mon['height']}"
            )
        else:
            ox, oy = self.config.get("screen_offset", [0, 0])
            sw, sh = self.config.get("screen_resolution", [0, 0])
            if not (ox <= ax < ox + sw and oy <= ay < oy + sh):
                warn = " — WARNING: outside profile monitor; re-capture Board top-left"

        rel = [ax - ox, ay - oy]
        self.config[field] = rel
        self.point_edits[field].set(rel)
        _STATE["config"] = self.config
        # Persist immediately so a crash/close doesn't lose the point
        save_profile(self.profile_name, self._collect_form())
        sync_legacy_config(self.profile_name)
        sync_overlay_geometry()
        self._update_monitor_hint()
        self._set_status(
            f"Captured {field} = {rel} (abs {ax},{ay} on mon {mon['left']},{mon['top']})"
            f"{warn}"
        )

    # --- Bot run ---
    def start_bot(self):
        if self._is_running() or self._capturing:
            return
        self.config = self._collect_form()
        self.config["play_mode"] = _STATE.get("play_mode", "autodrop")
        _STATE["config"] = self.config
        save_profile(self.profile_name, self.config)
        sync_legacy_config(self.profile_name)

        self.worker = BotWorker(dict(self.config), self)
        self.worker.stopped.connect(self._on_bot_stopped)
        self.worker.errored.connect(self._on_bot_error)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        mode = _STATE.get("play_mode", "autodrop")
        # Show overlay on the UI thread (never from BotWorker).
        if mode == "suggest" or _STATE.get("overlay_enabled"):
            win = _STATE.get("overlay_win")
            if win is not None:
                sync_overlay_geometry()
                win.show()
        self._set_status(f"Running ({mode})")
        self.worker.start()

    def stop_bot(self):
        if self.worker is not None:
            self.worker.request_stop()
            self._set_status("Stopping…")
            # If cooperative stop wedges inside a long place/CC call, force-quit.
            QtCore.QTimer.singleShot(1500, self._force_stop_bot)

    def _force_stop_bot(self):
        w = self.worker
        if w is None or not w.isRunning():
            return
        print("Stop timed out — force-terminating bot thread", flush=True)
        try:
            if w.bot is not None:
                w.bot.stop()
                try:
                    w.bot.close()
                except Exception:
                    pass
        except Exception:
            pass
        w.terminate()
        w.wait(1000)
        self._on_bot_stopped()

    def _on_bot_stopped(self):
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("Idle")

    def _on_bot_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Bot error", msg)

    # --- Global hotkeys ---
    def _clear_hotkeys(self):
        try:
            import keyboard
            for h in self._hotkey_hooks:
                try:
                    keyboard.remove_hotkey(h)
                except Exception:
                    pass
        except ImportError:
            pass
        self._hotkey_hooks = []

    def _register_hotkeys(self):
        self._clear_hotkeys()
        try:
            import keyboard
        except ImportError:
            self._set_status("Idle (install keyboard for global hotkeys)")
            return
        hotkeys = self._collect_app_hotkeys()
        mapping = {
            "start_bot": lambda: self.hotkey_start.emit(),
            "stop_bot": lambda: self.hotkey_stop.emit(),
            "toggle_overlay": lambda: self.hotkey_overlay.emit(),
            "toggle_play_mode": lambda: self.hotkey_play_mode.emit(),
        }
        for action, key in hotkeys.items():
            cb = mapping.get(action)
            if not cb or not key:
                continue
            try:
                handle = keyboard.add_hotkey(key, cb, suppress=False)
                self._hotkey_hooks.append(handle)
            except Exception as e:
                print(f"Hotkey '{key}' for {action} failed: {e}", flush=True)

    def closeEvent(self, event):
        if self._is_running():
            self.stop_bot()
            if self.worker is not None:
                self.worker.wait(3000)
        self._clear_hotkeys()
        event.accept()
        # Quit the shared Qt app (overlay + panel)
        QtWidgets.QApplication.instance().quit()


def _make_panel():
    return ControlPanel()


def main():
    # OverlayLib sizes to the primary monitor only; ControlPanel._after_show
    # and toggle re-pin it to the profile's game monitor.
    overlay = overlay_lib.Overlay(
        drawlistCallback=drawlist_callback,
        guiWindow=_make_panel,
        refreshTimeout=50,
    )
    _STATE["overlay_win"] = overlay.overlay
    sync_overlay_geometry()
    overlay.spawn()


if __name__ == "__main__":
    main()
