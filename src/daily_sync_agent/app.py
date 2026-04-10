"""System tray UI: window pick, recording, audio devices, post-process AI."""

from __future__ import annotations

import datetime as dt
import logging
import os
import traceback
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSystemTrayIcon,
    QWidget,
)

from daily_sync_agent.ai.pipeline import run_transcribe_and_summarize
from daily_sync_agent.ai.summarize import list_ollama_models, unload_ollama_models
from daily_sync_agent.ai.whisper_compute import compute_type_choices, pick_compute_index_for_value
from daily_sync_agent.ai.whisper_models import list_whisper_models_for_combo, normalize_whisper_device
from daily_sync_agent.audio.devices import AudioDevices, AudioMode, list_devices
from daily_sync_agent.capture.desktop_clip import clip_window_info_to_visible_desktop
from daily_sync_agent.capture.ffmpeg import FfmpegPaths, RecordingProcess, build_ffmpeg_command, start_recording
from daily_sync_agent.capture.window_x11 import WindowInfo, pick_window_x11
from daily_sync_agent.icons import icon_idle, icon_recording
from daily_sync_agent.settings import AppConfig, log_dir, output_dir
from daily_sync_agent.ui.coordinate_map import build_screen_coordinate_maps, map_native_rect_to_logical
from daily_sync_agent.ui.window_frame_overlay import WindowFrameOverlay

logger = logging.getLogger(__name__)


def _is_shift_pressed_now() -> bool:
    """Best-effort Shift state detection for tray activation on Linux/X11 shells."""
    mods = QApplication.keyboardModifiers() | QApplication.queryKeyboardModifiers()
    if mods & Qt.KeyboardModifier.ShiftModifier:
        return True

    # Some tray hosts do not forward modifier state to Qt; read X11 keymap directly.
    try:
        from Xlib import XK, display as xdisplay

        dpy = xdisplay.Display()
        try:
            keymap = dpy.query_keymap()

            def down(keysym_name: str) -> bool:
                keycode = dpy.keysym_to_keycode(XK.string_to_keysym(keysym_name))
                if keycode <= 0:
                    return False
                return bool(keymap[keycode >> 3] & (1 << (keycode & 7)))

            return down("Shift_L") or down("Shift_R")
        finally:
            dpy.close()
    except Exception as e:
        logger.debug("Could not read X11 Shift state for tray activation: %s", e)
        return False


def _fill_whisper_model_combo(combo: QComboBox, current_model: str) -> None:
    cur = (current_model or "").strip()
    combo.clear()
    for n in list_whisper_models_for_combo():
        combo.addItem(n, n)
    if cur and combo.findData(cur) < 0:
        combo.addItem(f"{cur} (not listed)", cur)
    idx = combo.findData(cur)
    combo.setCurrentIndex(idx if idx >= 0 else 0)


def _fill_whisper_compute_combo(combo: QComboBox, device_key: str, saved_compute: str) -> None:
    dk = device_key if device_key in ("auto", "cpu", "cuda") else normalize_whisper_device(device_key)
    combo.clear()
    for label, val in compute_type_choices(dk):
        combo.addItem(label, val)
    combo.setCurrentIndex(pick_compute_index_for_value(dk, saved_compute))


def _fill_ollama_model_combo(combo: QComboBox, base_url: str, current_model: str) -> None:
    """Populate combo: first entry = use first available model (empty string)."""
    cur = (current_model or "").strip()
    combo.clear()
    combo.addItem("— First available —", "")
    base = (base_url or "").strip() or "http://127.0.0.1:11434"
    for n in list_ollama_models(base):
        combo.addItem(n, n)
    if cur and combo.findData(cur) < 0:
        combo.addItem(f"{cur} (not on server)", cur)
    idx = combo.findData(cur)
    combo.setCurrentIndex(idx if idx >= 0 else 0)


class AiThread(QThread):
    """Emits the session directory so the tray message matches the job (not the latest recording)."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QWidget, audio_path: Path, session_dir: Path, config: AppConfig) -> None:
        super().__init__(parent)
        self._audio_path = audio_path
        self._session_dir = session_dir
        self._config = config

    def run(self) -> None:
        try:
            logger.debug("AI pipeline start audio=%s session=%s", self._audio_path, self._session_dir)
            t, s = run_transcribe_and_summarize(self._audio_path, self._session_dir, self._config)
            logger.debug("AI pipeline done transcript=%s summary=%s", t, s)
            self.finished_ok.emit(self._session_dir)
        except Exception:
            logger.exception("AI pipeline failed")
            self.failed.emit(traceback.format_exc())


class TrayApplication(QWidget):
    def __init__(self, *, debug: bool = False, debug_log_path: Path | None = None) -> None:
        super().__init__()
        self._debug = debug
        self._debug_log_path = debug_log_path
        self._config = AppConfig.load()
        self._tray = QSystemTrayIcon(icon_idle(), self)
        self._tray.setToolTip("Daily sync agent")
        self._selected_window: WindowInfo | None = None
        try:
            self._devices = list_devices()
        except Exception as e:
            self._devices = AudioDevices(
                default_sink="",
                default_source="",
                sinks=[],
                sources=[],
            )
            self._devices_error = str(e)
        else:
            self._devices_error = ""
        self._playback_override: str | None = None
        self._recording_override: str | None = None
        self._mode = AudioMode.MIX
        self._recording: RecordingProcess | None = None
        self._session_dir: Path | None = None
        self._ai_thread: AiThread | None = None
        self._ai_queue: list[tuple[Path, Path]] = []
        self._frame_preview: WindowFrameOverlay | None = None

        self._restore_last_capture_from_config()

        self._menu = QMenu()
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self._tray.setContextMenu(self._menu)
        self._tray.activated.connect(self._on_activated)
        self._rebuild_menu()

    def wait_for_ai_thread(self, timeout_ms: int = 120_000) -> None:
        """Block until the AI worker and queued jobs finish (e.g. on application exit)."""
        import time

        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            QApplication.processEvents()
            if self._ai_thread is not None and self._ai_thread.isRunning():
                self._ai_thread.wait(100)
            elif self._ai_queue:
                self._try_start_ai_worker()
            else:
                break
            if time.monotonic() >= deadline:
                if self._ai_queue or (
                    self._ai_thread is not None and self._ai_thread.isRunning()
                ):
                    logger.warning(
                        "AI work not fully finished on exit (queue=%s running=%s)",
                        len(self._ai_queue),
                        self._ai_thread.isRunning() if self._ai_thread else False,
                    )
                break

    def shutdown(self) -> None:
        """Best-effort cleanup before process exit."""
        logger.debug("Application shutdown started; draining AI workers before cleanup")
        self.wait_for_ai_thread()
        logger.debug("Application shutdown cleanup: requesting Ollama unload/stop")
        unload_ollama_models(
            self._config.ollama_base_url,
            preferred_model=self._config.ollama_model,
            stop_cli=True,
        )
        logger.debug("Application shutdown cleanup finished")

    def show(self) -> None:  # noqa: A003
        self._tray.show()
        if self._debug and self._debug_log_path:
            logger.info("Debug log file: %s", self._debug_log_path)
            self._tray.setToolTip(f"Daily sync agent (debug → {self._debug_log_path.name})")
        if self._devices_error:
            logger.warning("Audio device query failed: %s", self._devices_error)
            self._tray.showMessage(
                "Audio",
                f"Could not query audio devices (pactl): {self._devices_error}",
                QSystemTrayIcon.MessageIcon.Warning,
                8000,
            )

    def _restore_last_capture_from_config(self) -> None:
        c = self._config
        if c.last_capture_x is None or c.last_capture_y is None or c.last_capture_w is None or c.last_capture_h is None:
            return
        w, h = c.last_capture_w, c.last_capture_h
        if w <= 0 or h <= 0:
            return
        wid = c.last_capture_window_id if c.last_capture_window_id is not None else 0
        self._selected_window = WindowInfo(
            x=c.last_capture_x,
            y=c.last_capture_y,
            width=w,
            height=h,
            window_id=wid,
        )

    def _persist_last_capture(self, info: WindowInfo) -> None:
        self._config.last_capture_x = info.x
        self._config.last_capture_y = info.y
        self._config.last_capture_w = info.width
        self._config.last_capture_h = info.height
        self._config.last_capture_window_id = info.window_id
        self._config.save()

    def _show_saved_frame_preview(self) -> None:
        if self._selected_window is None:
            self._tray.showMessage(
                "Capture region",
                "No window saved yet — use “Select window…” in the tray menu.",
                QSystemTrayIcon.MessageIcon.Information,
                5000,
            )
            return
        info = self._selected_window
        native_rect = QRect(info.x, info.y, info.width, info.height)
        rect = map_native_rect_to_logical(native_rect, build_screen_coordinate_maps())
        logger.debug("Preview overlay rect native=%s logical=%s", native_rect, rect)
        prev = self._frame_preview
        overlay = WindowFrameOverlay(rect, parent=None)
        overlay.destroyed.connect(lambda o=overlay: self._on_frame_preview_destroyed(o))
        self._frame_preview = overlay
        overlay.show_and_expire()
        # Previous overlay may already be gone (expiry timer); wrapper can outlive the C++ object.
        if prev is not None and isValid(prev):
            try:
                prev.deleteLater()
            except RuntimeError:
                pass

    def _on_frame_preview_destroyed(self, overlay: WindowFrameOverlay) -> None:
        if self._frame_preview is overlay:
            self._frame_preview = None

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason not in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            return
        if _is_shift_pressed_now():
            self._toggle_recording_from_tray()
            return
        self._show_saved_frame_preview()

    def _toggle_recording_from_tray(self) -> None:
        if self._recording is not None:
            self._stop_recording()
            return
        if self._selected_window is None:
            self._tray.showMessage(
                "Recording",
                "No window saved yet — use “Select window…” in the tray menu.",
                QSystemTrayIcon.MessageIcon.Information,
                5000,
            )
            return
        self._start_recording()

    def _rebuild_menu(self) -> None:
        try:
            self._devices = list_devices()
            self._devices_error = ""
        except Exception as e:
            self._devices_error = str(e)
            logger.debug("list_devices on menu open: %s", e)

        self._menu.clear()

        act_window = QAction("Select window…", self)
        act_window.triggered.connect(self._select_window)
        self._menu.addAction(act_window)

        act_start = QAction("Start recording", self)
        act_start.setEnabled(self._selected_window is not None and self._recording is None)
        act_start.triggered.connect(self._start_recording)
        self._menu.addAction(act_start)

        act_stop = QAction("Stop recording", self)
        act_stop.setEnabled(self._recording is not None)
        act_stop.triggered.connect(self._stop_recording)
        self._menu.addAction(act_stop)

        mode_menu = self._menu.addMenu("Audio capture mode")
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        for mode, label in [
            (AudioMode.MIX, "Mix (playback monitor + microphone)"),
            (AudioMode.MONITOR, "Playback / system output (monitor)"),
            (AudioMode.MIC, "Microphone only"),
        ]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(self._mode == mode)
            mode_group.addAction(a)
            a.triggered.connect(lambda checked=False, m=mode: self._set_mode(m))
            mode_menu.addAction(a)

        pb_menu = self._menu.addMenu("Playback device (for monitor)")
        pb_group = QActionGroup(self)
        pb_group.setExclusive(True)
        act_def_pb = QAction("Use default sink", self)
        act_def_pb.setCheckable(True)
        act_def_pb.setChecked(self._playback_override is None)
        pb_group.addAction(act_def_pb)
        act_def_pb.triggered.connect(lambda: self._set_playback(None))
        pb_menu.addAction(act_def_pb)
        for name, desc in self._devices.sinks:
            a = QAction(f"{desc} ({name})", self)
            a.setCheckable(True)
            a.setChecked(self._playback_override == name)
            pb_group.addAction(a)
            a.triggered.connect(lambda checked=False, n=name: self._set_playback(n))
            pb_menu.addAction(a)

        rec_menu = self._menu.addMenu("Recording device (microphone)")
        rec_group = QActionGroup(self)
        rec_group.setExclusive(True)
        act_def_src = QAction("Use default source", self)
        act_def_src.setCheckable(True)
        act_def_src.setChecked(self._recording_override is None)
        rec_group.addAction(act_def_src)
        act_def_src.triggered.connect(lambda: self._set_record_src(None))
        rec_menu.addAction(act_def_src)
        for name, desc in self._devices.sources:
            a = QAction(f"{desc} ({name})", self)
            a.setCheckable(True)
            a.setChecked(self._recording_override == name)
            rec_group.addAction(a)
            a.triggered.connect(lambda checked=False, n=name: self._set_record_src(n))
            rec_menu.addAction(a)

        self._menu.addSeparator()
        act_folder = QAction("Open recordings folder", self)
        act_folder.triggered.connect(self._open_folder)
        self._menu.addAction(act_folder)

        act_prefs = QAction("Preferences…", self)
        act_prefs.triggered.connect(self._preferences)
        self._menu.addAction(act_prefs)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(QApplication.quit)
        self._menu.addAction(act_quit)

    def _set_mode(self, mode: AudioMode) -> None:
        self._mode = mode
        self._rebuild_menu()

    def _set_playback(self, name: str | None) -> None:
        self._playback_override = name
        self._rebuild_menu()

    def _set_record_src(self, name: str | None) -> None:
        self._recording_override = name
        self._rebuild_menu()

    def _select_window(self) -> None:
        self._tray.showMessage("Window", "Click a window to record (crosshair or xdotool).", QSystemTrayIcon.MessageIcon.Information, 4000)
        QApplication.processEvents()
        info = pick_window_x11()
        if info:
            self._selected_window = info
            self._persist_last_capture(info)
            self._tray.showMessage("Window", f"Selected {info.width}x{info.height} at ({info.x},{info.y})", QSystemTrayIcon.MessageIcon.Information, 4000)
        self._rebuild_menu()

    def _display_str(self) -> str:
        return os.environ.get("DISPLAY", self._config.display)

    def _start_recording(self) -> None:
        if self._recording is not None or self._selected_window is None:
            return
        win = clip_window_info_to_visible_desktop(self._selected_window)
        if win is None:
            QMessageBox.warning(
                None,
                "Recording",
                "The selected window is completely outside the visible desktop. Move it on-screen and try again.",
            )
            return
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        session = output_dir() / stamp
        session.mkdir(parents=True, exist_ok=True)
        self._session_dir = session
        video_path = session / "recording.mkv"
        audio_path = session / "recording.flac"
        paths = FfmpegPaths(video_path=video_path, audio_only_path=audio_path)
        if (
            win.x != self._selected_window.x
            or win.y != self._selected_window.y
            or win.width != self._selected_window.width
            or win.height != self._selected_window.height
        ):
            logger.info(
                "Clipped capture to visible desktop: %sx%s@%s,%s (was %sx%s@%s,%s)",
                win.width,
                win.height,
                win.x,
                win.y,
                self._selected_window.width,
                self._selected_window.height,
                self._selected_window.x,
                self._selected_window.y,
            )
        try:
            cmd = build_ffmpeg_command(
                win,
                display=self._display_str(),
                fps=self._config.ffmpeg_fps,
                mode=self._mode,
                devices=self._devices,
                playback_sink=self._playback_override,
                recording_source=self._recording_override,
                out=paths,
                ffmpeg_loglevel="verbose" if self._debug else "info",
            )
        except Exception as e:
            logger.exception("build_ffmpeg_command failed")
            QMessageBox.warning(None, "ffmpeg", str(e))
            return

        logger.info("Starting recording session=%s", session)
        logger.debug("ffmpeg argv: %s", cmd)

        log_path = log_dir() / f"ffmpeg-{stamp}.log"
        try:
            self._recording = start_recording(cmd, log_path)
        except FileNotFoundError:
            QMessageBox.warning(
                None,
                "ffmpeg",
                "ffmpeg executable not found. Install ffmpeg and ensure it is on PATH.",
            )
            return
        except Exception as e:
            QMessageBox.warning(None, "Recording", str(e))
            return

        self._tray.setIcon(icon_recording())
        self._tray.setToolTip("Recording…")
        self._rebuild_menu()

    def _stop_recording(self) -> None:
        if self._recording is None:
            return
        proc = self._recording
        self._recording = None
        proc.terminate()
        exit_code = proc.process.returncode
        logger.info("Recording stopped exit_code=%s ffmpeg_log=%s", exit_code, proc.log_path)
        self._tray.setIcon(icon_idle())
        tip = "Daily sync agent"
        if self._debug and self._debug_log_path:
            tip = f"Daily sync agent (debug → {self._debug_log_path.name})"
        self._tray.setToolTip(tip)

        session = self._session_dir
        if session is None:
            self._rebuild_menu()
            return

        audio_path = session / "recording.flac"
        if not audio_path.is_file():
            self._tray.showMessage(
                "Recording",
                f"Audio file missing (ffmpeg exit {exit_code}). Log: {proc.log_path}",
                QSystemTrayIcon.MessageIcon.Warning,
                8000,
            )
            self._rebuild_menu()
            return
        if exit_code not in (0, 255) and exit_code is not None:
            self._tray.showMessage(
                "Recording",
                f"ffmpeg reported exit {exit_code}; continuing if outputs exist. Log: {proc.log_path}",
                QSystemTrayIcon.MessageIcon.Warning,
                6000,
            )

        self._enqueue_ai_pipeline(audio_path, session)

    def _enqueue_ai_pipeline(self, audio_path: Path, session: Path) -> None:
        self._ai_queue.append((audio_path, session))
        n = len(self._ai_queue)
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self._tray.showMessage(
                "Queued",
                f"Transcription will start after the current job ({n} in queue).",
                QSystemTrayIcon.MessageIcon.Information,
                6000,
            )
        self._try_start_ai_worker()

    def _try_start_ai_worker(self) -> None:
        if self._ai_thread is not None and self._ai_thread.isRunning():
            return
        if not self._ai_queue:
            return
        audio_path, session = self._ai_queue.pop(0)
        self._tray.showMessage(
            "Processing",
            "Transcribing and summarizing (local AI)…",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )
        self._ai_thread = AiThread(self, audio_path, session, self._config)
        self._ai_thread.finished_ok.connect(self._on_ai_ok)
        self._ai_thread.failed.connect(self._on_ai_fail)
        self._ai_thread.start()
        self._rebuild_menu()

    def _on_ai_ok(self, session_dir: object) -> None:
        self._ai_thread = None
        sd = session_dir if isinstance(session_dir, Path) else Path(session_dir)
        self._tray.showMessage(
            "Done",
            f"Saved transcript and summary in {sd}",
            QSystemTrayIcon.MessageIcon.Information,
            8000,
        )
        self._rebuild_menu()
        self._try_start_ai_worker()

    def _on_ai_fail(self, err: str) -> None:
        self._ai_thread = None
        QMessageBox.critical(None, "AI pipeline failed", err[:4000])
        self._rebuild_menu()
        self._try_start_ai_worker()

    def _open_folder(self) -> None:
        path = output_dir()
        QDesktopServices_open(path)

    def _preferences(self) -> None:
        d = QDialog(self)
        d.setWindowTitle("Preferences")
        lay = QFormLayout(d)

        cfg = self._config
        if (
            cfg.prefs_window_x is not None
            and cfg.prefs_window_y is not None
            and cfg.prefs_window_w is not None
            and cfg.prefs_window_h is not None
        ):
            d.setGeometry(
                QRect(cfg.prefs_window_x, cfg.prefs_window_y, cfg.prefs_window_w, cfg.prefs_window_h)
            )

        def persist_window_geometry() -> None:
            g = d.geometry()
            self._config.prefs_window_x = g.x()
            self._config.prefs_window_y = g.y()
            self._config.prefs_window_w = g.width()
            self._config.prefs_window_h = g.height()
            self._config.save()

        d.finished.connect(lambda *_: persist_window_geometry())

        e_ollama = QLineEdit(cfg.ollama_base_url)
        combo_model = QComboBox()
        _fill_ollama_model_combo(combo_model, e_ollama.text(), cfg.ollama_model)

        row_model = QWidget()
        row_ml = QHBoxLayout(row_model)
        row_ml.setContentsMargins(0, 0, 0, 0)
        row_ml.addWidget(combo_model, 1)
        btn_refresh_models = QPushButton("Refresh list")

        def refresh_models() -> None:
            cur = combo_model.currentData()
            keep = cur if isinstance(cur, str) else ""
            _fill_ollama_model_combo(combo_model, e_ollama.text(), keep)

        btn_refresh_models.clicked.connect(refresh_models)
        e_ollama.editingFinished.connect(refresh_models)
        row_ml.addWidget(btn_refresh_models)

        combo_whisper = QComboBox()
        _fill_whisper_model_combo(combo_whisper, cfg.whisper_model)
        row_whisper = QWidget()
        row_wl = QHBoxLayout(row_whisper)
        row_wl.setContentsMargins(0, 0, 0, 0)
        row_wl.addWidget(combo_whisper, 1)
        btn_whisper_refresh = QPushButton("Refresh list")

        def refresh_whisper_models() -> None:
            keep = combo_whisper.currentData()
            _fill_whisper_model_combo(combo_whisper, keep if isinstance(keep, str) else "")

        btn_whisper_refresh.clicked.connect(refresh_whisper_models)
        row_wl.addWidget(btn_whisper_refresh)

        combo_whisper_dev = QComboBox()
        combo_whisper_dev.addItem("Auto-detect", "auto")
        combo_whisper_dev.addItem("CPU", "cpu")
        combo_whisper_dev.addItem("GPU (CUDA)", "cuda")
        wd = normalize_whisper_device(cfg.whisper_device)
        idx_wd = combo_whisper_dev.findData(wd)
        combo_whisper_dev.setCurrentIndex(idx_wd if idx_wd >= 0 else 0)

        combo_whisper_ct = QComboBox()
        _fill_whisper_compute_combo(
            combo_whisper_ct,
            str(combo_whisper_dev.currentData() or "auto"),
            cfg.whisper_compute_type,
        )

        def on_whisper_device_changed(_idx: int) -> None:
            prev = combo_whisper_ct.currentData()
            _fill_whisper_compute_combo(
                combo_whisper_ct,
                str(combo_whisper_dev.currentData() or "auto"),
                str(prev) if isinstance(prev, str) else cfg.whisper_compute_type,
            )

        combo_whisper_dev.currentIndexChanged.connect(on_whisper_device_changed)

        e_fps = QLineEdit(str(cfg.ffmpeg_fps))
        chk_unload_models = QCheckBox("Unload Whisper/Ollama models after each task")
        chk_unload_models.setChecked(cfg.unload_models_after_task)
        combo_summary = QComboBox()
        combo_summary.addItem("General (short paragraph)", "general")
        combo_summary.addItem("Daily scrum (structured)", "daily_scrum")
        sm = (cfg.summary_mode or "general").strip().lower()
        idx_sm = combo_summary.findData(sm)
        combo_summary.setCurrentIndex(idx_sm if idx_sm >= 0 else 0)

        lay.addRow("Ollama base URL", e_ollama)
        lay.addRow("Ollama chat model", row_model)
        lay.addRow("Summary format", combo_summary)
        lay.addRow("Whisper model", row_whisper)
        lay.addRow("Whisper device", combo_whisper_dev)
        lay.addRow("Whisper compute type", combo_whisper_ct)
        lay.addRow("Low-VRAM mode", chk_unload_models)
        lay.addRow("ffmpeg fps", e_fps)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(d.accept)
        buttons.rejected.connect(d.reject)
        lay.addRow(buttons)
        if d.exec() == QDialog.DialogCode.Accepted:
            self._config.ollama_base_url = e_ollama.text().strip() or self._config.ollama_base_url
            md = combo_model.currentData()
            self._config.ollama_model = md.strip() if isinstance(md, str) else ""
            smd = combo_summary.currentData()
            self._config.summary_mode = str(smd) if smd else "general"
            wm = combo_whisper.currentData()
            self._config.whisper_model = wm.strip() if isinstance(wm, str) and wm.strip() else "base"
            wdv = combo_whisper_dev.currentData()
            self._config.whisper_device = str(wdv) if wdv else "auto"
            wct = combo_whisper_ct.currentData()
            self._config.whisper_compute_type = str(wct) if wct else "default"
            self._config.unload_models_after_task = chk_unload_models.isChecked()
            try:
                self._config.ffmpeg_fps = max(1, int(e_fps.text().strip()))
            except ValueError:
                pass
            self._config.save()


def QDesktopServices_open(path: Path) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
