"""X11 window selection and geometry (absolute screen coordinates)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from Xlib import X, display


@dataclass
class WindowInfo:
    x: int
    y: int
    width: int
    height: int
    window_id: int


def _try_xdotool() -> int | None:
    try:
        r = subprocess.run(
            ["xdotool", "selectwindow"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if r.returncode != 0:
            return None
        wid = int(r.stdout.strip())
        return wid
    except (FileNotFoundError, ValueError):
        return None


def _client_window(dpy: display.Display, win) -> int:
    """Walk from subwindow toward a reasonable top-level client window."""
    root = dpy.screen().root
    wm_state = dpy.intern_atom("WM_STATE")
    while win != root:
        try:
            p = win.get_full_property(wm_state, wm_state)
            if p is not None:
                return win.id
        except Exception:
            pass
        parent = win.query_tree().parent
        if parent == win:
            break
        win = parent
    return win.id


def _geometry_for_window(dpy: display.Display, wid: int) -> WindowInfo:
    win = dpy.create_resource_object("window", wid)
    geom = win.get_geometry()
    tr = win.translate_coords(dpy.screen().root, 0, 0)
    abs_x, abs_y = tr.x, tr.y
    w, h = geom.width, geom.height
    return WindowInfo(x=abs_x, y=abs_y, width=w, height=h, window_id=wid)


def pick_window_x11() -> WindowInfo | None:
    """Let user pick a window; return geometry or None if cancelled."""
    wid = _try_xdotool()
    if wid is not None:
        info = geometry_after_xdotool(wid)
        return info
    dpy = display.Display()
    wid = _pick_with_grab(dpy)
    if wid is None:
        return None
    try:
        return _geometry_for_window(dpy, wid)
    except Exception:
        return None


def _pick_with_grab(dpy: display.Display) -> int | None:
    root = dpy.screen().root
    root.grab_pointer(
        True,
        X.ButtonPressMask | X.ButtonReleaseMask,
        X.GrabModeAsync,
        X.GrabModeAsync,
        X.NONE,
        X.NONE,
        X.CurrentTime,
    )
    try:
        while True:
            ev = dpy.next_event()
            if ev.type == X.ButtonPress:
                target = ev.child if ev.child != X.NONE else ev.window
                if target == X.NONE or target == root:
                    dpy.ungrab_pointer(X.CurrentTime)
                    return None
                wid = _client_window(dpy, dpy.create_resource_object("window", target.id))
                dpy.ungrab_pointer(X.CurrentTime)
                return int(wid)
    except KeyboardInterrupt:
        try:
            dpy.ungrab_pointer(X.CurrentTime)
        except Exception:
            pass
        return None


def parse_xwininfo_window_id(text: str) -> int | None:
    m = re.search(r"Window id:\s*(0x[0-9a-fA-F]+|\d+)", text)
    if not m:
        return None
    s = m.group(1)
    return int(s, 0)


def window_info_from_xwininfo(wid: int) -> WindowInfo | None:
    try:
        r = subprocess.run(
            ["xwininfo", "-id", hex(wid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return None
        return parse_xwininfo_geometry(r.stdout, wid)
    except FileNotFoundError:
        return None


def parse_xwininfo_geometry(text: str, wid: int) -> WindowInfo | None:
    abs_x = abs_y = width = height = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Absolute upper-left X:"):
            abs_x = int(line.split(":", 1)[1].strip())
        elif line.startswith("Absolute upper-left Y:"):
            abs_y = int(line.split(":", 1)[1].strip())
        elif line.startswith("Width:"):
            width = int(line.split(":", 1)[1].strip())
        elif line.startswith("Height:"):
            height = int(line.split(":", 1)[1].strip())
    if None in (abs_x, abs_y, width, height):
        return None
    return WindowInfo(x=abs_x, y=abs_y, width=width, height=height, window_id=wid)


def geometry_after_xdotool(wid: int) -> WindowInfo | None:
    info = window_info_from_xwininfo(wid)
    if info:
        return info
    dpy = display.Display()
    try:
        return _geometry_for_window(dpy, wid)
    except Exception:
        return None
