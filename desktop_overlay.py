"""Translucent desktop captions, isolated from the audio processing loop."""

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from ui_theme import COLORS, install_theme
from desktop_helpers import caption_geometry, work_areas
from app_settings import POSITIONS


class SpeechOverlay:
    """Send captions to a separate process whose main thread owns Tk."""

    def __init__(self, captions_only=False, caption_position="Top left"):
        self.captions_only = captions_only
        self.closed = False
        self.process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)] + (["--captions-only"] if captions_only else []) + ["--position", caption_position],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ready = queue.Queue()
        self.controls = queue.Queue()
        def read_events():
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "ready":
                    ready.put(True)
                elif event.get("event") == "control":
                    self.controls.put(event.get("action"))
            if not captions_only:
                self.controls.put("quit")
        self.reader = threading.Thread(target=read_events, daemon=True)
        self.reader.start()
        try:
            if not ready.get(timeout=5):
                raise RuntimeError("Desktop overlay could not start")
        except Exception:
            self.close()
            raise

    def _send(self, command):
        try:
            self.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # A closed caption window must not interrupt voice playback.

    def show(self, text, duration=None):
        self._send({"action": "show", "text": text, "duration": duration})

    def position(self, position):
        self._send({"action": "position", "position": position})

    def hide(self):
        self._send({"action": "hide"})

    def status(self, state, detail=""):
        self._send({"action": "status", "state": state, "detail": detail})

    def poll_controls(self):
        actions = []
        while True:
            try:
                actions.append(self.controls.get_nowait())
            except queue.Empty:
                return actions

    def close(self):
        if self.closed:
            return
        self._send({"action": "close"})
        self.closed = True
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=3)
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass  # Buffered pipe writes can fail again when an exited child is closed.
        self.reader.join(timeout=0.2)
        try:
            self.process.stdout.close()
        except (OSError, ValueError):
            pass


def run_overlay(demo=False, captions_only=False, position="Top left"):
    caption_position = position
    root = tk.Tk()
    root.withdraw()
    install_theme(root)
    root.title("Gemma Live captions")
    root.overrideredirect(True)
    root.configure(bg=COLORS["panel"])
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.82)
    width = min(640, max(240, root.winfo_screenwidth() - 48))
    caption = tk.Label(root, text="", font=("Segoe UI", 18),
                       fg=COLORS["text"], bg=COLORS["panel"], justify="left",
                       anchor="nw", wraplength=width - 40, padx=20, pady=16)
    caption.pack(fill="both", expand=True)
    commands = queue.Queue()
    hide_timer = None
    panel = tk.Toplevel(root)
    if captions_only:
        panel.withdraw()
    panel.title("Gemma Live")
    panel.configure(bg=COLORS["panel"], padx=16, pady=12)
    panel.attributes("-topmost", True)
    panel.geometry("520x185+24+300")
    state_text = tk.StringVar(value="Starting")
    detail_text = tk.StringVar(value="Local voice and screen assistant")
    tk.Label(panel, textvariable=state_text, font=("Segoe UI", 16, "bold"),
             bg=COLORS["panel"], fg=COLORS["accent"]).pack(anchor="w")
    tk.Label(panel, textvariable=detail_text, font=("Segoe UI", 10), wraplength=380,
             justify="left", bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", pady=(4, 10))
    buttons = tk.Frame(panel, bg=COLORS["panel"])
    buttons.pack(anchor="w")

    def control(action):
        print(json.dumps({"event": "control", "action": action}), flush=True)

    pause = ttk.Button(buttons, text="Pause listening", style="Primary.TButton",
                      command=lambda: control("resume" if state_text.get() in ("Paused", "Error") else "pause"))
    pause.pack(side="left", padx=(0, 8))
    cancel = ttk.Button(buttons, text="Stop reply", command=lambda: control("cancel"), state="disabled")
    cancel.pack(side="left", padx=(0, 8))
    ttk.Button(buttons, text="Quit", style="Quiet.TButton", command=lambda: control("quit")).pack(side="left")
    panel.protocol("WM_DELETE_WINDOW", lambda: control("quit"))
    panel.update_idletasks()
    if os.name == "nt" and not demo:
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [ctypes.c_void_p]
        user32.GetParent.restype = ctypes.c_void_p
        user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user32.SetWindowDisplayAffinity(user32.GetParent(panel.winfo_id()), 0x11)

    def hide():
        nonlocal hide_timer
        if hide_timer is not None:
            root.after_cancel(hide_timer)
            hide_timer = None
        root.withdraw()

    def show(text, duration=None):
        nonlocal hide_timer
        hide()
        if not text.strip():
            return
        caption.configure(text=text)
        root.update_idletasks()
        height = caption.winfo_reqheight()
        x, y, w, h = caption_geometry(caption_position, width, height, work_areas(root)[0])
        root.geometry(f"{w}x{h}{x:+d}{y:+d}")
        if os.name == "nt":
            # Keep captions out of the taskbar and let clicks reach the app below.
            user32 = ctypes.windll.user32
            user32.GetParent.argtypes = [ctypes.c_void_p]
            user32.GetParent.restype = ctypes.c_void_p
            hwnd = user32.GetParent(root.winfo_id())
            user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
            style = user32.GetWindowLongW(hwnd, -20)
            user32.SetWindowLongW(hwnd, -20, style | 0x08000000 | 0x20 | 0x80)
            # Avoid feeding the assistant's captions back into screen capture.
            user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            if not demo:
                user32.SetWindowDisplayAffinity(hwnd, 0x11)
        root.deiconify()
        if duration is not None:
            hide_timer = root.after(int(duration * 1000), hide)

    def read_commands():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        commands.put({"action": "close"})

    def poll():
        nonlocal caption_position
        try:
            while True:
                command = commands.get_nowait()
                action = command.get("action")
                if action == "close":
                    root.destroy()
                    return
                if action == "position" and command.get("position") in POSITIONS:
                    caption_position = command["position"]
                    if root.winfo_ismapped():
                        x, y, w, h = caption_geometry(caption_position, width, root.winfo_height(), work_areas(root)[0])
                        root.geometry(f"{w}x{h}{x:+d}{y:+d}")
                elif action == "hide":
                    hide()
                elif action == "show":
                    show(command.get("text", ""), command.get("duration"))
                elif action == "status":
                    if captions_only:
                        continue
                    state = command.get("state", "Starting")
                    state_text.set(state)
                    detail_text.set(command.get("detail", ""))
                    pause.configure(text="Resume listening" if state in ("Paused", "Error") else "Pause listening")
                    cancel.configure(state="normal" if state in ("Thinking", "Preparing voice", "Speaking") else "disabled")
        except queue.Empty:
            pass
        root.after(30, poll)

    if demo:
        show("Hello! This is where Gemma's spoken replies appear.\n"
             "Captions stay above your apps while the assistant speaks.")
        root.after(12000, root.destroy)
    else:
        threading.Thread(target=read_commands, daemon=True).start()
        root.after(30, poll)
        print(json.dumps({"event": "ready"}), flush=True)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma desktop captions")
    parser.add_argument("--demo", action="store_true", help="Preview captions for 12 seconds")
    parser.add_argument("--captions-only", action="store_true")
    parser.add_argument("--position", choices=POSITIONS, default="Top left")
    args = parser.parse_args()
    run_overlay(args.demo, args.captions_only, args.position)
