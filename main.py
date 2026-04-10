import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
import threading
import os
import sys
import subprocess
import cv2
from PIL import Image, ImageTk
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
load_dotenv('.env')
BASE_DIR     = os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, 'best_model', 'weights', 'best.pt')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'output')
STUBS_DIR    = os.path.join(BASE_DIR, 'analyzer', 'stubs')
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, 'output.mp4')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STUBS_DIR,  exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0a0e14"
SURFACE     = "#111820"
SURFACE2    = "#1a2332"
ACCENT      = "#00e5ff"
ACCENT2     = "#ff3d71"
TEXT        = "#e8f0fe"
TEXT_DIM    = "#4a6080"
BORDER      = "#1e3048"
SUCCESS     = "#00e096"
WARNING     = "#ffaa00"

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def bgr_to_hex(bgr):
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Swatch Button ──────────────────────────────────────────────────────────────
class ColorSwatch(tk.Canvas):
    def __init__(self, parent, color="#ffffff", size=32, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=SURFACE2, highlightthickness=0, cursor="hand2", **kwargs)
        self.size   = size
        self._color = color
        self._draw()
        self.bind("<Button-1>", self._pick)

    def _draw(self):
        self.delete("all")
        r = self.size // 2 - 2
        cx = cy = self.size // 2
        self.create_oval(cx-r, cy-r, cx+r, cy+r, fill=self._color, outline=ACCENT, width=1)

    def _pick(self, _=None):
        result = colorchooser.askcolor(color=self._color, title="Pick colour")
        if result and result[1]:
            self._color = result[1]
            self._draw()
            if self._callback:
                self._callback(self._color)

    def set_callback(self, fn):
        self._callback = fn

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, v):
        self._color = v
        self._draw()


# ── Video Player ───────────────────────────────────────────────────────────────
class VideoPlayer(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._cap       = None
        self._playing   = False
        self._after_id  = None
        self._total     = 0
        self._fps       = 25

        # Canvas
        self.canvas = tk.Canvas(self, bg="#050810", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._placeholder()

        # Controls bar
        ctrl = tk.Frame(self, bg=SURFACE, pady=6)
        ctrl.pack(fill="x")

        btn_cfg = dict(bg=SURFACE2, fg=TEXT, relief="flat",
                       activebackground=SURFACE, activeforeground=ACCENT,
                       font=("Courier", 13), bd=0, padx=10, cursor="hand2")

        self.btn_play = tk.Button(ctrl, text="▶", command=self._toggle, **btn_cfg)
        self.btn_play.pack(side="left", padx=(8, 4))

        tk.Button(ctrl, text="⏮", command=self._rewind, **btn_cfg).pack(side="left", padx=4)

        self.slider = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                command=self._seek)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

        self.lbl_time = tk.Label(ctrl, text="0:00 / 0:00",
                                 bg=SURFACE, fg=TEXT_DIM,
                                 font=("Courier", 9))
        self.lbl_time.pack(side="right", padx=10)

    def _placeholder(self):
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()  or 640
        h = self.canvas.winfo_height() or 360
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w, h, fill="#050810")
        self.canvas.create_text(w//2, h//2,
                                text="[ OUTPUT WILL APPEAR HERE ]",
                                fill=TEXT_DIM, font=("Courier", 11))

    def load(self, path):
        if self._cap:
            self._cap.release()
        self._playing = False
        self.btn_play.config(text="▶")
        self._cap   = cv2.VideoCapture(path)
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps   = self._cap.get(cv2.CAP_PROP_FPS) or 25
        self.slider.config(to=max(1, self._total - 1))
        self.slider.set(0)
        self._show_frame(0)

    def _toggle(self):
        if not self._cap:
            return
        self._playing = not self._playing
        self.btn_play.config(text="⏸" if self._playing else "▶")
        if self._playing:
            self._play_loop()

    def _play_loop(self):
        if not self._playing or not self._cap:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._playing = False
            self.btn_play.config(text="▶")
            return
        self._render(frame)
        pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.set(pos)
        self._update_time(pos)
        delay = max(1, int(1000 / self._fps))
        self._after_id = self.after(delay, self._play_loop)

    def _seek(self, val):
        if not self._cap:
            return
        frame_num = int(float(val))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self._show_frame(frame_num)

    def _rewind(self):
        if not self._cap:
            return
        self._playing = False
        self.btn_play.config(text="▶")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.slider.set(0)
        self._show_frame(0)

    def _show_frame(self, pos):
        ret, frame = self._cap.read()
        if ret:
            self._render(frame)
            self._update_time(pos)

    def _render(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()  or 640
        h = self.canvas.winfo_height() or 360
        img = Image.fromarray(frame)
        img.thumbnail((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(w//2, h//2, anchor="center", image=self._photo)

    def _update_time(self, pos):
        def fmt(s): return f"{int(s)//60}:{int(s)%60:02d}"
        cur = fmt(pos / self._fps)
        tot = fmt(self._total / self._fps)
        self.lbl_time.config(text=f"{cur} / {tot}")


# ── Main App ───────────────────────────────────────────────────────────────────
class FootballAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Football Analyzer")
        self.configure(bg=BG)
        self.geometry("1280x800")
        self.minsize(960, 640)

        # Color state  (BGR tuples matching OpenCV)
        self.colors = {
            "team1":    "#00aaff",
            "team2":    "#ff4444",
            "referee":  "#ffff00",
            "ball":     "#00ff88",
            "has_ball": "#ff0000",
        }

        self._build_styles()
        self._build_ui()

    # ── Styles ─────────────────────────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScale",
                         background=SURFACE,
                         troughcolor=SURFACE2,
                         sliderthickness=14,
                         sliderrelief="flat")
        style.configure("Accent.TButton",
                         background=ACCENT, foreground=BG,
                         font=("Courier", 11, "bold"),
                         relief="flat", padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", "#00bcd4")])

    # ── Layout ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ─────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=SURFACE, pady=14)
        header.pack(fill="x")

        tk.Label(header, text="⚽  FOOTBALL ANALYSIS SYSTEM",
                 bg=SURFACE, fg=ACCENT,
                 font=("Courier", 15, "bold")).pack(side="left", padx=20)

        self.status_dot = tk.Label(header, text="●", bg=SURFACE, fg=TEXT_DIM,
                                   font=("Courier", 12))
        self.status_dot.pack(side="right", padx=6)
        self.status_lbl = tk.Label(header, text="IDLE",
                                   bg=SURFACE, fg=TEXT_DIM,
                                   font=("Courier", 10))
        self.status_lbl.pack(side="right", padx=(0, 14))

        # ── Body ───────────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left panel
        left = tk.Frame(body, bg=SURFACE, width=280)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Right panel (video)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_right_panel(right)

    # ── Left Panel ─────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):

        def section(text):
            f = tk.Frame(parent, bg=SURFACE)
            f.pack(fill="x", padx=12, pady=(14, 4))
            tk.Label(f, text=text, bg=SURFACE, fg=ACCENT,
                     font=("Courier", 9, "bold")).pack(anchor="w")
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 8))
            return parent

        def row(parent_frame, label, color_key):
            f = tk.Frame(parent_frame, bg=SURFACE)
            f.pack(fill="x", padx=16, pady=4)
            tk.Label(f, text=label, bg=SURFACE, fg=TEXT,
                     font=("Courier", 10), width=14, anchor="w").pack(side="left")
            sw = ColorSwatch(f, color=self.colors[color_key], size=28)
            sw.pack(side="right")
            def _cb(hex_c, k=color_key):
                self.colors[k] = hex_c
            sw.set_callback(_cb)
            return sw

        # ── Input ──────────────────────────────────────────────────────────────
        section("▸  INPUT")

        inp_frame = tk.Frame(parent, bg=SURFACE)
        inp_frame.pack(fill="x", padx=12, pady=4)

        self.path_var = tk.StringVar(value="No file selected")
        path_lbl = tk.Label(inp_frame, textvariable=self.path_var,
                            bg=SURFACE2, fg=TEXT_DIM,
                            font=("Courier", 8), anchor="w",
                            wraplength=220, justify="left",
                            padx=6, pady=4)
        path_lbl.pack(fill="x", pady=(0, 6))

        tk.Button(inp_frame, text="Browse Video",
                  bg=SURFACE2, fg=ACCENT,
                  relief="flat", font=("Courier", 10),
                  activebackground=BORDER, activeforeground=ACCENT,
                  cursor="hand2", pady=6,
                  command=self._browse).pack(fill="x")

        # ── Options ────────────────────────────────────────────────────────────
        section("▸  OPTIONS")

        opts = tk.Frame(parent, bg=SURFACE)
        opts.pack(fill="x", padx=16, pady=4)

        # Batch size
        tk.Label(opts, text="Batch size", bg=SURFACE, fg=TEXT,
                 font=("Courier", 10)).grid(row=0, column=0, sticky="w", pady=4)
        self.batch_var = tk.IntVar(value=20)
        tk.Spinbox(opts, from_=1, to=100, textvariable=self.batch_var, width=5,
                   bg=SURFACE2, fg=TEXT, relief="flat",
                   buttonbackground=BORDER,
                   font=("Courier", 10),
                   insertbackground=TEXT).grid(row=0, column=1, padx=8, sticky="e")

        # Read from stubs
        self.stubs_var = tk.BooleanVar(value=False)
        stubs_chk = tk.Checkbutton(opts, text="Read from stubs",
                                   variable=self.stubs_var,
                                   bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
                                   activebackground=SURFACE,
                                   activeforeground=ACCENT,
                                   font=("Courier", 10))
        stubs_chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        opts.columnconfigure(1, weight=1)

        # ── Colours ────────────────────────────────────────────────────────────
        section("▸  ANNOTATION COLOURS")

        row(parent, "Team 1",    "team1")
        row(parent, "Team 2",    "team2")
        row(parent, "Referee",   "referee")
        row(parent, "Ball",      "ball")
        row(parent, "Has Ball",  "has_ball")

        # ── Start button ───────────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=16)

        self.start_btn = tk.Button(parent, text="▶  START PROCESSING",
                                   bg=ACCENT, fg=BG,
                                   relief="flat",
                                   font=("Courier", 11, "bold"),
                                   activebackground="#00bcd4",
                                   activeforeground=BG,
                                   cursor="hand2",
                                   pady=10,
                                   command=self._start)
        self.start_btn.pack(fill="x", padx=12, pady=(0, 6))

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode="indeterminate",
                                        style="TProgressbar")
        self.progress.pack(fill="x", padx=12, pady=(0, 8))

        # Log
        section("▸  LOG")
        log_frame = tk.Frame(parent, bg=SURFACE2)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log_text = tk.Text(log_frame, bg=SURFACE2, fg=TEXT_DIM,
                                font=("Courier", 8),
                                relief="flat", state="disabled",
                                wrap="word", height=8)
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Right Panel ────────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        tk.Label(parent, text="OUTPUT PREVIEW",
                 bg=BG, fg=TEXT_DIM,
                 font=("Courier", 9, "bold")).pack(anchor="w", pady=(0, 6))

        self.player = VideoPlayer(parent)
        self.player.pack(fill="both", expand=True)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if path:
            self.video_path = path
            self.path_var.set(os.path.basename(path))
            self._log(f"Loaded: {os.path.basename(path)}")

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"› {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_status(self, text, color):
        self.status_lbl.config(text=text, fg=color)
        self.status_dot.config(fg=color)

    def _start(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            messagebox.showwarning("No Video", "Please select an input video first.")
            return
        self.start_btn.config(state="disabled")
        self.progress.start(12)
        self._set_status("PROCESSING", WARNING)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            self._log("Reading video frames...")

            # Patch tracker colours before importing analyzer
            self._patch_colors()

            from utils.video_utils import read_video_as_frames, write_frames_as_video
            from analyzer import Analyzer

            video_frames = read_video_as_frames(self.video_path)
            self._log(f"Loaded {len(video_frames)} frames.")

            analyzer = Analyzer(
                model_path=MODEL_PATH,
                batch_size=self.batch_var.get(),
                stubs_dir=STUBS_DIR,
                read_from_stubs=self.stubs_var.get()
            )

            self._log("Running analysis pipeline...")
            output_frames = analyzer.analize(video_frames)

            self._log("Writing output video...")
            write_frames_as_video(output_frames, OUTPUT_VIDEO)

            self._log("Done! Loading preview...")
            self.after(0, self._on_done)

        except Exception as e:
            self._log(f"ERROR: {e}")
            self.after(0, self._on_error)

    def _patch_colors(self):
        """Inject chosen colours into the tracker module before analysis runs."""
        import importlib
        # We monkey-patch by storing colours in an env-like global the tracker can read.
        # Since draw_annotations reads from tracks dicts and team_colors, we intercept
        # by setting module-level overrides that analyzer/tracker pick up.
        os.environ["FA_COLOR_TEAM1"]    = ",".join(str(v) for v in hex_to_bgr(self.colors["team1"]))
        os.environ["FA_COLOR_TEAM2"]    = ",".join(str(v) for v in hex_to_bgr(self.colors["team2"]))
        os.environ["FA_COLOR_REFEREE"]  = ",".join(str(v) for v in hex_to_bgr(self.colors["referee"]))
        os.environ["FA_COLOR_BALL"]     = ",".join(str(v) for v in hex_to_bgr(self.colors["ball"]))
        os.environ["FA_COLOR_HAS_BALL"] = ",".join(str(v) for v in hex_to_bgr(self.colors["has_ball"]))

    def _on_done(self):
        self.progress.stop()
        self.start_btn.config(state="normal")
        self._set_status("DONE", SUCCESS)
        self._log("Preview loaded.")
        self.player.load(OUTPUT_VIDEO)

    def _on_error(self):
        self.progress.stop()
        self.start_btn.config(state="normal")
        self._set_status("ERROR", ACCENT2)


if __name__ == "__main__":
    app = FootballAnalyzerApp()
    app.mainloop()