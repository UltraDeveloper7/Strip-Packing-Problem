
"""
Strip Packing GUI (customtkinter)
- Left panel: fixed header, scrollable body, fixed footer.
- Tables use DualScrollFrame (x+y scroll, no bind_all).
- Draggable forbidden zones (move in x & y) synced με τα πεδία.
- Theme-aware palette + appearance selector (System/Dark/Light).

Requirements:
    pip install customtkinter matplotlib
    # optional for MILP:
    pip install pulp
"""

import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from typing import List, Tuple, Optional
import threading, subprocess, sys, os

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

# project imports
from spp.core import Rectangle, Instance
from spp.viz import Visualizer
from spp.bench import load_instance_csv
from spp.solvers.heuristics import HeuristicSolver
from spp.solvers.level_milp import LevelMilpSolver
from spp.solvers.milp import MilpSolver
from spp.solvers.metaheuristics import MetaheuristicSolver
from make_instances import SPPInstanceGenerator, GenConfig
from analysis_plots import (
    build_report,
    plot_boxplots_Q,
    plot_time_curves,
    plot_heatmap_Q,
    plot_bars_summary,
    export_stats_tables,
)


# -----------------------------------------------------------------------------
# Palette helpers
# -----------------------------------------------------------------------------
def palette() -> dict:
    """Return palette based on CTk appearance mode ('Dark' or 'Light')."""
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "window":  "#1a1a1a",
            "panel":   "#202225",  # left big panel + KPI bar
            "section": "#2a2d31",  # cards
            "header":  "#202225",
            "footer":  "#202225",
            "canvas":  "#2a2d31",  # tables canvas
            "axisbg":  "#1a1a1a",  # mpl facecolor
            "axisfg":  "#e8e8e8",  # mpl labels/ticks
        }
    else:
        return {
            "window":  "#f2f2f2",
            "panel":   "#f7f7f9",
            "section": "#ffffff",
            "header":  "#f7f7f9",
            "footer":  "#f7f7f9",
            "canvas":  "#ffffff",
            "axisbg":  "#ffffff",
            "axisfg":  "#222222",
        }

COL = palette()

class Tooltip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    def show(self, _=None):
        if self.tip: return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        lbl = tk.Label(self.tip, text=self.text, justify="left",
                       bg="#222" if ctk.get_appearance_mode()=="Dark" else "#ffffe0",
                       fg="#fff" if ctk.get_appearance_mode()=="Dark" else "#000",
                       relief="solid", borderwidth=1, font=("Segoe UI", 9))
        lbl.pack(ipadx=6, ipady=3)
        self.tip.wm_geometry(f"+{x}+{y}")
    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# -----------------------------------------------------------------------------
# Helpers: DualScrollFrame (x+y) WITHOUT bind_all, theme-aware
# -----------------------------------------------------------------------------
class DualScrollFrame(ctk.CTkFrame):
    """
    Canvas-based container with both x & y scrollbars.
    Mouse wheel over the area => vertical scroll
    Shift + wheel => horizontal scroll
    Linux wheel (<Button-4/5>) supported.
    Theme can be updated at runtime via set_theme_colors().
    """
    def __init__(self, master, height=160, width=280, bg_canvas: Optional[str] = None, bg_frame: Optional[str] = None):
        super().__init__(master, fg_color=bg_frame or COL["section"])
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Canvas + scrollbars
        self._canvas = tk.Canvas(self, highlightthickness=0, bd=0, relief="flat",
                                 bg=bg_canvas or COL["canvas"])
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._vsb = ctk.CTkScrollbar(self, orientation="vertical", command=self._canvas.yview, width=12)
        self._vsb.grid(row=0, column=1, sticky="ns")

        self._hsb = ctk.CTkScrollbar(self, orientation="horizontal", command=self._canvas.xview, height=12)
        self._hsb.grid(row=1, column=0, sticky="ew")

        self._canvas.configure(yscrollcommand=self._vsb.set, xscrollcommand=self._hsb.set)
        self._canvas.configure(width=width, height=height)

        # Inner frame
        self.inner = ctk.CTkFrame(self._canvas, fg_color=(bg_canvas or COL["canvas"]))
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Keep scrollregion up to date
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel bindings **only on the canvas**
        self._canvas.bind("<Enter>", self._on_enter)
        self._canvas.bind("<Leave>", self._on_leave)
        # Windows / macOS
        self._canvas.bind("<MouseWheel>", self._on_mousewheel_y)
        self._canvas.bind("<Shift-MouseWheel>", self._on_mousewheel_x)
        # Linux
        self._canvas.bind("<Button-4>", lambda e: self._scroll_y(-1))
        self._canvas.bind("<Button-5>", lambda e: self._scroll_y(+1))
        self._canvas.bind("<Shift-Button-4>", lambda e: self._scroll_x(-1))
        self._canvas.bind("<Shift-Button-5>", lambda e: self._scroll_x(+1))

        self._wheel_enabled = False

    # Theme update
    def set_theme_colors(self, canvas_bg: str, frame_bg: str):
        self.configure(fg_color=frame_bg)         # outer CTkFrame color
        self._canvas.configure(bg=canvas_bg)      # tk.Canvas background
        self.inner.configure(fg_color=canvas_bg)  # inner CTkFrame now matches canvas
        self._canvas.update_idletasks()

    # ---- scrolling helpers ----
    def _on_enter(self, _e=None):
        self._canvas.focus_set()
        self._wheel_enabled = True

    def _on_leave(self, _e=None):
        self._wheel_enabled = False

    def _on_mousewheel_y(self, event):
        if not self._wheel_enabled:
            return
        delta = int(-event.delta / 120) if event.delta else 0
        self._scroll_y(delta)

    def _on_mousewheel_x(self, event):
        if not self._wheel_enabled:
            return
        delta = int(-event.delta / 120) if event.delta else 0
        self._scroll_x(delta)

    def _scroll_y(self, units):
        self._canvas.yview_scroll(units, "units")

    def _scroll_x(self, units):
        self._canvas.xview_scroll(units, "units")

    # ---- geometry sync ----
    def _on_inner_configure(self, _evt=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, evt=None):
        bbox = self._canvas.bbox(self._win)
        if bbox:
            self._canvas.itemconfig(self._win, width=max(evt.width, bbox[2] - bbox[0]))


# -----------------------------------------------------------------------------
# Row widgets
# -----------------------------------------------------------------------------
class RectRow(ctk.CTkFrame):
    def __init__(self, master, rid="", w="", h="", rot=True, on_delete=None):
        super().__init__(master, fg_color="transparent")
        self.on_delete = on_delete

        self.e_id = ctk.CTkEntry(self, width=60); self.e_id.insert(0, str(rid))
        self.e_w  = ctk.CTkEntry(self, width=60); self.e_w.insert(0, str(w))
        self.e_h  = ctk.CTkEntry(self, width=60); self.e_h.insert(0, str(h))
        self.cb_rot = ctk.CTkCheckBox(self, text="", width=20)
        if rot: self.cb_rot.select()
        self.btn_del = ctk.CTkButton(self, text="✖", width=30, command=self._delete_clicked)

        self.e_id.grid(row=0, column=0, padx=4, pady=2)
        self.e_w.grid(row=0, column=1, padx=4, pady=2)
        self.e_h.grid(row=0, column=2, padx=4, pady=2)
        self.cb_rot.grid(row=0, column=3, padx=4, pady=2)
        self.btn_del.grid(row=0, column=4, padx=4, pady=2)

    def get_values(self):
        return int(self.e_id.get()), int(self.e_w.get()), int(self.e_h.get()), bool(self.cb_rot.get())

    def _delete_clicked(self):
        if self.on_delete:
            self.on_delete(self)


class FZRow(ctk.CTkFrame):
    def __init__(self, master, x="", y="", w="", h="", on_delete=None):
        super().__init__(master, fg_color="transparent")
        self.on_delete = on_delete
        self.patch: Optional[mpatches.Rectangle] = None  # bound by plot

        self.e_x = ctk.CTkEntry(self, width=60); self.e_x.insert(0, str(x))
        self.e_y = ctk.CTkEntry(self, width=60); self.e_y.insert(0, str(y))
        self.e_w = ctk.CTkEntry(self, width=60); self.e_w.insert(0, str(w))
        self.e_h = ctk.CTkEntry(self, width=60); self.e_h.insert(0, str(h))
        self.btn_del = ctk.CTkButton(self, text="✖", width=30, command=self._delete_clicked)

        self.e_x.grid(row=0, column=0, padx=4, pady=2)
        self.e_y.grid(row=0, column=1, padx=4, pady=2)
        self.e_w.grid(row=0, column=2, padx=4, pady=2)
        self.e_h.grid(row=0, column=3, padx=4, pady=2)
        self.btn_del.grid(row=0, column=4, padx=4, pady=2)

        for e in (self.e_x, self.e_y, self.e_w, self.e_h):
            e.bind("<FocusOut>", self._on_fields_changed)

    def get_values(self):
        return int(self.e_x.get()), int(self.e_y.get()), int(self.e_w.get()), int(self.e_h.get())

    def set_values(self, x, y, w, h):
        self._set_entry(self.e_x, x)
        self._set_entry(self.e_y, y)
        self._set_entry(self.e_w, w)
        self._set_entry(self.e_h, h)

    def bind_patch(self, patch: mpatches.Rectangle):
        self.patch = patch

    def _on_fields_changed(self, _evt=None):
        if not self.patch:
            return
        try:
            x, y, w, h = self.get_values()
        except Exception:
            return
        self.patch.set_xy((x, y))
        self.patch.set_width(w)
        self.patch.set_height(h)
        self.patch.figure.canvas.draw_idle()

    def _set_entry(self, entry: ctk.CTkEntry, v):
        entry.delete(0, tk.END)
        entry.insert(0, str(v))

    def _delete_clicked(self):
        if self.on_delete:
            self.on_delete(self)


class InstanceSettingsDialog(ctk.CTkToplevel):
    def __init__(self, master, on_done, defaults=None):
        super().__init__(master)
        self.title("Instance settings")

        # --- keep above parent ---
        self.transient(master)         # attach to parent window
        self.grab_set()                # modal
        self.lift()
        try:
            self.attributes("-topmost", True)
            # drop -topmost after it appears so it won't stay above EVERYTHING
            self.after(150, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

        # size & centering
        w, h = 640, 500
        self.geometry(f"{w}x{h}")
        try:
            px = master.winfo_rootx() + (master.winfo_width() - w)//2
            py = master.winfo_rooty() + (master.winfo_height() - h)//2
            self.geometry(f"+{max(0,px)}+{max(0,py)}")
        except Exception:
            pass

        self.resizable(False, False)
        self.on_done = on_done
        d = defaults or {}

        # layout: 2 columns, each column = stacked cards (label over control)
        self.grid_columnconfigure((0, 1), weight=1)
        P = dict(padx=10, pady=8)

        # ---------- Row 0: Output folder ----------
        ctk.CTkLabel(self, text="Output folder").grid(row=0, column=0, sticky="w", **P)
        out_row = ctk.CTkFrame(self, fg_color="transparent")
        out_row.grid(row=1, column=0, columnspan=2, sticky="ew", **P)
        out_row.grid_columnconfigure(0, weight=1)
        self.e_out = ctk.CTkEntry(out_row, placeholder_text="e.g., instances")
        self.e_out.insert(0, d.get("out", "instances"))
        self.e_out.grid(row=0, column=0, sticky="ew", padx=(0,8))
        btn_pick = ctk.CTkButton(out_row, text="Choose…", width=100, command=self._pick_folder)
        btn_pick.grid(row=0, column=1)

        Tooltip(self.e_out, "Φάκελος όπου θα αποθηκευτούν τα CSV instances")

        # ---------- Left column ----------
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=2, column=0, sticky="nsew", **P)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="W (strip width):").pack(anchor="w")
        self.opt_W = ctk.CTkOptionMenu(left, values=["30","40","50","60","80"])
        self.opt_W.set(str(d.get("W", 50)))
        self.opt_W.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_W, "Πλάτος λωρίδας W")

        ctk.CTkLabel(left, text="n-list:").pack(anchor="w")
        self.opt_n = ctk.CTkOptionMenu(left, values=["10,20,40","20,40,80","50,100"])
        self.opt_n.set(",".join(map(str, d.get("n_list",[10,20,40]))))
        self.opt_n.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_n, "Λίστα με πλήθη αντικειμένων (π.χ. 10,20,40)")

        ctk.CTkLabel(left, text="count (per n,kerf):").pack(anchor="w")
        self.opt_count = ctk.CTkOptionMenu(left, values=["5","10","20","30","50"])
        self.opt_count.set(str(d.get("count",10)))
        self.opt_count.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_count, "Πλήθος instances για κάθε συνδυασμό (n, kerf)")

        ctk.CTkLabel(left, text="w-range (min,max):").pack(anchor="w")
        self.opt_wr = ctk.CTkOptionMenu(left, values=["2,12","3,15","5,20","8,25"])
        self.opt_wr.set(",".join(map(str, d.get("w_range",(3,15)))))
        self.opt_wr.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_wr, "Εύρος πλάτους ορθογωνίων")

        # ---------- Right column ----------
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=2, column=1, sticky="nsew", **P)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="h-range (min,max):").pack(anchor="w")
        self.opt_hr = ctk.CTkOptionMenu(right, values=["2,12","3,15","5,20","8,25"])
        self.opt_hr.set(",".join(map(str, d.get("h_range",(3,15)))))
        self.opt_hr.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_hr, "Εύρος ύψους ορθογωνίων")

        ctk.CTkLabel(right, text="max-aspect:").pack(anchor="w")
        self.opt_aspect = ctk.CTkOptionMenu(right, values=["2.0","3.0","4.0","9999"])
        self.opt_aspect.set(str(d.get("max_aspect",3.0)))
        self.opt_aspect.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_aspect, "Μέγιστος λόγος πλευρών max(w/h, h/w)")

        ctk.CTkLabel(right, text="rotation:").pack(anchor="w")
        self.opt_rot = ctk.CTkOptionMenu(right, values=["0","1","mix"])
        self.opt_rot.set(d.get("rot","1"))
        self.opt_rot.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_rot, "0=κανένα, 1=όλα, mix=~50%")

        ctk.CTkLabel(right, text="kerf-list:").pack(anchor="w")
        self.opt_kerf = ctk.CTkOptionMenu(right, values=["0","0,1,2","1,2,3","2"])
        self.opt_kerf.set(",".join(map(str, d.get("kerf_list",[0,1,2]))))
        self.opt_kerf.pack(fill="x", pady=(4,10))
        Tooltip(self.opt_kerf, "Τιμές kerf (meta στο όνομα αρχείου)")

        ctk.CTkLabel(right, text="seed:").pack(anchor="w")
        self.e_seed = ctk.CTkEntry(right, placeholder_text="123")
        self.e_seed.insert(0, str(d.get("seed",123)))
        self.e_seed.pack(fill="x", pady=(4,10))
        Tooltip(self.e_seed, "Seed RNG για αναπαραγωγιμότητα")

        # push footer to the bottom
        self.grid_rowconfigure(3, weight=1)
        
        # ---------- footer buttons ----------
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(6,12))
        row.grid_columnconfigure(0, weight=1)
        self.lbl = ctk.CTkLabel(row, text="")
        self.lbl.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(row, text="Generate", width=120, command=self._generate).grid(row=0, column=1, padx=6)
        ctk.CTkButton(row, text="Close", width=100, command=self.destroy).grid(row=0, column=2)

    def _pick_folder(self):
        d = fd.askdirectory(title="Output folder")
        if d:
            self.e_out.delete(0, tk.END); self.e_out.insert(0, d)

    @staticmethod
    def _parse_list(s: str) -> list[int]:
        s = (s or "").strip()
        return [int(x) for x in s.replace(";",",").split(",") if x.strip()]

    @staticmethod
    def _parse_pair(s: str) -> tuple[int,int]:
        xs = [int(x) for x in (s or "").replace(";",",").split(",") if x.strip()]
        if len(xs) != 2:
            raise ValueError("Use min,max")
        a,b = xs
        return (a,b) if a<=b else (b,a)

    def _collect_cfg(self) -> GenConfig:
        # fallbacks if user clears fields
        n_list = self._parse_list(self.opt_n.get()) or [10,20,40]
        kerfs  = self._parse_list(self.opt_kerf.get()) or [0,1,2]
        w_rng  = self._parse_pair(self.opt_wr.get())
        h_rng  = self._parse_pair(self.opt_hr.get())
        seed   = int(self.e_seed.get() or 123)
        return GenConfig(
            out=self.e_out.get().strip() or "instances",
            W=int(self.opt_W.get() or 50),
            n_list=n_list,
            count=int(self.opt_count.get() or 10),
            w_range=w_rng,
            h_range=h_rng,
            max_aspect=float(self.opt_aspect.get() or 3.0),
            rot=self.opt_rot.get() or "1",
            kerf_list=kerfs,
            seed=seed,
        )

    def _generate(self):
        try:
            cfg = self._collect_cfg()
        except Exception as e:
            self.lbl.configure(text=f"⚠ {e}")
            return

        self.on_done(cfg)
        self.lbl.configure(text="Generating…")

        def worker():
            try:
                total, outdir = SPPInstanceGenerator(cfg).generate()
                msg = f"✅ Created {total} files in {outdir}"
            except Exception as e:
                msg = f"Σφάλμα: {e}"
            self.after(0, lambda: self.lbl.configure(text=msg))
        threading.Thread(target=worker, daemon=True).start()

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
class SPPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Strip Packing (customtkinter GUI)")
        self.minsize(1050, 650)
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # theme
        self.configure(fg_color=COL["window"])

        self.grid_columnconfigure(0, weight=0, minsize=380)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

        # state
        self.rect_rows: List[RectRow] = []
        self.fz_rows: List[FZRow] = []
        self._drag_state = {"active": False, "row": None, "dx": 0.0, "dy": 0.0}

        # defaults
        self._add_rect_row(1, 3, 4, True)
        self._add_rect_row(2, 4, 3, True)
        self._add_rect_row(3, 2, 6, True)
        self._add_rect_row(4, 5, 2, True)
        self._add_fz_row(6, 0, 2, 10)
        
        # debounce για auto-solve
        self._pending_solve = None


    # ---------- Left (fixed header + scrollable body + fixed footer) ----------
    def _build_left_panel(self):
        self.left = ctk.CTkFrame(self, corner_radius=0, fg_color=COL["panel"])
        self.left.grid(row=0, column=0, sticky="nsew", padx=(12, 8), pady=12)
        self.left.grid_rowconfigure(1, weight=1)
        self.left.grid_columnconfigure(0, weight=1)

        # Header
        self.header = ctk.CTkFrame(self.left, fg_color=COL["header"])
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_columnconfigure(1, weight=1)
        self.header.grid_columnconfigure(4, weight=1)

        ctk.CTkLabel(self.header, text="Πλάτος Λωρίδας (W):").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.e_W = ctk.CTkEntry(self.header, width=100); self.e_W.insert(0, "10")
        self.e_W.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        self.cb_rotation = ctk.CTkCheckBox(self.header, text="Allow rotation 90°"); self.cb_rotation.select()
        self.cb_rotation.grid(row=1, column=0, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(self.header, text="Kerf δ:").grid(row=1, column=1, padx=6, pady=6, sticky="e")
        self.e_kerf = ctk.CTkEntry(self.header, width=60); self.e_kerf.insert(0, "1")
        self.e_kerf.grid(row=1, column=2, padx=6, pady=6, sticky="w")

        # Appearance selector
        ctk.CTkLabel(self.header, text="Appearance:").grid(row=0, column=3, padx=6, pady=6, sticky="e")
        self.opt_appearance = ctk.CTkOptionMenu(self.header, values=["System", "Dark", "Light"],
                                                command=self._on_change_appearance, width=120)
        self.opt_appearance.set("System")
        self.opt_appearance.grid(row=0, column=4, padx=6, pady=6, sticky="w")

        # Body (scrollable vertically). Tables inside have x+y scroll via DualScrollFrame.
        self.body = ctk.CTkScrollableFrame(self.left, fg_color=COL["panel"])
        self.body.grid(row=1, column=0, sticky="nsew", pady=(8, 8))

        # Rectangles
        self.sec_rects = ctk.CTkFrame(self.body, fg_color=COL["section"], corner_radius=8)
        self.sec_rects.pack(fill="x", padx=0, pady=6)
        ctk.CTkLabel(self.sec_rects, text="Ορθογώνια (id, w, h, rot)").pack(anchor="w", padx=8, pady=(8, 2))

        hdr = ctk.CTkFrame(self.sec_rects, fg_color="transparent")
        hdr.pack(fill="x", padx=8)
        for j, t in enumerate(["id", "w", "h", "rot", ""]):
            ctk.CTkLabel(hdr, text=t).grid(row=0, column=j, padx=4, pady=2)

        self.rect_table = DualScrollFrame(self.sec_rects, height=160, width=320,
                                          bg_canvas=COL["canvas"], bg_frame=COL["section"])
        self.rect_table.pack(fill="x", padx=8, pady=(2, 6))
        self.rect_container = self.rect_table.inner

        btns = ctk.CTkFrame(self.sec_rects, fg_color="transparent")
        btns.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkButton(btns, text="+ Προσθήκη",
                      command=lambda: self._add_rect_row("", "", "", True), width=110).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="Φόρτωση CSV",
                      command=self._load_csv, width=110).pack(side="left", padx=4)

        # Forbidden Zones
        self.sec_fz = ctk.CTkFrame(self.body, fg_color=COL["section"], corner_radius=8)
        self.sec_fz.pack(fill="x", padx=0, pady=6)
        ctk.CTkLabel(self.sec_fz, text="Forbidden Zones (x, y, w, h)").pack(anchor="w", padx=8, pady=(8, 2))

        hdr_fz = ctk.CTkFrame(self.sec_fz, fg_color="transparent")
        hdr_fz.pack(fill="x", padx=8)
        for j, t in enumerate(["x", "y", "w", "h", ""]):
            ctk.CTkLabel(hdr_fz, text=t).grid(row=0, column=j, padx=4, pady=2)

        self.fz_table = DualScrollFrame(self.sec_fz, height=120, width=320,
                                        bg_canvas=COL["canvas"], bg_frame=COL["section"])
        self.fz_table.pack(fill="x", padx=8, pady=(2, 6))
        self.fz_container = self.fz_table.inner

        ctk.CTkButton(self.sec_fz, text="+ Προσθήκη",
                      command=lambda: self._add_fz_row("", "", "", ""), width=110).pack(anchor="w", padx=8, pady=(0, 8))

        # Footer (fixed)
        self.footer = ctk.CTkFrame(self.left, fg_color=COL["footer"], corner_radius=8)
        self.footer.grid(row=2, column=0, sticky="ew")
        self.footer.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.footer, text="Solver:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.opt_solver = ctk.CTkOptionMenu(
            self.footer, 
            values=["Heuristic", "Level MILP", "Coordinate MILP", "Metaheuristic"], 
            width=180,
            command=self._on_solver_changed
        )
        self.opt_solver.grid(row=0, column=1, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(self.footer, text="Heuristic policy:").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        self.opt_policy = ctk.CTkOptionMenu(self.footer, values=["Skyline", "NFDH"], width=120)
        self.opt_policy.grid(row=1, column=1, padx=8, pady=6, sticky="w")

        self.cb_guillotine = ctk.CTkCheckBox(self.footer, text="Guillotine (enforce shelves/NFDH)")
        self.cb_guillotine.grid(row=2, column=0, padx=8, pady=6, sticky="w")

        ctk.CTkLabel(self.footer, text="Meta strategy:").grid(row=3, column=0, padx=8, pady=6, sticky="w")
        self.opt_meta = ctk.CTkOptionMenu(self.footer, values=["GA", "SA", "Tabu"], width=120)
        self.opt_meta.grid(row=3, column=1, padx=8, pady=6, sticky="w")
        Tooltip(
            self.opt_meta,
            "Διαθέσιμο μόνο όταν ο solver είναι 'Metaheuristic'.\n"
            "Επιλέξτε GA (Genetic Algorithm), SA (Simulated Annealing) ή Tabu."
        )

        ctk.CTkLabel(self.footer, text="Time limit (sec):").grid(row=4, column=0, padx=8, pady=6, sticky="w")
        self.e_tlimit = ctk.CTkEntry(self.footer, width=80); self.e_tlimit.insert(0, "30")
        self.e_tlimit.grid(row=4, column=1, padx=8, pady=6, sticky="w")

        # NEW: Auto-solve after dragging forbidden zones
        self.cb_autosolve = ctk.CTkCheckBox(self.footer, text="Auto-solve after drag")
        self.cb_autosolve.grid(row=5, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="w")

        bar = ctk.CTkFrame(self.footer, fg_color="transparent")
        bar.grid(row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 8))
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_columnconfigure(3, weight=1)

        ctk.CTkButton(bar, text="Solve", command=self._solve, width=140).grid(row=0, column=1, padx=6, pady=4)
        ctk.CTkButton(bar, text="Save Image", command=self._save_image, width=120).grid(row=0, column=2, padx=6, pady=4)

        # --- Benchmark section ---
        bench_sec = ctk.CTkFrame(self.footer, fg_color="transparent")
        bench_sec.grid(row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 8))
        bench_sec.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(bench_sec, text="Benchmark:").grid(row=0, column=0, padx=4, pady=(6, 2), sticky="w")
        bar2 = ctk.CTkFrame(bench_sec, fg_color="transparent")
        bar2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=(2, 6))
        
        self.btn_cfg_gen = ctk.CTkButton(bar2, text="Instance settings…", width=160,
                                        command=self._open_gen_dialog)
        self.btn_cfg_gen.pack(side="left", padx=4)
        
        self.btn_run_bench = ctk.CTkButton(bar2, text="Run benchmark (folder)", width=180,
                                        command=self._benchmark_pick_folder)
        self.btn_run_bench.pack(side="left", padx=4)
        
        self.btn_open_bench = ctk.CTkButton(bar2, text="Open folder", width=120,
                                            command=self._benchmark_open_folder)
        self.btn_open_bench.pack(side="left", padx=4)

        # ➜ status line (keep as-is)
        self.lbl_bench = ctk.CTkLabel(bench_sec, text="", anchor="w")
        self.lbl_bench.grid(row=2, column=0, sticky="ew", padx=2, pady=(0, 4))

        # ➜ PROGRESS (hidden by default)
        self.pb_row = ctk.CTkFrame(bench_sec, fg_color="transparent")
        self.pb_row.grid(row=3, column=0, sticky="ew", padx=2, pady=(0, 6))
        self.pb_row.grid_columnconfigure(0, weight=1)

        # short/slim bar
        self.pb_bench = ctk.CTkProgressBar(self.pb_row, height=8)   
        self.pb_bench.grid(row=0, column=0, sticky="ew", padx=(0,6))
        self.pb_bench.set(0.0)

        self.lbl_pct = ctk.CTkLabel(self.pb_row, text="0%", width=34)  
        self.lbl_pct.grid(row=0, column=1, sticky="e")

        # hide it until user presses Run
        self.pb_row.grid_remove()

        self._last_bench_dir = None
        
        # Initialize enable/disable state based on current solver selection
        self._update_meta_availability(self.opt_solver.get())

    def _bench_progress(self, value_0_1: float, msg: str = ""):
        """Ασφαλής ενημέρωση UI από thread: 0..1 και μήνυμα."""
        value_0_1 = max(0.0, min(1.0, float(value_0_1)))
        def _do():
            self.pb_bench.set(value_0_1)
            if msg:
                self.lbl_bench.configure(text=msg)
            self.lbl_pct.configure(text=f"{int(round(value_0_1*100))}%")
            self.update_idletasks()
        self.after(0, _do)

    # ---------- Right (plot + KPIs) ----------
    def _build_right_panel(self):
        self.right = ctk.CTkFrame(self, corner_radius=0, fg_color=COL["window"])
        self.right.grid(row=0, column=1, sticky="nsew", padx=(4, 12), pady=12)
        self.right.grid_rowconfigure(0, weight=1)
        self.right.grid_rowconfigure(1, weight=0)
        self.right.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(7.0, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.ax.set_title("Solution preview")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Forbidden-zone drag events
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self.kpi = ctk.CTkFrame(self.right, fg_color=COL["panel"], corner_radius=8)
        self.kpi.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for c in range(6):
            self.kpi.grid_columnconfigure(c, weight=1)

        self.lbl_H   = ctk.CTkLabel(self.kpi, text="H: -");           self.lbl_H.grid(row=0, column=0, padx=6, pady=6)
        self.lbl_LB  = ctk.CTkLabel(self.kpi, text="LB_area: -");     self.lbl_LB.grid(row=0, column=1, padx=6, pady=6)
        self.lbl_rat = ctk.CTkLabel(self.kpi, text="H/LB_area: -");   self.lbl_rat.grid(row=0, column=2, padx=6, pady=6)
        self.lbl_tm  = ctk.CTkLabel(self.kpi, text="time: - s");      self.lbl_tm.grid(row=0, column=3, padx=6, pady=6)
        self.lbl_st  = ctk.CTkLabel(self.kpi, text="status: -");      self.lbl_st.grid(row=0, column=4, padx=6, pady=6)
        self.lbl_mth = ctk.CTkLabel(self.kpi, text="method: -");      self.lbl_mth.grid(row=0, column=5, padx=6, pady=6)

    # ---------- Appearance change ----------
    def _on_change_appearance(self, choice: str):
        if choice == "System":
            ctk.set_appearance_mode("system")
        else:
            ctk.set_appearance_mode(choice.lower().capitalize())
        global COL
        COL = palette()
        self._cancel_pending_solve()
        self._apply_theme()

    def _apply_theme(self):
        # root
        self.configure(fg_color=COL["window"])
        # left
        for f, key in [(self.left, "panel"), (self.header, "header"),
                       (self.body, "panel"), (self.footer, "footer"),
                       (self.sec_rects, "section"), (self.sec_fz, "section")]:
            f.configure(fg_color=COL[key])
        self.rect_table.set_theme_colors(COL["canvas"], COL["section"])
        self.fz_table.set_theme_colors(COL["canvas"], COL["section"])
        # right
        self.right.configure(fg_color=COL["window"])
        self.kpi.configure(fg_color=COL["panel"])
        # matplotlib
        self._style_axes(self.ax)
        self._draw_solution(getattr(self, "_last_inst", None), getattr(self, "_last_sol", None))
        self.canvas.draw_idle()

    @staticmethod
    def _style_axes(ax):
        ax.figure.patch.set_facecolor(COL["axisbg"])
        ax.set_facecolor(COL["axisbg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(COL["axisfg"])
        ax.tick_params(colors=COL["axisfg"])
        ax.xaxis.label.set_color(COL["axisfg"])
        ax.yaxis.label.set_color(COL["axisfg"])
        ax.title.set_color(COL["axisfg"])

    def _on_solver_changed(self, choice: str):
        self._update_meta_availability(choice)

    def _update_meta_availability(self, solver_name: str | None = None):
        name = solver_name or self.opt_solver.get()
        is_meta = (name == "Metaheuristic")
        try:
            # normally supported by customtkinter
            self.opt_meta.configure(state=("normal" if is_meta else "disabled"))
        except Exception:
            # fallback αν κάποια έκδοση δεν έχει state σε CTkOptionMenu
            if is_meta:
                self.opt_meta.configure(text_color=None, fg_color=None)
            else:
                self.opt_meta.configure(text_color=("#888", "#aaa"), fg_color=("#333", "#e6e6e6"))

    # ---------- Rows helpers ----------
    def _add_rect_row(self, rid, w, h, rot):
        row = RectRow(self.rect_container, rid, w, h, rot, on_delete=self._remove_rect_row)
        row.pack(fill="x", padx=0, pady=1)
        self.rect_rows.append(row)

    def _remove_rect_row(self, row: RectRow):
        self.rect_rows.remove(row)
        row.destroy()

    def _add_fz_row(self, x, y, w, h):
        row = FZRow(self.fz_container, x, y, w, h, on_delete=self._remove_fz_row)
        row.pack(fill="x", padx=0, pady=1)
        self.fz_rows.append(row)

    def _remove_fz_row(self, row: FZRow):
        self.fz_rows.remove(row)
        row.destroy()
        self._draw_solution(getattr(self, "_last_inst", None), getattr(self, "_last_sol", None))

    # ---------- CSV ----------
    def _load_csv(self):
        path = fd.askopenfilename(title="Επέλεξε CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            inst = load_instance_csv(path)
        except Exception as e:
            mb.showerror("Σφάλμα CSV", f"Αποτυχία ανάγνωσης: {e}")
            return
        self.e_W.delete(0, tk.END); self.e_W.insert(0, str(inst.W))
        for r in self.rect_rows[:]:
            self._remove_rect_row(r)
        for r in inst.rectangles:
            self._add_rect_row(r.id, r.w, r.h, r.rotatable)

    # ---------- Solve ----------
    def _collect_instance(self) -> Instance:
        W = int(self.e_W.get())
        kerf = int(self.e_kerf.get())

        rects: List[Rectangle] = []
        seen = set()
        for row in self.rect_rows:
            rid, w, h, rot = row.get_values()
            if rid in seen:
                raise ValueError(f"Διπλό id ορθογωνίου: {rid}")
            if w <= 0 or h <= 0:
                raise ValueError("Τα w,h πρέπει να είναι θετικά.")
            seen.add(rid)
            rects.append(Rectangle(rid, w, h, rot))

        fz: List[Tuple[int,int,int,int]] = []
        for row in self.fz_rows:
            x, y, w, h = row.get_values()
            if w <= 0 or h <= 0:
                raise ValueError("Οι διαστάσεις forbidden zone πρέπει να είναι θετικές.")
            fz.append((x, y, w, h))

        return Instance(W=W, rectangles=rects, kerf_delta=kerf, forbidden_zones=fz)

    def _solve(self):
        self._cancel_pending_solve()
        try:
            inst = self._collect_instance()
        except Exception as e:
            mb.showerror("Σφάλμα εισόδου", str(e))
            return

        allow_rotation = bool(self.cb_rotation.get())
        solver_name = self.opt_solver.get()
        policy = self.opt_policy.get()
        guillotine = bool(self.cb_guillotine.get())
        meta_strategy = self.opt_meta.get()
        try:
            tlimit = int(self.e_tlimit.get())
        except Exception:
            tlimit = 30

        import time
        t0 = time.time()
        try:
            if solver_name == "Heuristic":
                sol = HeuristicSolver(inst, allow_rotation=allow_rotation,
                                      policy=policy, guillotine=guillotine).solve()
            elif solver_name == "Level MILP":
                sol = LevelMilpSolver(inst, allow_rotation=allow_rotation,
                                      max_levels=max(1, len(inst.rectangles)//2+1),
                                      time_limit=tlimit).solve()
            elif solver_name == "Coordinate MILP":
                sol = MilpSolver(inst, allow_rotation=allow_rotation,
                                 time_limit=tlimit, bigM_mode="tight",
                                 warm_start=True, guide_radius=3).solve()
            else:
                sol = MetaheuristicSolver(inst, allow_rotation=allow_rotation,
                                          strategy=meta_strategy, time_limit=5.0).solve()
        except Exception as e:
            mb.showerror("Σφάλμα επίλυσης", str(e))
            return
        t1 = time.time()

        LB = inst.area_lb()
        ratio = round(sol.H / LB, 3) if LB > 0 else "-"
        self.lbl_H.configure(text=f"H: {sol.H}")
        self.lbl_LB.configure(text=f"LB_area: {LB}")
        self.lbl_rat.configure(text=f"H/LB_area: {ratio}")
        self.lbl_tm.configure(text=f"time: {round(t1-t0,3)} s")
        self.lbl_st.configure(text=f"status: {sol.optimality}")
        self.lbl_mth.configure(text=f"method: {sol.method}")

        self._last_inst = inst
        self._last_sol = sol
        self._draw_solution(inst, sol)

    # ---------- Auto-solve helpers ----------
    def _trigger_solve_debounced(self, delay_ms: int = 300):
        """Προγραμμάτισε επίλυση με μικρή καθυστέρηση (debounce)."""
        if self._pending_solve is not None:
            try:
                self.after_cancel(self._pending_solve)
            except Exception:
                pass
        self._pending_solve = self.after(delay_ms, self._solve)

    def _cancel_pending_solve(self):
        if self._pending_solve is not None:
            try:
                self.after_cancel(self._pending_solve)
            except Exception:
                pass
            self._pending_solve = None


    # ---------- Drawing + draggable FZ ----------
    def _draw_solution(self, inst: Optional[Instance], sol):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self._style_axes(self.ax)

        if not inst or not sol:
            self.ax.set_title("Solution preview")
            self.canvas.draw_idle()
            return

        self.ax.set_title(f"{sol.method} – H={sol.H}")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.set_xlim(0, inst.W)
        self.ax.set_ylim(0, sol.H)
        self.ax.set_aspect('equal', adjustable='box')

        import random
        rnd = random.Random(1234)

        for p in sol.placements:
            color = (rnd.random(), rnd.random(), rnd.random())
            rect = mpatches.Rectangle((p.x, p.y), p.w_eff, p.h_eff, edgecolor='black',
                                      facecolor=color, alpha=0.6)
            self.ax.add_patch(rect)
            self.ax.text(p.x + p.w_eff/2, p.y + p.h_eff/2,
                         f"id={p.rect_id}\n{p.w_eff}×{p.h_eff}",
                         ha='center', va='center', fontsize=8, color=COL["axisfg"])

        self.ax.plot([0, inst.W, inst.W, 0, 0], [0, 0, sol.H, sol.H, 0], color=COL["axisfg"])

        for row in self.fz_rows:
            try:
                x, y, w, h = row.get_values()
            except Exception:
                continue
            rz = mpatches.Rectangle((x, y), w, h, edgecolor='red', facecolor='none',
                                    linestyle='--', linewidth=1.3)
            self.ax.add_patch(rz)
            self.ax.text(x + w/2, y + h/2, "FORB", color='red', ha='center', va='center', fontsize=8)
            row.bind_patch(rz)

        self.ax.grid(True, linestyle='--', alpha=0.3, color=COL["axisfg"])
        self.canvas.draw_idle()

    # hit-test for draggable zones
    def _hit_zone_row(self, event) -> Optional[FZRow]:
        if event.inaxes != self.ax:
            return None
        for row in self.fz_rows:
            if not row.patch:
                continue
            x, y = row.patch.get_xy()
            w, h = row.patch.get_width(), row.patch.get_height()
            if x <= (event.xdata or -1e9) <= x + w and y <= (event.ydata or -1e9) <= y + h:
                return row
        return None

    def _on_press(self, event):
        row = self._hit_zone_row(event)
        if not row:
            self._drag_state["active"] = False
            return
        x, y = row.patch.get_xy()
        self._drag_state.update({"active": True, "row": row, "dx": (event.xdata or 0) - x, "dy": (event.ydata or 0) - y})

    def _on_motion(self, event):
        if not self._drag_state["active"] or event.inaxes != self.ax:
            return
        row: FZRow = self._drag_state["row"]
        if not row or not row.patch:
            return
        new_x = int(round((event.xdata or 0) - self._drag_state["dx"]))
        new_y = int(round((event.ydata or 0) - self._drag_state["dy"]))
        row.patch.set_xy((new_x, new_y))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._drag_state["active"]:
            return
        row: FZRow = self._drag_state["row"]
        if row and row.patch:
            x, y = row.patch.get_xy()
            w, h = row.patch.get_width(), row.patch.get_height()
            row.set_values(int(round(x)), int(round(y)), int(round(w)), int(round(h)))
        self._drag_state["active"] = False
        self._drag_state["row"] = None
        
                # Auto-solve if the checkbox is enabled
        try:
            if bool(self.cb_autosolve.get()):
                # Optional: clamp zones to current strip limits (W, H) before solving
                if hasattr(self, "_last_inst") and hasattr(self, "_last_sol"):
                    W = self._last_inst.W
                    H = self._last_sol.H
                    for row2 in self.fz_rows:
                        if row2.patch:
                            x, y = row2.patch.get_xy()
                            w, h = row2.patch.get_width(), row2.patch.get_height()
                            # Clamp so that the forbidden zone stays within the strip
                            x = max(0, min(int(round(x)), max(0, W - int(round(w)))))
                            y = max(0, min(int(round(y)), max(0, H - int(round(h)))))
                            row2.patch.set_xy((x, y))
                            row2.set_values(x, y, int(round(w)), int(round(h)))
                # Debounced solve
                self._trigger_solve_debounced(delay_ms=300)
        except Exception:
            pass 

    def _open_gen_dialog(self):
        # keep user's last selection
        defaults = getattr(self, "_gen_defaults", dict(
            out="instances", W=50, n_list=[10,20,40], count=10,
            w_range=(3,15), h_range=(3,15), max_aspect=3.0,
            rot="1", kerf_list=[0,1,2], seed=123
        ))

        def on_done(cfg: GenConfig):
            # αποθήκευσε defaults για επόμενη φορά
            self._gen_defaults = dict(
                out=cfg.out, W=cfg.W, n_list=list(cfg.n_list), count=cfg.count,
                w_range=tuple(cfg.w_range), h_range=tuple(cfg.h_range),
                max_aspect=cfg.max_aspect, rot=cfg.rot,
                kerf_list=list(cfg.kerf_list), seed=cfg.seed
            )
            # προαιρετικά: δείξε στο status ότι είναι έτοιμα instances
            self.lbl_bench.configure(text=f"Generator config saved → {cfg.out}")

            # bonus: αν θέλεις να ανοίγει ο φάκελος μετά τη δημιουργία:
            self._last_bench_dir = cfg.out

        InstanceSettingsDialog(self, on_done, defaults=defaults)


    def _benchmark_pick_folder(self):
        d = fd.askdirectory(title="Επίλεξε φάκελο με CSV instances")
        if not d:
            return
        self._last_bench_dir = d
        # ενημέρωση UI
        self.lbl_bench.configure(text="Εκτέλεση benchmark… παρακαλώ αναμονή.")
        self.btn_run_bench.configure(state="disabled")
        self.btn_open_bench.configure(state="disabled")
        # show bar only now
        self.pb_row.grid() # make visible
        self.pb_bench.set(0.0)
        self.lbl_pct.configure(text="0%")
        # τρέχει σε thread για να μην «παγώνει» το GUI
        t = threading.Thread(target=self._benchmark_worker, args=(d,), daemon=True)
        t.start() 

    def _benchmark_worker(self, folder):
        import traceback

        try:
            allow_rotation = bool(self.cb_rotation.get())

            # 0% → 10%: start
            self._bench_progress(0.10, "Σάρωση αρχείων…")

            # --- Progress mapping (10%→35%) ---
            def map_prog(local_frac, msg=""):
                try:
                    f = max(0.0, min(1.0, float(local_frac)))
                except Exception:
                    f = 0.0
                global_val = 0.10 + 0.25 * f
                self.after(0, lambda: self._bench_progress(global_val, msg or f"Benchmark {int(f*100)}%"))

            # --- build_report με προοδευτική ενημέρωση ---
            try:
                df, report_csv, results_dir = build_report(
                    folder,
                    out_csv="report.csv",
                    allow_rotation=allow_rotation,
                    progress_cb=map_prog,
                    per_milp_time=30,
                    per_coord_time=60,
                    per_meta_time=3.0,
                    clean_results=True,
                )
            except TypeError as te:
                # Πιθανό παλιό build_report χωρίς progress_cb → κάνε fallback
                print("[BENCH] build_report δεν δέχεται progress_cb → fallback:", te, flush=True)
                df, report_csv, results_dir = build_report(
                    folder,
                    out_csv="report.csv",
                    allow_rotation=allow_rotation
                )
                # φέρε τη μπάρα ως το 35%
                self._bench_progress(0.35, f"Report: report.csv")
            except Exception as e:
                # Τύπωσε πλήρες traceback για να δούμε ακριβώς το πρόβλημα
                print("[BENCH] ΣΦΑΛΜΑ μέσα στο build_report:", flush=True)
                traceback.print_exc()
                raise

            self._bench_progress(0.35, f"Report: {os.path.basename(report_csv)}")

            # 35% → 55%: boxplots
            box_png = os.path.join(results_dir, "boxplots_Q.png")
            plot_boxplots_Q(df, box_png)
            self._bench_progress(0.55, f"Boxplots: {os.path.basename(box_png)}")

            # 55% → 70%: time curves
            time_png = os.path.join(results_dir, "time_curves.png")
            plot_time_curves(df, time_png, agg="median")
            self._bench_progress(0.70, f"Time curves: {os.path.basename(time_png)}")

            # 70% → 85%: heatmap
            heat_png = os.path.join(results_dir, "heatmap_Q_Heuristic.png")
            plot_heatmap_Q(df, heat_png, solver="Heuristic", n_target=None)
            self._bench_progress(0.85, f"Heatmap: {os.path.basename(heat_png)}")

            # 85% → 92%: stats
            stats_csv = os.path.join(results_dir, "stats_tables.csv")
            export_stats_tables(df, stats_csv)
            self._bench_progress(0.92, f"Stats: {os.path.basename(stats_csv)}")

            # 92% → 100%: bars summary
            bars_png = os.path.join(results_dir, "bars_summary.png")
            plot_bars_summary(df, bars_png)
            self._bench_progress(1.00, "Ολοκληρώθηκε.")

            msg = (
                "✅ Benchmark ολοκληρώθηκε – αποθηκεύτηκαν "
                "report, plots και στατιστικά στον φάκελο 'results'."
            )

        except Exception as e:
            # Δείξε τί ακριβώς έγινε και στο τερματικό και στο GUI
            print("[BENCH] ΣΦΑΛΜΑ (traceback πιο πάνω):", e, flush=True)
            msg = f"Σφάλμα benchmark: {e}"

        # restore UI state (στο UI thread)
        def _done():
            self.lbl_bench.configure(text=msg)
            self.btn_run_bench.configure(state="normal")
            self.btn_open_bench.configure(state="normal")
            self.pb_row.grid_remove()
        self.after(0, _done)

    def _benchmark_open_folder(self):
        d = self._last_bench_dir
        if not d:
            mb.showinfo("Benchmark", "Δεν έχει οριστεί φάκελος.")
            return
        # άνοιγμα στον explorer
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", d])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])

    # ---------- Save ----------
    def _save_image(self):
        if not hasattr(self, "_last_inst") or not hasattr(self, "_last_sol"):
            mb.showinfo("Πληροφορία", "Δεν υπάρχει διαθέσιμη λύση για αποθήκευση.")
            return
        path = fd.asksaveasfilename(defaultextension=".png",
                                    filetypes=[("PNG", "*.png")],
                                    title="Αποθήκευση εικόνας")
        if not path:
            return
        try:
            Visualizer().draw(self._last_inst, self._last_sol,
                              title=f"{self._last_sol.method} – H={self._last_sol.H}",
                              save_path=path, show=False)
            mb.showinfo("OK", "Αποθηκεύτηκε.")
        except Exception as e:
            mb.showerror("Σφάλμα", f"Αποτυχία αποθήκευσης: {e}")


if __name__ == "__main__":
    app = SPPApp()
    app.mainloop()
