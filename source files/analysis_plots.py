# Δημιουργεί report.csv από φάκελο με instances και παράγει:
# 1) Boxplots Q
# 2) Καμπύλες χρόνου
# 3) Heatmaps Q ως προς (rotation, kerf)
# 4) Barplots μέσου Q και χρόνου
#
# Χρησιμοποιεί το run_benchmark του πακέτου σου.  (bench.py)
# Το LB_area/Q προκύπτει από Instance.area_lb.     (core.py)

import os, re, glob
import numpy as np
import pandas as pd
import time
import shutil
import matplotlib
# force non-GUI backend for background threads
if os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spp.bench import run_benchmark   # file,solver,H,LB_area,ratio_H_over_LBarea,runtime_sec,status
from spp.bench import load_instance_csv
from spp.core import Instance
from spp.solvers.heuristics import HeuristicSolver
from spp.solvers.level_milp import LevelMilpSolver
from spp.solvers.milp import MilpSolver
from spp.solvers.metaheuristics import MetaheuristicSolver


def parse_meta_from_filename(fname):
    base = os.path.basename(fname)
    meta = {}
    def grab(tag, default=None):
        m = re.search(fr"{tag}(\d+)", base)
        return int(m.group(1)) if m else default
    meta["W"]    = grab("W")
    meta["n"]    = grab("n")
    meta["rot"]  = grab("rot")
    meta["kerf"] = grab("kerf")
    meta["seed"] = grab("seed")
    return meta

def build_report(
    instances_dir,
    out_csv="report.csv",
    allow_rotation=True,
    progress_cb=None,               # <-- NEW
    per_milp_time=30,               # optional caps
    per_coord_time=60,
    per_meta_time=3.0,
    clean_results=True,
):
    """
    Incremental benchmark:
      - loops files
      - runs Heuristic (Skyline), Level MILP, Coordinate MILP, Meta-GA
      - writes one combined CSV at the end
      - calls progress_cb(fraction, message) if provided
    """
    paths = sorted(glob.glob(os.path.join(instances_dir, "*.csv")))
    if not paths:
        raise RuntimeError("Δεν βρέθηκαν CSV instances στον φάκελο.")

    # results directory (auto-clean) 
    results_dir = os.path.join(instances_dir, "results")
    if clean_results:
        # safety: ensure we only ever delete "<instances_dir>/results"
        if os.path.commonpath([results_dir, instances_dir]) != os.path.abspath(instances_dir):
            raise RuntimeError("Safety check failed for results_dir path.")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)
    if progress_cb:
        progress_cb(0.02, "Καθαρισμός φακέλου αποτελεσμάτων…")

    rows = []
    N = len(paths)
    t0 = time.time()

    for i, path in enumerate(paths, 1):
        if progress_cb:
            progress_cb(i / N, f"Benchmark {i}/{N}: {os.path.basename(path)}")

        inst = load_instance_csv(path)

        # Heuristic (Skyline)
        st = time.time()
        sol = HeuristicSolver(inst, allow_rotation=allow_rotation, policy="Skyline",
                              guillotine=False).solve()
        dt = time.time() - st
        LB = inst.area_lb()
        rows.append({
            "file": path,
            "solver": "Heuristic",
            "H": sol.H,
            "LB_area": LB,
            "ratio_H_over_LBarea": (sol.H / LB) if LB > 0 else float("nan"),
            "runtime_sec": dt,
            "status": getattr(sol, "optimality", "")
        })

        # Level MILP
        st = time.time()
        try:
            sol = LevelMilpSolver(inst, allow_rotation=allow_rotation,
                                  max_levels=max(1, len(inst.rectangles)//2+1),
                                  time_limit=per_milp_time).solve()
            dt = time.time() - st
            rows.append({
                "file": path,
                "solver": "Level MILP",
                "H": sol.H,
                "LB_area": LB,
                "ratio_H_over_LBarea": (sol.H / LB) if LB > 0 else float("nan"),
                "runtime_sec": dt,
                "status": getattr(sol, "optimality", "")
            })
        except Exception as e:
            rows.append({
                "file": path,
                "solver": "Level MILP",
                "H": float("nan"),
                "LB_area": LB,
                "ratio_H_over_LBarea": float("nan"),
                "runtime_sec": time.time()-st,
                "status": f"Error: {e}"
            })

        # Coordinate MILP
        st = time.time()
        try:
            sol = MilpSolver(inst, allow_rotation=allow_rotation,
                             time_limit=per_coord_time, bigM_mode="tight",
                             warm_start=True, guide_radius=3).solve()
            dt = time.time() - st
            rows.append({
                "file": path,
                "solver": "Coordinate MILP",
                "H": sol.H,
                "LB_area": LB,
                "ratio_H_over_LBarea": (sol.H / LB) if LB > 0 else float("nan"),
                "runtime_sec": dt,
                "status": getattr(sol, "optimality", "")
            })
        except Exception as e:
            rows.append({
                "file": path,
                "solver": "Coordinate MILP",
                "H": float("nan"),
                "LB_area": LB,
                "ratio_H_over_LBarea": float("nan"),
                "runtime_sec": time.time()-st,
                "status": f"Error: {e}"
            })

        # Metaheuristic (GA)
        st = time.time()
        try:
            sol = MetaheuristicSolver(inst, allow_rotation=allow_rotation,
                                      strategy="GA", time_limit=per_meta_time).solve()
            dt = time.time() - st
            rows.append({
                "file": path,
                "solver": "Metaheuristic",
                "H": sol.H,
                "LB_area": LB,
                "ratio_H_over_LBarea": (sol.H / LB) if LB > 0 else float("nan"),
                "runtime_sec": dt,
                "status": getattr(sol, "optimality", "")
            })
        except Exception as e:
            rows.append({
                "file": path,
                "solver": "Metaheuristic",
                "H": float("nan"),
                "LB_area": LB,
                "ratio_H_over_LBarea": float("nan"),
                "runtime_sec": time.time()-st,
                "status": f"Error: {e}"
            })

    df = pd.DataFrame(rows)
    # enrich with meta from filename
    metas = [parse_meta_from_filename(f) for f in df["file"]]
    M = pd.DataFrame(metas)
    df = pd.concat([df, M], axis=1)

    # harmonize names to what your plots expect
    df.rename(columns={"ratio_H_over_LBarea":"Q", "runtime_sec":"time"}, inplace=True)

    # make results folder 
    results_dir = os.path.join(instances_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_csv = os.path.join(results_dir, out_csv)
    df.to_csv(out_csv, index=False)

    if progress_cb:
        progress_cb(1.0, f"Report: {os.path.basename(out_csv)}")
    return df, out_csv, results_dir   

def plot_boxplots_Q(df, out_png):
    ns = sorted(df["n"].dropna().unique())
    solvers = ["Heuristic","Metaheuristic","Level MILP","Coordinate MILP"]
    fig, axes = plt.subplots(1, max(1,len(ns)), figsize=(6*max(1,len(ns)), 5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, n in zip(axes, ns):
        sub = df[df["n"] == n]
        data = [sub[sub["solver"]==s]["Q"].dropna().values for s in solvers]
        ax.boxplot(data, labels=solvers, showfliers=True)
        ax.set_title(f"n={n}")
        ax.set_ylabel("Q = H / LB_area")
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Κατανομή ποιότητας Q ανά solver και πλήθος n")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")

def plot_time_curves(df, out_png, agg="median"):
    solvers = ["Heuristic","Metaheuristic","Level MILP","Coordinate MILP"]
    ns = sorted(df["n"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7,5))
    for s in solvers:
        sub = df[df["solver"]==s]
        y = []
        for n in ns:
            vals = sub[sub["n"]==n]["time"].dropna().values
            if len(vals)==0: y.append(np.nan)
            else: y.append(np.median(vals) if agg=="median" else np.mean(vals))
        ax.plot(ns, y, marker="o", label=s)
    ax.set_xlabel("n (πλήθος ορθογωνίων)")
    ax.set_ylabel(f"{agg} χρόνος (s)")
    ax.set_title("Κλιμάκωση χρόνου ανά solver")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")

def plot_heatmap_Q(df, out_png, solver="Heuristic", n_target=None):
    sub = df[df["solver"]==solver].copy()
    if n_target is not None:
        sub = sub[sub["n"]==n_target]
    piv = sub.pivot_table(index="kerf", columns="rot", values="Q", aggfunc="mean")
    arr = piv.values
    fig, ax = plt.subplots(figsize=(4.5,4))
    im = ax.imshow(arr, origin="lower", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns.tolist())
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index.tolist())
    ax.set_xlabel("rotation (0/1)")
    ax.set_ylabel("kerf δ")
    ax.set_title(f"Heatmap μέσου Q — {solver}" + (f", n={n_target}" if n_target else ""))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="mean Q")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")

def plot_bars_summary(df, out_png):
    solvers = ["Heuristic","Metaheuristic","Level MILP","Coordinate MILP"]
    means_Q = [df[df["solver"]==s]["Q"].dropna().mean() for s in solvers]
    means_T = [df[df["solver"]==s]["time"].dropna().mean() for s in solvers]
    x = np.arange(len(solvers)); w = 0.35
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(x - w/2, means_Q, width=w, label="mean Q")
    ax.bar(x + w/2, means_T, width=w, label="mean time (s)")
    ax.set_xticks(x); ax.set_xticklabels(solvers, rotation=15)
    ax.set_title("Σύνοψη μέσου Q & χρόνου ανά solver")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")

def export_stats_tables(df, out_csv):
    rows = []
    for s in sorted(df["solver"].unique()):
        sub = df[df["solver"]==s]
        Q = sub["Q"].dropna(); T = sub["time"].dropna()
        rows.append([s, Q.mean(), Q.std(), Q.min(), Q.max(), T.mean(), T.std(), T.min(), T.max(),
                     (sub["status"].astype(str)=="Optimal").mean() if "status" in sub else np.nan])
    cols = ["solver","Q_mean","Q_std","Q_min","Q_max","time_mean","time_std","time_min","time_max","pct_optimal"]
    S = pd.DataFrame(rows, columns=cols)
    S.to_csv(out_csv, index=False)
    return S, out_csv

def run_full_analysis(instances_dir, allow_rotation=True):
    # 1) report
    df, report_csv, results_dir = build_report(instances_dir, out_csv="report.csv", allow_rotation=allow_rotation, clean_results=True)

    # 2) plots
    box_png  = os.path.join(results_dir, "boxplots_Q.png")
    time_png = os.path.join(results_dir, "time_curves.png")
    heat_png = os.path.join(results_dir, "heatmap_Q_Heuristic.png")
    bars_png = os.path.join(results_dir, "bars_summary.png")

    plot_boxplots_Q(df, box_png)
    plot_time_curves(df, time_png, agg="median")
    plot_heatmap_Q(df, heat_png, solver="Heuristic", n_target=None)
    plot_bars_summary(df, bars_png)

    # 3) stats tables
    stats_csv = os.path.join(results_dir, "stats_tables.csv")
    export_stats_tables(df, stats_csv)

    return {
        "report_csv": report_csv,
        "boxplots": box_png,
        "time_curves": time_png,
        "heatmap": heat_png,
        "bars": bars_png,
        "stats_csv": stats_csv,
        "results_dir": results_dir
    }
