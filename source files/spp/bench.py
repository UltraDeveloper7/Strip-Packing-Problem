
from __future__ import annotations
import csv, time, os
from typing import List, Tuple
from .core import Rectangle, Instance
from .solvers.heuristics import HeuristicSolver
from .solvers.milp import MilpSolver
from .solvers.level_milp import LevelMilpSolver
from .solvers.metaheuristics import MetaheuristicSolver

def load_instance_csv(path: str) -> Instance:
    """Μορφή CSV:
    W,<int>
    id,w,h,rotatable(0/1)
    ..."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        W_line = next(rd)
        assert W_line[0].strip().upper() == "W"
        W = int(W_line[1])
        rects = []
        for row in rd:
            if not row or row[0].startswith("#"): continue
            rid, w, h, rot = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            rects.append(Rectangle(rid, w, h, bool(rot)))
    return Instance(W=W, rectangles=rects)

def run_benchmark(dataset_paths: List[str], out_csv: str, allow_rotation: bool = True):
    """Τρέχει Heuristic (Skyline), Level MILP και Coordinate MILP, και αναφέρει H, LB_area, H/LB_area, runtime."""
    rows = [("file","solver","H","LB_area","ratio_H_over_LBarea","runtime_sec","status")]
    for path in dataset_paths:
        inst = load_instance_csv(path)
        LB = inst.area_lb()
        # Heuristic
        t0=time.time(); hsol = HeuristicSolver(inst, allow_rotation, policy="Skyline").solve(); t1=time.time()
        rows.append((os.path.basename(path), "Heuristic-Skyline", hsol.H, LB, round(hsol.H/LB,3), round(t1-t0,3), "feasible"))
        # Level MILP
        try:
            t0=time.time(); lsol = LevelMilpSolver(inst, allow_rotation, max_levels=len(inst.rectangles)//2+1, time_limit=30).solve(); t1=time.time()
            rows.append((os.path.basename(path), "MILP-ShelfLevel", lsol.H, LB, round(lsol.H/LB,3), round(t1-t0,3), "pulp"))
        except Exception as e:
            rows.append((os.path.basename(path), "MILP-ShelfLevel", "", LB, "", "", f"error:{e}"))
        # Coordinate MILP (tight, warm-start)
        try:
            t0=time.time(); msol = MilpSolver(inst, allow_rotation, time_limit=60, bigM_mode="tight", warm_start=True, guide_radius=3).solve(); t1=time.time()
            rows.append((os.path.basename(path), "MILP-Coordinate", msol.H, LB, round(msol.H/LB,3), round(t1-t0,3), "pulp"))
        except Exception as e:
            rows.append((os.path.basename(path), "MILP-Coordinate", "", LB, "", "", f"error:{e}"))
        # Metaheuristic (GA)
        t0=time.time(); gsol = MetaheuristicSolver(inst, allow_rotation, strategy="GA", time_limit=3.0).solve(); t1=time.time()
        rows.append((os.path.basename(path), "Meta-GA", gsol.H, LB, round(gsol.H/LB,3), round(t1-t0,3), "feasible"))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerows(rows)
    return out_csv
