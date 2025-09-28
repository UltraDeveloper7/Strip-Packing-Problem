from __future__ import annotations
from typing import Optional, Dict, Tuple, List
import itertools

try:
    import pulp
except Exception:
    pulp = None

from ..core import Instance, Rectangle, Placement, Solution
from .base import BaseSolver
from .heuristics import HeuristicSolver


class MilpSolver(BaseSolver):
    """Coordinate-based MILP (patched, ASCII-only).
    - Integer x_i, y_i, H; optional binary r_i for 90 deg rotation.
    - Big-M non-overlap with kerf delta.
    - Optional forbidden zones via 4-way disjunction per item/zone.
    - Warm-start from Skyline. No hard guide box when kerf > 0.
    """
    def __init__(self, instance: Instance, allow_rotation: bool = True,
                 time_limit: Optional[int] = None, bigM_mode: str = "tight",
                 warm_start: bool = True, guide_radius: int = 2):
        super().__init__(instance, allow_rotation)
        self.time_limit = time_limit
        self.bigM_mode = bigM_mode
        self.warm_start = warm_start
        self.guide_radius = guide_radius

    def solve(self) -> Solution:
        if pulp is None:
            raise RuntimeError("PuLP is not available. Please `pip install pulp`.")

        # Heuristic upper bound (also seed for warm-start)
        hsol = HeuristicSolver(self.instance, allow_rotation=self.allow_rotation,
                               policy="Skyline", guillotine=False).solve()
        H_ub = max(hsol.H, self.instance.maxh_lb(self.allow_rotation))

        # Lower bounds
        LB = max(self.instance.area_lb(), self.instance.maxh_lb(self.allow_rotation))

        model = pulp.LpProblem("SPP_COORDINATE_BASED", pulp.LpMinimize)
        ids = [r.id for r in self.instance.rectangles]

        x = pulp.LpVariable.dicts("x", ids, lowBound=0, cat="Integer")
        y = pulp.LpVariable.dicts("y", ids, lowBound=0, cat="Integer")
        H = pulp.LpVariable("H", lowBound=LB, cat="Integer")

        if self.allow_rotation:
            r = pulp.LpVariable.dicts("r", ids, lowBound=0, upBound=1, cat="Binary")
        else:
            r = {rid: None for rid in ids}

        def w_eff(rect: Rectangle):
            # linear expression in r[rect.id]
            if not self.allow_rotation or not rect.rotatable:
                return rect.w
            return (1 - r[rect.id]) * rect.w + r[rect.id] * rect.h

        def h_eff(rect: Rectangle):
            if not self.allow_rotation or not rect.rotatable:
                return rect.h
            return (1 - r[rect.id]) * rect.h + r[rect.id] * rect.w

        # safer Big-M values
        d = self.instance.kerf_delta
        W = self.instance.W
        sum_w = sum(rr.w for rr in self.instance.rectangles)
        sum_h = sum(rr.h for rr in self.instance.rectangles)

        if self.bigM_mode == "tight":
            # πιο “σφιχτά” αλλά ασφαλή
            Mx = W + max(0, sum_w // 2)
            My = max(H_ub, LB) + max(0, sum_h // 2)
        else:  # "loose"
            # πολύ μεγάλα, πάντα ασφαλή
            Mx = W + sum_w
            My = LB + sum_h

        # Objective
        model += H

        # Strip bounds
        for rect in self.instance.rectangles:
            model += x[rect.id] + w_eff(rect) <= W
            model += y[rect.id] + h_eff(rect) <= H

        # Non-overlap with kerf
        bL: Dict[Tuple[int, int], pulp.LpVariable] = {}
        bR: Dict[Tuple[int, int], pulp.LpVariable] = {}
        bU: Dict[Tuple[int, int], pulp.LpVariable] = {}
        bD: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for i, j in itertools.combinations(ids, 2):
            bL[(i, j)] = pulp.LpVariable(f"bL_{i}_{j}", 0, 1, cat="Binary")
            bR[(i, j)] = pulp.LpVariable(f"bR_{i}_{j}", 0, 1, cat="Binary")
            bU[(i, j)] = pulp.LpVariable(f"bU_{i}_{j}", 0, 1, cat="Binary")
            bD[(i, j)] = pulp.LpVariable(f"bD_{i}_{j}", 0, 1, cat="Binary")

            Ri = next(rct for rct in self.instance.rectangles if rct.id == i)
            Rj = next(rct for rct in self.instance.rectangles if rct.id == j)
            wi, hi = w_eff(Ri), h_eff(Ri)
            wj, hj = w_eff(Rj), h_eff(Rj)

            model += bL[(i, j)] + bR[(i, j)] + bU[(i, j)] + bD[(i, j)] == 1

            model += x[i] + wi + d <= x[j] + Mx * (1 - bL[(i, j)])
            model += x[j] + wj + d <= x[i] + Mx * (1 - bR[(i, j)])
            model += y[i] + hi + d <= y[j] + My * (1 - bD[(i, j)])
            model += y[j] + hj + d <= y[i] + My * (1 - bU[(i, j)])

        # Forbidden zones (optional)
        if self.instance.forbidden_zones:
            zL: Dict[Tuple[int, int], pulp.LpVariable] = {}
            zR: Dict[Tuple[int, int], pulp.LpVariable] = {}
            zU: Dict[Tuple[int, int], pulp.LpVariable] = {}
            zD: Dict[Tuple[int, int], pulp.LpVariable] = {}
            for rid in ids:
                for k, (zx, zy, zw, zh) in enumerate(self.instance.forbidden_zones):
                    key = (rid, k)
                    zL[key] = pulp.LpVariable(f"fzL_{rid}_{k}", 0, 1, cat="Binary")
                    zR[key] = pulp.LpVariable(f"fzR_{rid}_{k}", 0, 1, cat="Binary")
                    zU[key] = pulp.LpVariable(f"fzU_{rid}_{k}", 0, 1, cat="Binary")
                    zD[key] = pulp.LpVariable(f"fzD_{rid}_{k}", 0, 1, cat="Binary")
                    Rct = next(rct for rct in self.instance.rectangles if rct.id == rid)
                    wi, hi = w_eff(Rct), h_eff(Rct)

                    model += zL[key] + zR[key] + zU[key] + zD[key] == 1
                    # left of zone
                    model += x[rid] + wi <= zx + Mx * (1 - zL[key])
                    # right of zone
                    model += zx + zw <= x[rid] + Mx * (1 - zR[key])
                    # under zone (use My)
                    model += y[rid] + hi + d <= zy + My * (1 - zD[key])
                    # over zone (use My)
                    model += zy + zh + d <= y[rid] + My * (1 - zU[key])

        # Strengthen with lower bounds
        model += H >= self.instance.area_lb()
        model += H >= self.instance.maxh_lb(self.allow_rotation)

        # Mild symmetry break: anchor tallest at x=0
        tallest = max(self.instance.rectangles,
                      key=lambda rc: max(rc.h, rc.w) if self.allow_rotation else rc.h)
        model += x[tallest.id] == 0

        # Warm-start (no hard guide when kerf>0)
        if self.warm_start:
            by_id = {p.rect_id: p for p in hsol.placements}
            for rid in ids:
                p = by_id.get(rid)
                if p is None:
                    continue
                x[rid].setInitialValue(int(p.x))
                y[rid].setInitialValue(int(p.y))
                if self.allow_rotation and self.instance.rectangles[rid-1].rotatable:
                    try:
                        r[rid].setInitialValue(1 if p.rotated else 0)
                    except Exception:
                        pass
            if self.instance.kerf_delta == 0 and self.guide_radius > 0:
                R = int(self.guide_radius)
                for rid in ids:
                    p = by_id.get(rid)
                    if p is None:
                        continue
                    model += x[rid] >= max(0, p.x - R)
                    model += x[rid] <= p.x + R
                    model += y[rid] >= max(0, p.y - R)
                    model += y[rid] <= p.y + R + H * 0  # keep integrality

        # Solve
        cmd = pulp.PULP_CBC_CMD(msg=False, timeLimit=(self.time_limit or None))
        status = model.solve(cmd)

        # Build solution
        placements: List[Placement] = []
        for rect in self.instance.rectangles:
            xi = int(pulp.value(x[rect.id]) or 0)
            yi = int(pulp.value(y[rect.id]) or 0)
            if self.allow_rotation and rect.rotatable:
                ri = int(round(pulp.value(r[rect.id]) or 0))
                wv = rect.h if ri == 1 else rect.w
                hv = rect.w if ri == 1 else rect.h
                rotated = bool(ri)
            else:
                wv, hv, rotated = rect.w, rect.h, False
            placements.append(Placement(rect.id, xi, yi, int(wv), int(hv), rotated))

        H_val = int(pulp.value(H) or max((p.y + p.h_eff) for p in placements))
        
        status_code = model.status
        status_str = getattr(pulp, "LpStatus", {}).get(status_code, str(status_code))
        sol = Solution(H=H_val, placements=placements,
                       method="MILP-Coordinate-Extended",
                       optimality=status_str)
        return sol
