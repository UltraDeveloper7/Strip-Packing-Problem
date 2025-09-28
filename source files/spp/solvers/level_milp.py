from __future__ import annotations
from typing import Optional, List

try:
    import pulp
except Exception:
    pulp = None

from ..core import Instance, Rectangle, Placement, Solution
from .base import BaseSolver


class LevelMilpSolver(BaseSolver):
    """Shelf/Level MILP (ASCII-only, linearized, rotation supported).
    Variables per level ell:
      - u[ell] in {0,1}: level enabled
      - s[ell] integer:  level height
      - zH[i,ell] in {0,1}: item i at level ell without rotation
      - zV[i,ell] in {0,1}: item i at level ell with rotation (only if allow_rotation)
    Constraints:
      - Each item to exactly one level (sum zH+zV = 1).
      - Width per level: sum(w_i*zH + h_i*zV) + d*(count-1) <= W * u[ell].
      - Height per level: s[ell] >= h_i*zH and s[ell] >= w_i*zV for all i.
      - Non-increasing s[ell] to break symmetry.
      - H = sum s[ell], plus lower bounds.
    """
    def __init__(self, instance: Instance, allow_rotation: bool = True,
                 max_levels: Optional[int] = None, time_limit: Optional[int] = None):
        super().__init__(instance, allow_rotation)
        self.max_levels = max_levels
        self.time_limit = time_limit

    def solve(self) -> Solution:
        if pulp is None:
            raise RuntimeError("PuLP is not available. Please `pip install pulp`.")

        n = len(self.instance.rectangles)
        L = self.max_levels or n
        ids = [r.id for r in self.instance.rectangles]
        W = self.instance.W
        d = self.instance.kerf_delta

        model = pulp.LpProblem("SPP_SHELF_LEVEL_LINEAR", pulp.LpMinimize)

        # Variables
        u = pulp.LpVariable.dicts("u", range(L), lowBound=0, upBound=1, cat="Binary")
        s = pulp.LpVariable.dicts("s", range(L), lowBound=0, cat="Integer")
        H = pulp.LpVariable("H", lowBound=0, cat="Integer")

        zH = pulp.LpVariable.dicts("zH", (ids, range(L)), lowBound=0, upBound=1, cat="Binary")
        if self.allow_rotation:
            zV = pulp.LpVariable.dicts("zV", (ids, range(L)), lowBound=0, upBound=1, cat="Binary")
        else:
            zV = {rid: {ell: 0 for ell in range(L)} for rid in ids}

        # Objective
        model += H

        # Assignment: sum_{ell} (zH+zV) = 1
        for rid in ids:
            model += pulp.lpSum(zH[rid][ell] + (zV[rid][ell] if self.allow_rotation else 0)
                                for ell in range(L)) == 1

        # Width per level with kerf
        for ell in range(L):
            count_ell = pulp.lpSum(zH[rid][ell] + (zV[rid][ell] if self.allow_rotation else 0)
                                   for rid in ids)
            width_ell = pulp.lpSum(
                (next(r for r in self.instance.rectangles if r.id == rid).w) * zH[rid][ell]
                + (next(r for r in self.instance.rectangles if r.id == rid).h) * (zV[rid][ell] if self.allow_rotation else 0)
                for rid in ids
            )
            model += width_ell + d * (count_ell - 1) <= W * u[ell]

        # Height per level (max-type linearization)
        for ell in range(L):
            for rid in ids:
                rect = next(r for r in self.instance.rectangles if r.id == rid)
                model += s[ell] >= rect.h * zH[rid][ell]
                if self.allow_rotation:
                    model += s[ell] >= rect.w * zV[rid][ell]

        # Non-increasing heights to break symmetry
        for ell in range(L - 1):
            model += s[ell] >= s[ell + 1]

        # Total height and lower bounds
        model += H == pulp.lpSum(s[ell] for ell in range(L))
        model += H >= self.instance.area_lb()
        model += H >= self.instance.maxh_lb(self.allow_rotation)

        # Solve
        cmd = pulp.PULP_CBC_CMD(msg=False, timeLimit=(self.time_limit or None))
        status = model.solve(cmd)

        # Build placements: left-to-right within each level
        placements: List[Placement] = []
        y = 0
        for ell in range(L):
            s_val = int(pulp.value(s[ell]) or 0)
            if s_val <= 0:
                continue
            x = 0
            first = True
            # items of this level
            ids_on = []
            for rid in ids:
                if (pulp.value(zH[rid][ell]) or 0) > 0.5 or \
                   (self.allow_rotation and (pulp.value(zV[rid][ell]) or 0) > 0.5):
                    ids_on.append(rid)
            # order by effective height desc (for nicer drawing)
            def eff_h(rid: int) -> int:
                rect = next(rr for rr in self.instance.rectangles if rr.id == rid)
                zv = (pulp.value(zV[rid][ell]) or 0) if self.allow_rotation else 0
                return rect.w if (self.allow_rotation and zv > 0.5) else rect.h
            ids_on.sort(key=eff_h, reverse=True)

            for rid in ids_on:
                rect = next(rr for rr in self.instance.rectangles if rr.id == rid)
                rotated = False
                if self.allow_rotation:
                    zv = pulp.value(zV[rid][ell]) or 0
                    rotated = zv > 0.5
                wv = rect.h if rotated else rect.w
                hv = rect.w if rotated else rect.h
                if not first:
                    x += d
                placements.append(Placement(rid, x, y, int(wv), int(hv), rotated))
                x += int(wv)
                first = False
            y += s_val + (d if s_val > 0 else 0)

        H_val = max([p.y + p.h_eff for p in placements], default=0)
        
        status_code = model.status
        status_str = getattr(pulp, "LpStatus", {}).get(status_code, str(status_code))
        sol = Solution(H=int(H_val), placements=placements,
                       method="MILP-ShelfLevel", optimality=status_str)
        return sol
