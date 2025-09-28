
from __future__ import annotations
from typing import List, Tuple, Literal, Optional
from ..core import Instance, Rectangle, Placement, Solution
from .base import BaseSolver

class HeuristicSolver(BaseSolver):
    """Ευρετικές NFDH (guillotine-friendly) & Skyline-like (non-guillotine)."""

    def __init__(self, instance: Instance, allow_rotation: bool = True,
                 policy: Literal["NFDH", "Skyline"] = "NFDH",
                 guillotine: bool = False):
        super().__init__(instance, allow_rotation)
        self.policy = policy
        self.guillotine = guillotine  # αν True, επιβάλλεται χρήση NFDH (ράφια)

    @staticmethod
    def _orient(rect: Rectangle, allow_rotation: bool, prefer_low: bool = False) -> Tuple[int, int, bool]:
        if not allow_rotation or not rect.rotatable:
            return rect.w, rect.h, False
        if prefer_low:
            # Επιλογή orientation με μικρότερο h
            if rect.h <= rect.w:
                return rect.w, rect.h, False
            else:
                return rect.h, rect.w, True
        return rect.w, rect.h, False

    def _nfdh(self) -> Solution:
        W = self.instance.W
        d = self.instance.kerf_delta
        items = []
        for r in self.instance.rectangles:
            w1, h1, _ = self._orient(r, self.allow_rotation, prefer_low=True)
            items.append((r, h1))
        items.sort(key=lambda t: t[1], reverse=True)

        placements: List[Placement] = []
        x = 0
        shelf_y = 0
        shelf_h = 0
        first_in_shelf = True
        for (r, _) in items:
            w_eff, h_eff, rotated = self._orient(r, self.allow_rotation, prefer_low=True)
            if w_eff > W:
                raise ValueError(f"Rectangle {r.id} width {w_eff} exceeds strip width {W}")
            # προσθήκη kerf οριζόντια μεταξύ διαδοχικών στο ράφι
            extra = 0 if first_in_shelf else d
            if x + extra + w_eff <= W:
                x += 0 if first_in_shelf else d
                first_in_shelf = False
                placements.append(Placement(r.id, x, shelf_y, w_eff, h_eff, rotated))
                x += w_eff
                shelf_h = max(shelf_h, h_eff)
            else:
                # νέο ράφι (κάθετη guillotine τομή)
                shelf_y += shelf_h + (d if shelf_h > 0 else 0)  # κατακόρυφος kerf μεταξύ ραφιών
                x = 0
                shelf_h = h_eff
                first_in_shelf = False
                placements.append(Placement(r.id, x, shelf_y, w_eff, h_eff, rotated))
                x += w_eff

        H = shelf_y + shelf_h
        return Solution(H=H, placements=placements, method="Heuristic-NFDH", optimality="feasible")

    def _skyline(self) -> Solution:
        if self.guillotine:
            # Αν ζητηθεί guillotine, ανάγεται σε NFDH (ράφια ~ οριζόντιες πλήρεις τομές).
            return self._nfdh()

        W = self.instance.W
        skyline = [(0, W, 0)]  # (x_start, width, height)
        placements: List[Placement] = []
        rects = list(self.instance.rectangles)
        rects.sort(key=lambda r: max(r.h, r.w) if self.allow_rotation else r.h, reverse=True)

        def find_position(w_eff: int, h_eff: int):
            best = None
            for i in range(len(skyline)):
                xs, width, y = skyline[i]
                if width < w_eff:
                    continue
                cand = (xs, y, i)
                if best is None or cand[1] < best[1] or (cand[1] == best[1] and cand[0] < best[0]):
                    best = cand
            return best

        def update_skyline(x: int, w_eff: int, y: int, h_eff: int):
            x_end = x + w_eff
            new_seg = (x, w_eff, y + h_eff)
            updated = []
            i = 0
            while i < len(skyline):
                xs, width, sy = skyline[i]
                xe = xs + width
                if xe <= x or xs >= x_end:
                    updated.append(skyline[i])
                else:
                    if xs < x:
                        updated.append((xs, x - xs, sy))
                    if xe > x_end:
                        updated.append((x_end, xe - x_end, sy))
                i += 1
            updated.append(new_seg)
            updated.sort(key=lambda t: t[0])
            merged = []
            for seg in updated:
                if not merged:
                    merged.append(seg)
                else:
                    px, pw, py = merged[-1]
                    sx, sw, sy = seg
                    if py == sy and px + pw == sx:
                        merged[-1] = (px, pw + sw, py)
                    else:
                        merged.append(seg)
            return merged

        for r in rects:
            w_eff, h_eff, rotated = self._orient(r, self.allow_rotation, prefer_low=True)
            if w_eff > W:
                raise ValueError(f"Rectangle {r.id} width {w_eff} exceeds strip width {W}")
            pos = find_position(w_eff, h_eff)
            if pos is None:
                x = 0
                y = max(y for _, _, y in skyline)
            else:
                x, y, _ = pos
            placements.append(Placement(r.id, x, y, w_eff, h_eff, rotated))
            skyline = update_skyline(x, w_eff, y, h_eff)

        H = max((p.y + p.h_eff) for p in placements) if placements else 0
        return Solution(H=H, placements=placements, method="Heuristic-Skyline", optimality="feasible")

    def solve(self) -> Solution:
        if self.policy == "NFDH":
            return self._nfdh()
        elif self.policy == "Skyline":
            return self._skyline()
        raise ValueError("Unknown heuristic policy")
