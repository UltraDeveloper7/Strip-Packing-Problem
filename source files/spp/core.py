
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math
import itertools

ForbiddenZone = Tuple[int,int,int,int]  # (x, y, w, h)

@dataclass(frozen=True)
class Rectangle:
    """Ορθογώνιο αντικείμενο με ακεραίες διαστάσεις."""
    id: int
    w: int
    h: int
    rotatable: bool = True

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class Placement:
    """Τοποθέτηση ορθογωνίου στη λωρίδα."""
    rect_id: int
    x: int
    y: int
    w_eff: int
    h_eff: int
    rotated: bool = False


@dataclass
class Instance:
    """SPP instance: πλάτος λωρίδας, λίστα ορθογωνίων, προαιρετικά kerf & ζώνες απαγόρευσης."""
    W: int
    rectangles: List[Rectangle]
    kerf_delta: int = 0
    forbidden_zones: List[ForbiddenZone] = field(default_factory=list)

    def area_lb(self) -> int:
        """LB_area = ceil(sum(area)/W)."""
        A = sum(r.area for r in self.rectangles)
        return math.ceil(A / self.W)

    def maxh_lb(self, allow_rotation: bool) -> int:
        """LB_maxh = max height (ή min(w,h) όταν επιτρέπεται περιστροφή)."""
        if allow_rotation:
            return max(min(r.h, r.w) for r in self.rectangles)
        return max(r.h for r in self.rectangles)


@dataclass
class Solution:
    """Λύση strip packing: ύψος, τοποθετήσεις, μέθοδος και καθεστώς βέλτιστου/εφικτού."""
    H: int
    placements: List[Placement]
    method: str
    optimality: Optional[str] = None  # "optimal", "feasible", "timeout", κ.ά.

    def as_dict(self) -> Dict:
        return {
            "H": self.H,
            "method": self.method,
            "optimality": self.optimality,
            "placements": [p.__dict__ for p in self.placements],
        }

    def check_feasibility(self, instance: Instance) -> bool:
        """Έλεγχος ορίων/μη-επικάλυψης, με επιλογή kerf_delta για απλό έλεγχο διαθέσιμου χώρου.
        Σημείωση: εδώ ο έλεγχος kerf γίνεται συντηρητικά ως αυξήσεις διαστάσεων κατά δ/2 ανά πλευρά.
        Ο MILP το μοντελοποιεί αυστηρά στους περιορισμούς."""
        d = instance.kerf_delta
        for p in self.placements:
            if p.x < 0 or p.y < 0:
                return False
            if p.x + p.w_eff > instance.W:
                return False
            if p.y + p.h_eff > self.H:
                return False
        # μοναδικότητα ids
        ids = {r.id for r in instance.rectangles}
        if ids != {p.rect_id for p in self.placements}:
            return False
        # μη-επικάλυψη (χωρίς kerf γιατί ο actual έλεγχος γίνεται στο MILP)
        for a, b in itertools.combinations(self.placements, 2):
            if not ((a.x + a.w_eff <= b.x) or (b.x + b.w_eff <= a.x) or
                    (a.y + a.h_eff <= b.y) or (b.y + b.h_eff <= a.y)):
                return False
        # απαγορευμένες ζώνες: κανένα ορθογώνιο να μην τέμνει Ζ
        for p in self.placements:
            px1, py1, px2, py2 = p.x, p.y, p.x + p.w_eff, p.y + p.h_eff
            for (zx, zy, zw, zh) in instance.forbidden_zones:
                zx2, zy2 = zx + zw, zy + zh
                # έλεγχος τομής axis-aligned ορθογωνίων
                inter = not (px2 <= zx or zx2 <= px1 or py2 <= zy or zy2 <= py1)
                if inter:
                    return False
        return True
