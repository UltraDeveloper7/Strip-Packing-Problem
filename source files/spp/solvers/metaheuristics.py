
from __future__ import annotations
from typing import List, Tuple, Literal, Optional
import random, math, time
from ..core import Instance, Rectangle, Placement, Solution
from .base import BaseSolver
from .heuristics import HeuristicSolver

class MetaheuristicSolver(BaseSolver):
    """GA / SA / Tabu πάνω σε representation (permutation + orientation bits)."""

    def __init__(self, instance: Instance, allow_rotation: bool = True,
                 strategy: Literal["GA", "SA", "Tabu"] = "GA",
                 time_limit: Optional[float] = 5.0,
                 population: int = 40, elite: int = 4):
        super().__init__(instance, allow_rotation)
        self.strategy = strategy
        self.time_limit = time_limit
        self.population = population
        self.elite = elite

    # --- κοινά βοηθητικά ---
    def _decode(self, perm: List[int], orient_bits: List[int]) -> Solution:
        """Αποκωδικοποίηση μέσω Skyline (γρήγορη και αποδοτική)."""
        # Δημιουργία reorder instance
        id_to_rect = {r.id: r for r in self.instance.rectangles}
        rects = []
        for rid, bit in zip(perm, orient_bits):
            r = id_to_rect[rid]
            rr = Rectangle(r.id, r.w, r.h, r.rotatable and self.allow_rotation)
            rects.append(rr)
        # προσωρινό instance μόνο για σειρά
        tmp_inst = Instance(self.instance.W, rects, kerf_delta=self.instance.kerf_delta,
                            forbidden_zones=self.instance.forbidden_zones)
        # Χρήση Skyline για γρήγορη τοποθέτηση (orientation: prefer_low=True)
        hs = HeuristicSolver(tmp_inst, allow_rotation=self.allow_rotation, policy="Skyline", guillotine=False)
        sol = hs.solve()
        sol.method = f"Meta-{self.strategy}-SkylineDecode"
        return sol

    def _random_solution(self) -> Tuple[List[int], List[int]]:
        ids = [r.id for r in self.instance.rectangles]
        perm = ids[:]
        random.shuffle(perm)
        bits = [random.randint(0, 1) if (self.allow_rotation and next(r for r in self.instance.rectangles if r.id==rid).rotatable) else 0
                for rid in perm]
        return perm, bits

    # --- GA ---
    def _run_ga(self) -> Solution:
        start = time.time()
        # αρχικοποίηση
        pool = []
        for _ in range(self.population):
            perm, bits = self._random_solution()
            sol = self._decode(perm, bits)
            pool.append((sol.H, perm, bits, sol))
        pool.sort(key=lambda t: t[0])
        best = pool[0][3]

        while time.time() - start < (self.time_limit or 5.0):
            # Ελιτισμός
            new_pool = pool[:self.elite]
            # Αναπαραγωγή
            while len(new_pool) < self.population:
                p1 = random.choice(pool[:max(2, self.population//2)])
                p2 = random.choice(pool[:max(2, self.population//2)])
                child_perm = self._ox_crossover(p1[1], p2[1])
                child_bits = self._bit_crossover(p1[2], p2[2])
                # μετάλλαξη
                self._mutate(child_perm, child_bits, rate=0.1)
                sol = self._decode(child_perm, child_bits)
                new_pool.append((sol.H, child_perm, child_bits, sol))
            new_pool.sort(key=lambda t: t[0])
            pool = new_pool
            if pool[0][0] < best.H:
                best = pool[0][3]
        return best

    @staticmethod
    def _ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [None]*n
        child[a:b+1] = p1[a:b+1]
        fill = [x for x in p2 if x not in child]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]; idx += 1
        return child

    @staticmethod
    def _bit_crossover(b1: List[int], b2: List[int]) -> List[int]:
        return [random.choice([u,v]) for u,v in zip(b1,b2)]

    @staticmethod
    def _mutate(perm: List[int], bits: List[int], rate: float = 0.1):
        n = len(perm)
        if random.random() < rate and n >= 2:
            i, j = random.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]
        for k in range(n):
            if random.random() < rate*0.5:
                bits[k] ^= 1

    # --- SA ---
    def _run_sa(self) -> Solution:
        start_perm, start_bits = self._random_solution()
        cur_sol = self._decode(start_perm, start_bits)
        best = cur_sol
        T0, Tend = 1.0, 1e-3
        max_iter = 10000
        t0 = time.time()
        for it in range(max_iter):
            if self.time_limit and time.time() - t0 > self.time_limit: break
            cand_p, cand_b = start_perm[:], start_bits[:]
            # γειτονιά: swap δύο θέσεις και flip ενός bit
            i, j = random.sample(range(len(cand_p)), 2)
            cand_p[i], cand_p[j] = cand_p[j], cand_p[i]
            k = random.randrange(len(cand_b))
            cand_b[k] ^= 1
            cand_sol = self._decode(cand_p, cand_b)
            dH = cand_sol.H - cur_sol.H
            T = T0 * (Tend/T0) ** (it/max_iter)
            if dH < 0 or random.random() < math.exp(-dH / max(T,1e-6)):
                start_perm, start_bits, cur_sol = cand_p, cand_b, cand_sol
                if cur_sol.H < best.H:
                    best = cur_sol
        return best

    # --- Tabu ---
    def _run_tabu(self) -> Solution:
        perm, bits = self._random_solution()
        cur = self._decode(perm, bits)
        best = cur
        tabu = set()
        tenure = max(5, len(perm)//10)
        t0 = time.time()
        while not self.time_limit or (time.time() - t0) < self.time_limit:
            neighborhood = []
            for _ in range(30):
                i, j = random.sample(range(len(perm)), 2)
                key = (min(perm[i], perm[j]), max(perm[i], perm[j]))
                if key in tabu: continue
                cand_p = perm[:]; cand_p[i], cand_p[j] = cand_p[j], cand_p[i]
                cand_b = bits[:]
                if random.random() < 0.5:
                    k = random.randrange(len(bits))
                    cand_b[k] ^= 1
                cand = self._decode(cand_p, cand_b)
                neighborhood.append((cand.H, cand_p, cand_b, cand))
            if not neighborhood: break
            neighborhood.sort(key=lambda t: t[0])
            best_nei = neighborhood[0]
            perm, bits, cur = best_nei[1], best_nei[2], best_nei[3]
            tabu.add((min(perm[0], perm[-1]), max(perm[0], perm[-1])))
            if len(tabu) > tenure:
                tabu.pop()
            if cur.H < best.H:
                best = cur
        return best

    def solve(self) -> Solution:
        if self.strategy == "GA":
            return self._run_ga()
        if self.strategy == "SA":
            return self._run_sa()
        if self.strategy == "Tabu":
            return self._run_tabu()
        raise ValueError("Unknown metaheuristic strategy")
