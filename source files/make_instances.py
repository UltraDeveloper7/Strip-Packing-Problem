from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Tuple, List, Iterable

def _clamp_range_pair(s: Tuple[int,int]) -> Tuple[int,int]:
    a,b = int(s[0]), int(s[1])
    if a > b: a,b = b,a
    return a,b

@dataclass
class GenConfig:
    out: str = "instances"
    W: int = 50
    n_list: Iterable[int] = (10, 20, 40)
    count: int = 10
    w_range: Tuple[int,int] = (3, 15)
    h_range: Tuple[int,int] = (3, 15)
    max_aspect: float = 3.0
    rot: str = "1"                 # "0" | "1" | "mix"
    kerf_list: Iterable[int] = (0, 1, 2)
    seed: int = 123

class SPPInstanceGenerator:
    """Γεννήτρια CSV instances για SPP — χωρίς argparse, έτοιμη για χρήση από GUI."""
    def __init__(self, cfg: GenConfig):
        # canonicalize
        cfg.w_range = _clamp_range_pair(cfg.w_range)
        cfg.h_range = _clamp_range_pair(cfg.h_range)
        self.cfg = cfg

    @staticmethod
    def sample_wh(W: int, w_range: Tuple[int,int], h_range: Tuple[int,int],
                  max_aspect: float, rng: random.Random) -> Tuple[int,int]:
        wmin, wmax = w_range
        hmin, hmax = h_range
        for _ in range(10000):
            w = rng.randint(wmin, wmax)
            h = rng.randint(hmin, hmax)
            if w <= W and max(w/h, h/w) <= max_aspect:
                return w, h
        # fallback: χαλάρωσε ή clamp-αρε
        return min(W, max(wmin, 1)), max(hmin, 1)

    @staticmethod
    def write_instance_csv(path: str, W: int, rects: List[Tuple[int,int,int,int]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"W,{W}\n")
            for (rid, w, h, rot) in rects:
                f.write(f"{rid},{w},{h},{rot}\n")

    def generate(self) -> tuple[int, str]:
        """Επιστρέφει (total_created, out_dir)."""
        c = self.cfg
        os.makedirs(c.out, exist_ok=True)
        base_rng = random.Random(c.seed)
        total = 0
        for n in c.n_list:
            for kerf in c.kerf_list:
                batch_seed = base_rng.randint(1, 10**9)
                for idx in range(c.count):
                    rng = random.Random(batch_seed + idx)
                    rects = []
                    for rid in range(1, n+1):
                        w, h = self.sample_wh(c.W, c.w_range, c.h_range, c.max_aspect, rng)
                        if c.rot == "0":
                            rot_flag = 0
                        elif c.rot == "1":
                            rot_flag = 1
                        else:
                            rot_flag = 1 if rng.random() < 0.5 else 0
                        rects.append((rid, w, h, rot_flag))
                    rot_in_name = 1 if c.rot in ("1", "mix") else 0
                    seed_in_name = batch_seed + idx
                    fname = f"W{c.W}_n{n}_rot{rot_in_name}_kerf{kerf}_seed{seed_in_name}.csv"
                    self.write_instance_csv(os.path.join(c.out, fname), c.W, rects)
                    total += 1
        return total, c.out
