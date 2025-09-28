
from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
import random
from .core import Instance, Solution

class Visualizer:
    """Οπτικοποίηση strip & τοποθετήσεων με matplotlib."""
    def draw(self, instance: Instance, solution: Solution,
             title: Optional[str] = None, save_path: Optional[str] = None,
             annotate: bool = True, show: bool = True):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, instance.W)
        ax.set_ylim(0, solution.H)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)

        rnd = random.Random(1234)
        for p in solution.placements:
            color = (rnd.random(), rnd.random(), rnd.random())
            rect = plt.Rectangle((p.x, p.y), p.w_eff, p.h_eff, edgecolor='black',
                                 facecolor=color, alpha=0.6)
            ax.add_patch(rect)
            if annotate:
                ax.text(p.x + p.w_eff/2, p.y + p.h_eff/2,
                        f"id={p.rect_id}\n{p.w_eff}×{p.h_eff}",
                        ha='center', va='center', fontsize=8)

        # strip outline
        ax.plot([0, instance.W, instance.W, 0, 0], [0, 0, solution.H, solution.H, 0], 'k-')

        # forbidden zones overlay
        for (zx, zy, zw, zh) in instance.forbidden_zones:
            rect = plt.Rectangle((zx, zy), zw, zh, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(zx+zw/2, zy+zh/2, "FORB", color='red', ha='center', va='center', fontsize=8)

        ax.grid(True, linestyle='--', alpha=0.3)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
        if show:
            plt.show()
        plt.close(fig)
