
from __future__ import annotations
from ..core import Instance, Solution

class BaseSolver:
    """Αφηρημένη βάση λύτη SPP."""
    def __init__(self, instance: Instance, allow_rotation: bool = True):
        self.instance = instance
        self.allow_rotation = allow_rotation
    def solve(self) -> Solution:
        raise NotImplementedError
