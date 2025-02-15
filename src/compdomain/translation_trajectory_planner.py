import numpy as np
import torch

from compdomain.translation_trajectory import TranslationTrajectory


class TranslationTrajectoryPlanner:
    """
    """
    def __init__(self, domains: list[str]):
        """
        """
        self.domains = domains

    def __call__(self, influence: list[float]) -> TranslationTrajectory:
        """
        """
        assert len(self.domains) == len(influence) and np.isclose(sum(influence), 1.0)