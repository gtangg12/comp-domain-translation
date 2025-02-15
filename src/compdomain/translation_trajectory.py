import numpy as np
import torch


from compdomain.translation import Translation


def check_valid_traj(traj: str, domains: list[str]):
    traj = set(traj)
    return all([i in traj for i in range(len(domains))])


class TranslationTrajectory:
    """
    """
    def __init__(self, traj: str, domains: list[str]):
        """
        """
        assert check_valid_traj(traj, domains)

    def __call__(self, image):
        """
        """
        pass