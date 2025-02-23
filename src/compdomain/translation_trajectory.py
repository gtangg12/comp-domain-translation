from abc import ABC, abstractmethod

import numpy as np
import torch
from jaxtyping import Float

from compdomain.translation import TranslationGuidance, Translation


class TranslationTrajectory(ABC):
    """
    """
    def __init__(self, guidance: TranslationGuidance, prompt: str, source_prompt: str, target_prompts: list[str]):
        """
        """
        self.guidance = guidance
        self.prompt = prompt
        self.source_prompt = source_prompt
        self.target_prompts = target_prompts

    @abstractmethod
    def __call__(self, image: Float[np.ndarray, "H W 3"], **kwargs):
        """
        """
        pass


class TranslationTrajectoryFixed(TranslationTrajectory):
    """
    """
    def __init__(self, traj: str, **kwargs):
        super().__init__(**kwargs)
        assert all([0 <= i < len(self.target_prompts) for i in traj]), \
            "Trajectory is denoted by [0...(k-1)]* where k is number of target domains"

    def __call__(self, image: Float[np.ndarray, "H W 3"]):
        """
        """
        #TODO execute fixed translation traj
        pass


class TranslationTrajectoryDynamic(TranslationTrajectory):
    """
    """
    def __init__(self, influences: list[float], **kwargs):
        super().__init__(**kwargs)
        assert len(influences) == len(self.target_prompts), \
            "Number of influence weights must be equal to number of target domains"

    def __call__(self, image: Float[np.ndarray, "H W 3"]):
        """
        """
        #TODO dynamically adjust using clip
        pass


if __name__ == '__main__':
    pass
    # TODO test here