import os
from copy import deepcopy

import numpy as np
import torch

import sys
sys.path.append(os.path.abspath(__file__) + "/../../third_party/SDS-Bridge/2D_experiments")
from guidance import Guidance, GuidanceConfig
sys.path.pop()


class Translation:
    """
    """
    def __init__(self, image):
        """
        """
        self.image = deepcopy(image)

    def step(self, domain: str):
        """
        """
        pass


if __name__ == "__main__":
    pass