import os
import random
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
