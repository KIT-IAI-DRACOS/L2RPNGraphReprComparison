import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# agent and policy keys:
DO_NOTHING_AGENT = "do_nothing_agent"
RL_AGENT = "reinforcement_learning_agent"
HIGH_LEVEL_AGENT = "high_level_agent"
DO_NOTHING_POLICY = "do_nothing_policy"
RL_POLICY = "reinforcement_learning_policy"
HIGH_LEVEL_POLICY = "high_level_policy"

class Style:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


SEED = 42


def set_seed(seed):
    global SEED
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


set_seed(42)