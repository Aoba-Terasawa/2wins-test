import os
import random
import numpy as np
import torch

def fix_seed(seed=42):
    """
    再現性を確保するためにシードを固定する関数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed fixed to {seed}")
