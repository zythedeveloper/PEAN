import os, torch, random, string
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
NOTEBOOKS_DIR = os.path.join(CURRENT_DIR, "notebooks")
LOGS_DIR = os.path.join(CURRENT_DIR, "runs")
CKPT_DIR = os.path.join(CURRENT_DIR, "ckpt")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("device:", device)
    return device


def filter_str(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    return str_