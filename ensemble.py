import argparse
import sys
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings('ignore')

from ensembles import *


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./experiments/ensemble.yaml")

args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = OmegaConf.load(f)

if cfg.root_path not in sys.path:
    sys.path.append(cfg.root_path)

if cfg.ensemble_type == "hard_voting":
    print("======================hard_voting======================")
    hard_voting(cfg)
if cfg.ensemble_type == "soft_voting":
    print("======================soft_voting======================")
    soft_voting(cfg)