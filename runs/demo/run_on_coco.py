import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import joblib
import scipy.io
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from configs import constants as _C
from runs.tools import util_fn
from runs.tools.registry import load_checkpoint, load_model

DATA_DIR = "/large_experiments/3po/data/images/coco-train-2014-pruned/train2014"