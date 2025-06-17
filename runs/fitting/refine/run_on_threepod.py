import os
import argparse
import yaml
import torch
from loguru import logger

import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from llib.viz.renderer import Renderer
from llib.fitting.fitting_fn import FittingFunction
from ..tools.registry import load_ATLAS

CDD_MODEL = "encdec_Cond-ViT-Large-verts_306_2025-06-04-03-26-16"
DATA_DIR = "/large_experiments/3po/data/images/coco-train-2014-pruned/train2014"
CDD_RESULTS_DIR = f"/checkpoint/soyongshin/results/CondDenseDetection/threepod/{CDD_MODEL}"
THREEPO_PRED_PTH = '/large_experiments/3po/threepo_airstore/0528_threepod_pseudo/threepod_train.pkl'
DATA_MAPPER_PTH = '/checkpoint/soyongshin/data/threepod/key_path_mappers/2025-06-02.pkl'
SMPLX2LDMKS_PTH = "/large_experiments/3po/model/smplx/verts_306.pkl"

optim_kwargs = [
    # Stage 1
    {
        "params": "all",
        
        # Optimizer
        "num_steps": 750,
        "lr": 0.003,
        "optim_type": "adamw",
        "lr_scheduler": "cosine",
        
        # Loss weights
        "lw_ldmks": 5.0,
        "lw_reg": 0.1,
        "lw_pose": 0.02,
        "lw_shape": 0.004,
    },

    # Stage 2
    {
        "params": "all",

        # Optimizer
        "num_steps": 250,
        "lr": 0.0003,
        "optim_type": "adamw",
        "lr_scheduler": "cosine",
        
        # Loss weights
        "lw_ldmks": 5.0,
        "lw_reg": 0.01,
        "lw_pose": 0.005,
        "lw_shape": 0.001,
    },
]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))

    fitting_fn = FittingFunction(device=device)
    smplx2ldmks = joblib.load(SMPLX2LDMKS_PTH)
    smplx2ldmks = torch.argmax(smplx2ldmks, axis=1).to(device)

    # Load ThreePO pred
    threepo_pred_list = joblib.load(THREEPO_PRED_PTH)

    # Load key-filepath path mapper
    data_mapper = joblib.load(DATA_MAPPER_PTH)

    for threepo_pred in threepo_pred_list:
        key = threepo_pred['key']
        try: 
            image_path = data_mapper[key]
        except: 
            print(f"No key {key} exists. Skip !")
        
        basename = os.path.basename(image_path)
        if 'gaia' in basename:
            cdd_results_pth = os.path.join(CDD_RESULTS_DIR, 'gaia', image_path.split('/')[-3], basename.replace(".jpg", ".data.pyd"))
        elif basename.startswith('sa_'):
            cdd_results_pth = os.path.join(CDD_RESULTS_DIR, 'SA1B', basename.replace(".jpg", ".data.pyd"))
        elif 'egoexo' in image_path.split('/')[-2]:
            cdd_results_pth = os.path.join(CDD_RESULTS_DIR, 'egoexo4d', basename.replace(".jpg", ".data.pyd"))
        else:
            cdd_results_pth = os.path.join(CDD_RESULTS_DIR, basename.replace(".jpg", ".data.pyd"))
        
        if not os.path.exists(cdd_results_pth):
            continue
        
        cdd_results = joblib.load(cdd_results_pth)

        n_people = len(threepo_pred['annotations'])
        n_det_people = len(cdd_results['predictions'])
        
        if n_people != n_det_people:
            import pdb; pdb.set_trace()
        # for pid in range(n_people):
        #     person_id = threepo_pred['annotations'][pid]['person_id']
        #     pred_atlas_params = threepo_pred['annotations'][pid]['atlas_params']
        

if __name__ == '__main__':
    main()