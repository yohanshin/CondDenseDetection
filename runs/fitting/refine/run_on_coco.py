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

CDD_MODEL = "diffusion-ViT-Large-verts_306_2025-06-04-21-34-45"
DATA_DIR = "/large_experiments/3po/data/images/coco-train-2014-pruned/train2014"
CDD_RESULTS_DIR = f"/checkpoint/soyongshin/results/CondDenseDetection/coco-train-2014/{CDD_MODEL}"
RESULTS_DIR = "/checkpoint/soyongshin/results/atlas_refinement/coco-train-2014-pruned"
INIT_ATLAS_RESULTS_DIR = '/large_experiments/3po/data/atlas_nlf+wilor/coco'
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
    atlas, pose_prior = load_ATLAS(lod="smplx", device=device)

    image_path_list = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])
    annot_path_list = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.data.pyd')])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    renderer = Renderer(1000, 1000, 1000, device=device, faces=atlas.faces.detach().cpu().numpy())
    for image_path, annot_path in (pbar := tqdm(zip(image_path_list, annot_path_list), 
                                                total=len(image_path_list), dynamic_ncols=True)):
        
        assert image_path.split('/')[-1].split('.')[0] == annot_path.split('/')[-1].split('.')[0]
        
        # Read CDD results
        cdd_results_pth = os.path.join(CDD_RESULTS_DIR, os.path.basename(annot_path))
        cdd_results = joblib.load(cdd_results_pth)
        
        # Read initial ATLAS parameters
        # TODO: Change this line if we have other initialization to use
        basename = os.path.basename(annot_path).split('.')[0].split('_')[-1]
        init_atlas_results_pth = glob(os.path.join(INIT_ATLAS_RESULTS_DIR, f'*{basename}*'))[0]
        init_atlas_results = joblib.load(init_atlas_results_pth)
        intrinsic = torch.from_numpy(init_atlas_results[0]['intrinsic']).unsqueeze(0).float().to(device)

        # Prepare fitting
        ldmks = np.stack([cdd_result["pred_joints2d"] for cdd_result in cdd_results])
        ldmks_conf = np.stack([cdd_result["pred_conf"] for cdd_result in cdd_results])
        ldmks_conf = ldmks_conf / ldmks_conf.max(1, keepdims=True)
        ldmks = torch.from_numpy(np.concatenate((ldmks, ldmks_conf), axis=-1)).float().to(device)
        centers = torch.from_numpy(np.stack([cdd_result["center"] for cdd_result in cdd_results])).float().to(device)
        scales = torch.from_numpy(np.stack([cdd_result["scale"] for cdd_result in cdd_results])).float().to(device) * 200.0
        init_params = {k: torch.from_numpy(
            np.stack([init_atlas_result['atlas_params'][k] for init_atlas_result in init_atlas_results])
            ).float().to(device) for k in init_atlas_results[0]['atlas_params']}
        
        # Run Fitting
        pred_params, losses = fitting_fn.run_fitting(atlas=atlas,
                                                     pose_prior=pose_prior,
                                                     init_params=init_params,
                                                     ldmks=ldmks,
                                                     intrinsic=intrinsic,
                                                     scales=scales,
                                                     optim_kwargs_list=optim_kwargs,
                                                     smplx2ldmks=smplx2ldmks)

        # Visualize
        cmap = plt.get_cmap("tab10")
        image = cv2.imread(image_path)
        renderer.initialize_camera_params(K=intrinsic, width=image.shape[1], height=image.shape[0])
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rend_image = image.copy()
        
        with torch.no_grad():
            verts = fitting_fn.atlas_forward(atlas, pred_params)
        
        trans_normed = pred_params["global_trans"].norm(dim=-1)
        order = torch.sort(trans_normed)[1].cpu().numpy()[::-1]

        for p_id in order:
            rend_image = renderer.render_mesh(verts[p_id], 
                                              background=rend_image.copy(), 
                                              colors=list(cmap(p_id))[:3], 
                                              alpha=0.8)
            
        image = np.concatenate((image, rend_image), axis=1)

        save_img_pth = os.path.join(RESULTS_DIR, os.path.basename(image_path))
        cv2.imwrite(save_img_pth, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        

if __name__ == '__main__':
    main()