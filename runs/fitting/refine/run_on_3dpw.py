import os
import argparse
import pickle
import torch
from loguru import logger

import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import joblib
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d import transforms as tfs

from llib.viz.renderer import Renderer
from llib.fitting.fitting_fn import FittingFunction
from ..tools.registry import load_ATLAS

CDD_MODEL = "encdec_Cond-ViT-Large-verts_306_2025-06-04-03-26-16"
DATA_DIR = "/large_experiments/3po/data/images/3dpw"
CDD_RESULTS_DIR = f"/checkpoint/soyongshin/results/CondDenseDetection/threedpw/{CDD_MODEL}"
RESULTS_DIR = "/checkpoint/soyongshin/results/atlas_refinement/threedpw"
RESULTS_DIR = "/checkpoint/soyongshin/results/atlas_refinement/threedpw_w_smoothing"
INIT_ATLAS_RESULTS_DIR = '/large_experiments/3po/data/atlas_nlf+wilor/threedpw'
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
        "lw_shape": 0.004,
        "lw_shape_consist": 10.0,
        "lw_smoothing": 100.0,
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
        "lw_shape_consist": 10.0,
        "lw_smoothing": 250.0,
    },
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))
    
    fitting_fn = FittingFunction(device=device)
    smplx2ldmks = joblib.load(SMPLX2LDMKS_PTH)
    smplx2ldmks = torch.argmax(smplx2ldmks, axis=1).to(device)
    atlas, pose_prior = load_ATLAS(lod="smplx", device=device)

    renderer = Renderer(1000, 1000, 1000, device=device, faces=atlas.faces.detach().cpu().numpy())
    for dset in (dbar := tqdm(['train', 'test', 'validation'], dynamic_ncols=True)):
        dbar.set_description_str(f'3DPW {dset} set')
        annot_fldr = os.path.join(DATA_DIR, 'sequenceFiles', dset)
        annot_file_names = sorted(os.listdir(annot_fldr))

        for annot_file_name in (pbar := tqdm(annot_file_names, dynamic_ncols=True)):
            sequence = annot_file_name.replace('.pkl', '')
            annot_file_pth = os.path.join(annot_fldr, annot_file_name)
            annot = pickle.load(open(annot_file_pth, 'rb'), encoding='latin1')
            num_people = len(annot['poses'])
            
            image_pth_list = sorted(glob(os.path.join(DATA_DIR, 'imageFiles', sequence, "*.jpg")))
            verts_multiperson = []
            frames_multiperson = []
            for p_id in range(num_people):
                cdd_results_pth = os.path.join(CDD_RESULTS_DIR, f'{sequence}_P{p_id}.data.pyd')
                cdd_results = joblib.load(cdd_results_pth)

                # Read initial ATLAS parameters
                # TODO: Change this line if we have other initialization to use
                init_atlas_results_pth = os.path.join(INIT_ATLAS_RESULTS_DIR, f'{sequence}_P{p_id}.data.pyd')
                init_atlas_results = joblib.load(init_atlas_results_pth)
                intrinsic = torch.from_numpy(init_atlas_results[0]['intrinsic']).unsqueeze(0).float().to(device)
                
                ldmks = cdd_results["pred_joints2d"]
                ldmks_conf = cdd_results["pred_conf"]
                ldmks_conf = ldmks_conf / ldmks_conf.max(1, keepdims=True)
                ldmks = torch.from_numpy(np.concatenate((ldmks, ldmks_conf), axis=-1)).float().to(device)
                scales = torch.from_numpy(cdd_results["scale"].max(1)).float().to(device) * 200.0
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

                with torch.no_grad():
                    verts = fitting_fn.atlas_forward(atlas, pred_params)
                    
                verts_multiperson.append(verts.cpu())
                frames_multiperson.append(cdd_results["frames"])
            
            image = cv2.imread(image_pth_list[0])
            renderer.initialize_camera_params(K=intrinsic, width=image.shape[1], height=image.shape[0])
            cmap = plt.get_cmap("tab10")
            
            video_pth = os.path.join(RESULTS_DIR, f'{sequence}.mp4')
            writer = imageio.get_writer(video_pth, fps=30, mode="I", format="FFMPEG", macro_block_size=None)
            
            rotate = tfs.axis_angle_to_matrix(torch.Tensor([0, np.pi/2, 0])).float().to(device)
            for img_i, image_pth in tqdm(enumerate(image_pth_list), total=len(image_pth_list), leave=False):
                cvimg = cv2.cvtColor(cv2.imread(image_pth), cv2.COLOR_BGR2RGB)
                rend_image = cvimg.copy()
                side_image = np.ones_like(rend_image) * 255
                for pi in range(num_people):
                    if img_i in frames_multiperson[pi]:
                        _i = np.where(frames_multiperson[pi] == img_i)[0][0].item()
                
                        verts = verts_multiperson[pi][_i].to("cuda")
                        rend_image = renderer.render_mesh(verts, 
                                                          background=rend_image.copy(), 
                                                          colors=list(cmap(pi))[:3], 
                                                          alpha=1.0)

                        verts_mean = verts.mean(0, keepdims=True)
                        verts_norm = verts - verts_mean
                        verts_rotated = (rotate @ verts_norm.T).T
                        verts_rotated = verts_rotated + verts_mean
                        side_image = renderer.render_mesh(verts_rotated, 
                                                          background=side_image.copy(), 
                                                          colors=list(cmap(pi))[:3], 
                                                          alpha=1.0)
                        
                cvimg = np.concatenate((cvimg, rend_image, side_image), axis=1)
                writer.append_data(cv2.resize(cvimg, None, fx=0.5, fy=0.5))
            writer.close()


if __name__ == "__main__":
    main()