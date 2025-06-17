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
import hydra
import imageio
from smplx import SMPL
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d import transforms as tfs

from llib.fitting.fitting_fn import FittingFunction
from ..demo.run_on_3dpw import prepare_labels


CDD_MODEL = "encdec_Cond-ViT-Large-verts_306_2025-06-04-03-26-16"
DATA_DIR = "/large_experiments/3po/data/images/3dpw"
CDD_RESULTS_DIR = f"/checkpoint/soyongshin/results/CondDenseDetection/threedpw/{CDD_MODEL}"
RESULTS_DIR = "/checkpoint/soyongshin/results/misc/sensitivity"
INIT_ATLAS_RESULTS_DIR = '/large_experiments/3po/data/atlas_nlf+wilor/threedpw'
SMPLX2LDMKS_PTH = "/large_experiments/3po/model/smplx/verts_306.pkl"
SMPL_MODEL_DIR = "/large_experiments/3po/model/smpl"

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

CUR_PATH = os.getcwd()
from omegaconf import OmegaConf, DictConfig
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/hydra"), config_name="config.yaml")
def run_cdd(cfg: DictConfig):
    from runs.tools.registry import load_checkpoint, load_model
    from runs.tools import util_fn
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    OmegaConf.register_new_resolver("leq", lambda a, b: a >= b)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    # Load model
    work_dir = osp.join(CUR_PATH, cfg.work_dir, 'train', 'regressor_2d', cfg.exp_name)
    ckpt_dir = os.path.join(work_dir, 'checkpoints')
    model = load_model(cfg, device)
    model = load_checkpoint(model, ckpt_dir)
    model.eval()

    body_model = {
        k: SMPL(SMPL_MODEL_DIR, gender=k, num_betas=10)
        for k in ["male", "female"]
    }

    trg_seq = "courtyard_backpack_00"
    trg_frame = 67
    method = "head"
    use_full_cond = True
    for dset in (dbar := tqdm(['train', 'test', 'validation'], dynamic_ncols=True)):
        dbar.set_description_str(f'3DPW {dset} set')
        annot_fldr = os.path.join(DATA_DIR, 'sequenceFiles', dset)
        annot_file_names = sorted(os.listdir(annot_fldr))
        
        for annot_file_name in (pbar := tqdm(annot_file_names, dynamic_ncols=True)):
            sequence = annot_file_name.replace('.pkl', '')
            if sequence != trg_seq: continue
            
            annot_file_pth = os.path.join(annot_fldr, annot_file_name)
            annot = pickle.load(open(annot_file_pth, 'rb'), encoding='latin1')
            
            p_id = 0
            vis_pth = os.path.join(RESULTS_DIR, f'CDD_{sequence}_P{p_id}_{method}{"_full_cond" * int(use_full_cond)}.mp4')
            save_pth = os.path.join(RESULTS_DIR, f'CDD_{sequence}_P{p_id}_{method}{"_full_cond" * int(use_full_cond)}.data.pyd')
            
            batch = prepare_labels(annot, p_id, body_model)
            image_pth_list = sorted(glob(os.path.join(DATA_DIR, 'imageFiles', sequence, "*.jpg")))
            image_pth_list = np.array(image_pth_list)[batch["valid_frames"]]
            
            center = batch["centers"][trg_frame]
            scale = batch["scales"][trg_frame]
            
            scale_min = scale * 0.4
            image = cv2.imread(image_pth_list[trg_frame])
            cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            joints2d = batch["joints2d"][trg_frame]
            img_tensors, conds, cond_masks, xyxys, centers, scales = [], [], [], [], [], []
            for i in range(60):
                alpha = i / 61
                # alpha = (61 - i) / 61
                if method == "pelvis":
                    trg_center = joints2d[[11, 12]].mean(0).cpu().numpy()
                elif method == "head":
                    trg_center = joints2d[[0, 1, 2, 3, 4]].mean(0).cpu().numpy()
                
                s = alpha * scale_min + (1 - alpha) * scale
                c = alpha * trg_center + (1 - alpha) * center
                centers.append(c)
                scales.append(s)
                    
                img_tensor = util_fn.process_cvimg(cvimg.copy(), c, s, cfg.image_size)
                img_tensors.append(torch.from_numpy(img_tensor).float().to(device))
                xyxy = util_fn.cs2xyxy(c, s)
                cond = joints2d[:, :2]
                cond_mask = np.zeros(joints2d.shape[0]).astype(bool)
                cond_mask[-6:] = 1   # Ignore feet 
                cond = util_fn.process_cond(cond, c, s, cfg.image_size)
                if not use_full_cond:
                    _j2d = joints2d[:, :2].numpy()
                    out_x = np.logical_or(_j2d[:, 0] < xyxy[0], _j2d[:, 0] > xyxy[2])
                    out_y = np.logical_or(_j2d[:, 1] < xyxy[1], _j2d[:, 1] > xyxy[3])
                    out = np.logical_or(out_x, out_y)
                    cond_mask[out] = 1
                
                cond_masks.append(torch.from_numpy(cond_mask).bool().to(device))
                conds.append(torch.from_numpy(cond).float().to(device))
                xyxys.append(xyxy)

            img_tensors = torch.stack(img_tensors)
            conds = torch.stack(conds)
            cond_masks = torch.stack(cond_masks)
            xyxys = np.stack(xyxys)
            
            full_pred_joints2d = []
            full_pred_conf = []
            bs = 4
            for start in tqdm(range(0, img_tensors.shape[0], bs), desc="Inference ...", leave=False):
                end = min(img_tensors.shape[0], start + bs)
                with torch.no_grad():
                    if "xCond" in cfg.model.name:
                        pred = model._forward(img_tensors[start:end], cond=None, cond_mask=None)
                    else:
                        pred = model._forward(img_tensors[start:end], cond=conds[start:end], cond_mask=cond_masks[start:end])

                joints2d = pred["joints2d"].cpu().numpy()
                conf = pred["conf"].cpu().numpy()
                full_pred_joints2d.append(joints2d)
                full_pred_conf.append(conf)
            
            joints2d = np.concatenate(full_pred_joints2d)
            conf = np.concatenate(full_pred_conf)
            joints2d = np.concatenate((joints2d, np.ones_like(joints2d[..., :1])), axis=-1)

            for fi, (j2d, center, scale) in enumerate(zip(joints2d, centers, scales)):
                _joints2d = util_fn.convert_kps_to_full_img(j2d, center, scale, cfg.image_size)
                joints2d[fi] = _joints2d
            
            results = dict(
                pred_joints2d=joints2d,
                pred_conf=conf,
                center=np.stack(centers),
                scale=np.stack(scales),
                imgpth=image_pth_list[trg_frame]
            )
            os.makedirs(os.path.dirname(save_pth), exist_ok=True)
            joblib.dump(results, save_pth)

            cmap = plt.get_cmap("tab10")
            color = [int(c * 255) for c in cmap(1)]
            writer = imageio.get_writer(vis_pth, fps=20, format='FFMPEG', mode='I', macro_block_size=None)  
            for joint2d, xyxy in tqdm(zip(joints2d, xyxys), total=len(joints2d)):
                outimg = cvimg.copy()
                radius = 2
                for xy in joint2d:
                    x = int(xy[0])
                    y = int(xy[1])
                    cv2.circle(outimg, (x, y), radius=radius, color=color, thickness=-1)
                
                pt1 = (int(xyxy[0]), int(xyxy[1]))
                pt2 = (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(outimg, pt1, pt2, color=(0, 0, 255), thickness=3)
                writer.append_data(outimg)
                

def run_fitting():
    from ..fitting.tools.registry import load_ATLAS
    from llib.viz.renderer import Renderer
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))
    run_cdd()