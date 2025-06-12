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

RESULTS_DIR = "examples/out"

aspect_ratio = 4/3
pixel_std = 200
scale_factor = 1.0
CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/hydra"), config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    OmegaConf.register_new_resolver("leq", lambda a, b: a >= b)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))
    
    # Load model
    work_dir = osp.join(CUR_PATH, cfg.work_dir, 'train', 'regressor_2d', cfg.exp_name)
    ckpt_dir = os.path.join(work_dir, 'checkpoints')
    model = load_model(cfg, device)
    model = load_checkpoint(model, ckpt_dir)
    model.eval()

    if os.path.isdir(cfg.image_path):
        pass
    elif os.path.exists(cfg.image_path):
        image_path_list = [cfg.image_path]
    else:
        NotImplementedError, "No such file exists !"
    
    results_dir = os.path.join(RESULTS_DIR, cfg.exp_name)
    for image_path in image_path_list:
        cvimg = cv2.imread(image_path)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        
        if cfg.use_detection:
            pass
        else:
            xyxy = np.array([0, 0, cvimg.shape[1], cvimg.shape[0]]).astype(float)
            # xyxy = np.array([-cvimg.shape[1], -cvimg.shape[1], 2 * cvimg.shape[1], 2 * cvimg.shape[0]]).astype(float)

        center, scale = util_fn.xyxy2cs(xyxy, aspect_ratio, pixel_std=200.0, scale_factor=1.0)
        scale = scale.max()
        img_tensor = util_fn.process_cvimg(cvimg.copy(), center, scale, cfg.image_size)
        img_tensor = torch.from_numpy(img_tensor).float().to(device)
        
        with torch.no_grad():
            if "diffusion" in cfg.model.name:
                pred = model._forward(img_tensor.unsqueeze(0), cond=cond, cond_mask=cond_mask)
            elif "xCond" in cfg.model.name:
                pred = model(img_tensor.unsqueeze(0), cond=None, cond_mask=None)
            else:
                pred = model(img_tensor.unsqueeze(0), cond=cond, cond_mask=cond_mask)

        pred_joints2d = pred['joints2d'].cpu().squeeze(0).numpy()
        if pred_joints2d.shape[-1] == 2:
            pred_joints2d = np.concatenate((pred_joints2d, np.ones_like(pred_joints2d[..., :1])), axis=-1)
            
        pred_joints2d = util_fn.convert_kps_to_full_img(pred_joints2d, center, scale, cfg.image_size)
        for xy in pred_joints2d:
            x = int(xy[0])
            y = int(xy[1])
            cv2.circle(cvimg, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

        import pdb; pdb.set_trace()
        cv2.imwrite("test.png", cvimg[..., ::-1])

if __name__ == '__main__':
    main()