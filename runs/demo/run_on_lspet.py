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
from ultralytics import YOLO
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from configs import constants as _C
from runs.tools import util_fn
from runs.tools.registry import load_checkpoint, load_model

DATA_DIR = "/large_experiments/3po/data/images/hr-lspet"
VITPOSE_DIR = "/checkpoint/soyongshin/results/keypoints_detection/lspe"
RESULTS_DIR = "/checkpoint/soyongshin/results/CondDenseDetection/lspet"

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

    # Inference preparation
    annots = scipy.io.loadmat(os.path.join(DATA_DIR, "joints.mat"))
    joints_list = annots['joints'].transpose(2, 0, 1)
    image_pth_list = sorted(glob(os.path.join(DATA_DIR, "*.png")))
    results_dir = os.path.join(RESULTS_DIR, cfg.exp_name)
    os.makedirs(results_dir, exist_ok=True)

    # Target
    target_idxs = ["im08783"]

    for joint, image_pth in (pbar := tqdm(zip(joints_list, image_pth_list), total=len(joints_list), dynamic_ncols=True)):
        if not image_pth.split("/")[-1].split(".")[0] in target_idxs:
            continue
        
        cvimg = cv2.imread(image_pth)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        img_size = cvimg.shape[:2]
        xyxy, valid = util_fn.kp2xyxy(joint[None], img_size, aspect_ratio=aspect_ratio)

        if isinstance(valid, bool):
            continue

        base_name = os.path.basename(image_pth).replace(".png", ".data.pyd")
        vitpose_pth = os.path.join(VITPOSE_DIR, base_name)
        vitpose = joblib.load(vitpose_pth)[0]['joints2d']
        vitpose = vitpose[:23]
        cond_mask = vitpose[..., -1] < 0.3
        cond = vitpose[:, :2]

        # Prepare input
        center, scale = util_fn.xyxy2cs(xyxy[0], aspect_ratio, pixel_std=200.0, scale_factor=1.0)
        scale = scale.max()
        img_tensor = util_fn.process_cvimg(cvimg.copy(), center, scale, cfg.image_size)
        img_tensor = torch.from_numpy(img_tensor).float().to(device)
        cond = util_fn.process_cond(cond, center, scale, cfg.image_size)
        
        cond_mask = torch.from_numpy(cond_mask).unsqueeze(0).bool().to(device)
        cond = torch.from_numpy(cond).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            if "xCond" in cfg.model.name:
                pred = model(img_tensor.unsqueeze(0), cond=None, cond_mask=None)
            else:
                pred = model(img_tensor.unsqueeze(0), cond=cond, cond_mask=cond_mask)

        pred_joints2d = pred['joints2d'].cpu().squeeze(0).numpy()
        if pred_joints2d.shape[-1] == 2:
            pred_joints2d = np.concatenate((pred_joints2d, np.ones_like(pred_joints2d[..., :1])), axis=-1)
            
        pred_joints2d = util_fn.convert_kps_to_full_img(pred_joints2d, center, scale, cfg.image_size)
        
        # Visualize prediction
        for xy in pred_joints2d:
            x = int(xy[0])
            y = int(xy[1])
            cv2.circle(cvimg, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(results_dir, os.path.basename(image_pth)), cvimg)
        pbar.update(1)

if __name__ == '__main__':
    main()