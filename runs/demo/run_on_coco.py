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
from omegaconf import OmegaConf, DictConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from configs import constants as _C
from runs.tools import util_fn
from runs.tools.registry import load_checkpoint, load_model

DATA_DIR = "/large_experiments/3po/data/images/coco-train-2014-pruned/train2014"
RESULTS_DIR = "/checkpoint/soyongshin/results/CondDenseDetection/coco-train-2014"

OPENPOSE_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'Nose',
'Neck',
'RShoulder',
'RElbow',
'RWrist',
'LShoulder',
'LElbow',
'LWrist',
'MidHip',
'RHip',
'RKnee',
'RAnkle',
'LHip',
'LKnee',
'LAnkle',
'REye',
'LEye',
'REar',
'LEar',
'LBigToe',
'LSmallToe',
'LHeel',
'RBigToe',
'RSmallToe',
'RHeel',
]

COCO_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'Nose',
'LEye',
'REye',
'LEar',
'REar',
'LShoulder',
'RShoulder',
'LElbow',
'RElbow',
'LWrist',
'RWrist',
'LHip',
'RHip',
'LKnee',
'RKnee',
'LAnkle',
'RAnkle',
'LBigToe',
'LSmallToe',
'LHeel',
'RBigToe',
'RSmallToe',
'RHeel',
]
openpose_to_coco = [OPENPOSE_NAMES.index(name) for name in COCO_NAMES]

aspect_ratio = 3/4
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

    results_dir = os.path.join(RESULTS_DIR, cfg.exp_name)
    os.makedirs(results_dir, exist_ok=True)

    image_path_list = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])
    annot_path_list = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.data.pyd')])
    
    for image_path, annot_path in (pbar := tqdm(zip(image_path_list, annot_path_list), 
                                                total=len(image_path_list), dynamic_ncols=True)):
            
        assert image_path.split('/')[-1].split('.')[0] == annot_path.split('/')[-1].split('.')[0]
        
        image = cv2.imread(image_path)
        if image is None: continue
        
        if len(image.shape) == 2:
            image = image[..., None]
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation
        annots = joblib.load(annot_path)
        n_people = len(annots)
        images = []
        conds = []
        cond_masks = []
        centers, scales = [], []
        for p_id in range(n_people):
            annot = annots[p_id]
            joints2d = annot["keypoints_2d"][openpose_to_coco]
            cond = joints2d[:, :2]
            cond_mask = joints2d[:, -1] < 0.3
            
            center = annot['center'].copy()
            scale = annot['scale'].copy().max()

            # Image
            img_tensor = util_fn.process_cvimg(cvimg.copy(), center, scale, cfg.image_size)
            img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)

            # Condition
            cond = util_fn.process_cond(cond, center, scale, cfg.image_size)
            cond_mask = torch.from_numpy(cond_mask).unsqueeze(0).bool().to(device)
            cond = torch.from_numpy(cond).unsqueeze(0).float().to(device)

            images.append(img_tensor)
            conds.append(cond)
            cond_masks.append(cond_mask)
            centers.append(center)
            scales.append(scale)

        with torch.no_grad():
            if "diffusion" in cfg.model.name:
                pred = model._forward(torch.cat(images), cond=torch.cat(conds), cond_mask=torch.cat(cond_masks))
            elif "xCond" in cfg.model.name:
                pred = model._forward(torch.cat(images), cond=None, cond_mask=None)
            else:
                pred = model._forward(torch.cat(images), cond=torch.cat(conds), cond_mask=torch.cat(cond_masks))
        
        pred_joints2d = pred['joints2d'].cpu().numpy()
        pred_conf = pred['conf'].cpu().numpy()
        _pred_joints2d = []
        
        cmap = plt.get_cmap("tab10")
        for p_id in range(n_people):
            center = centers[p_id]
            scale = scales[p_id]
            joints2d = pred_joints2d[p_id]
            if joints2d.shape[-1] == 2:
                joints2d = np.concatenate((joints2d, np.ones_like(joints2d[..., :1])), axis=-1)
            joints2d = util_fn.convert_kps_to_full_img(joints2d, center, scale, cfg.image_size)
            _pred_joints2d.append(joints2d)
            
            color = [int(c * 255) for c in cmap(p_id)]
            radius = max(int(1.2 * scale), 1)
            for xy in joints2d:
                x = int(xy[0])
                y = int(xy[1])
                cv2.circle(cvimg, (x, y), radius=radius, color=color, thickness=-1)

        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
        
        results = [dict(
            pred_joints2d=_pred_joints2d[i],
            pred_conf=pred_conf[i],
            center=centers[i],
            scale=scales[i]

        ) for i in range(n_people)]
        joblib.dump(results, os.path.join(results_dir, os.path.basename(annot_path)))
        cv2.imwrite(os.path.join(results_dir, os.path.basename(image_path)), cvimg)
        

if __name__ == '__main__':
    main()