import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import imageio
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

SAPIENS_KEYPOINTS_308_NAMES = [
    # COCO 17 keypoints
    "Nose", "LEye", "REye", "LEar", "REar", # 5 faces
    "LShoulder", "RShoulder", "LElbow", "RElbow", # 4 Upper body
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",   # 6 Lower body
    
    # Feet
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",   # 6 Feet

    # Hands
    "RThumb_3", "RThumb_2", "RThumb_1", "RThumb_0", 
    "RIndex_3", "RIndex_2", "RIndex_1", "RIndex_0", 
    "RMiddle_3", "RMiddle_2", "RMiddle_1", "RMiddle_0",
    "RRing_3", "RRing_2", "RRing_1", "RRing_0",
    "RPinky_3", "RPinky_2", "RPinky_1", "RPinky_0",
    "RWrist", 
    
    "LThumb_3", "LThumb_2", "LThumb_1", "LThumb_0", 
    "LIndex_3", "LIndex_2", "LIndex_1", "LIndex_0", 
    "LMiddle_3", "LMiddle_2", "LMiddle_1", "LMiddle_0",
    "LRing_3", "LRing_2", "LRing_1", "LRing_0",
    "LPinky_3", "LPinky_2", "LPinky_1", "LPinky_0",
    "LWrist"
]

COCO_kEYPOINTS_23_NAMES = [
    # COCO 17 keypoints
    "Nose", "LEye", "REye", "LEar", "REar", # 5 faces
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", # 4 Upper body
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",   # 6 Lower body
    
    # Feet
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",   # 6 Feet
]

joint_order = [SAPIENS_KEYPOINTS_308_NAMES.index(name) for name in COCO_kEYPOINTS_23_NAMES]

DATA_DIR = "/large_experiments/3po/data/mesh_fitting/egoexo4d"
FITTED_RESULTS_DIR = "/private/home/taoshaf/data/annotation/egoexo4d/annos"
RESULTS_DIR = "/checkpoint/soyongshin/results/CondDenseDetection/egoexo4d"

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

    trg_seq = ["indiana_bike_10_12"]
    all_seq_list = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    for seq in all_seq_list:
        if not seq in trg_seq: continue
        
        cameras = sorted(os.listdir(os.path.join(FITTED_RESULTS_DIR, seq)))
        for camera in cameras:
            annot_pth_list = sorted(os.listdir(os.path.join(FITTED_RESULTS_DIR, seq, camera)))
            
            vis_pth = os.path.join(RESULTS_DIR, seq, f"{camera}.mp4")
            os.makedirs(os.path.dirname(vis_pth), exist_ok=True)
            
            bs = 64
            full_pred_joints2d = []
            full_pred_conf = []
            writer = imageio.get_writer(vis_pth, fps=30, mode="I", format="FFMPEG", macro_block_size=None)
            for start in tqdm(range(0, len(annot_pth_list), bs), desc="Processing ...", leave=False):
                
                end = min(len(annot_pth_list), start + bs)

                img_tensors, centers, scales, conds, cond_masks, images, xyxys, cond_joints = [], [], [], [], [], [], [], []
                for annot_pth in tqdm(annot_pth_list[start:end], leave=False):
                    image_pth = os.path.join(DATA_DIR, seq, 'images', camera, annot_pth.replace('.pkl', '.jpg'))
                    annot = joblib.load(os.path.join(FITTED_RESULTS_DIR, seq, camera, annot_pth))[0]
                    gt_joints2d = annot['keypoints_2d']
                    center = annot["center"]
                    scale = annot["scale"] / pixel_std
                    # scale = np.array([scale[1] * aspect_ratio, scale[1]])
                    scale_0 = max(scale[1] * aspect_ratio, scale[0])
                    scale_1 = max(scale[0] / aspect_ratio, scale[1])
                    scale = np.array([scale_0, scale_1])
                    xyxy = annot['bbox']
                    
                    # Process image
                    image = cv2.imread(image_pth)
                    cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img_tensor = util_fn.process_cvimg(cvimg, center, scale, cfg.image_size, pixel_std=pixel_std)
                    img_tensors.append(torch.from_numpy(img_tensor).float())

                    # Process condition
                    cond = gt_joints2d[joint_order, :2]
                    cond = util_fn.process_cond(cond, center, scale, cfg.image_size)
                    cond_mask = gt_joints2d[joint_order, -1] < 0.3
                    cond_masks.append(torch.from_numpy(cond_mask).bool())
                    conds.append(torch.from_numpy(cond).float())

                    # Auxilary
                    centers.append(center)
                    scales.append(scale)
                    images.append(cvimg)
                    xyxys.append(xyxy)
                    cond_joints.append(gt_joints2d[joint_order, :2])

                img_tensors = torch.stack(img_tensors)
                conds = torch.stack(conds)
                cond_masks = torch.stack(cond_masks)
                
                with torch.no_grad():
                    if "xCond" in cfg.model.name:
                        pred = model._forward(img_tensors.to(device), cond=None, cond_mask=None)
                    else:
                        pred = model._forward(img_tensors.to(device), 
                                              cond=conds.to(device), 
                                              cond_mask=cond_masks.to(device))
                
                full_joints2d = pred["joints2d"].cpu().numpy()
                conf = pred["conf"].cpu().numpy()
                full_joints2d = np.concatenate((full_joints2d, np.ones_like(full_joints2d[..., :1])), axis=-1)
                for fi, (joints2d, center, scale) in enumerate(zip(full_joints2d, centers, scales)):
                    _joints2d = util_fn.convert_kps_to_full_img(joints2d, center, scale, cfg.image_size)
                    full_joints2d[fi] = _joints2d
            
                cmap = plt.get_cmap("tab10")
                color1 = [int(c * 255) for c in cmap(1)]
                color2 = [int(c * 255) for c in cmap(2)]
                for joints2d, cvimg, center, scale, xyxy, cond_joint in tqdm(
                    zip(full_joints2d, images, centers, scales, xyxys, cond_joints), 
                    total=len(full_joints2d), desc="Drawing ...", leave=False):
                    for x, y in joints2d[:, :2].astype(int):
                        cvimg = cv2.circle(cvimg, (x, y), radius=3, color=color1, thickness=-1)
                    
                    for x, y in cond_joint.astype(int):
                        cvimg = cv2.circle(cvimg, (x, y), radius=5, color=color2, thickness=-1)

                    pt1 = (int(xyxy[0]), int(xyxy[1]))
                    pt2 = (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(cvimg, pt1, pt2, color=(0, 0, 255), thickness=3)
                    
                    cvimg = util_fn.process_cvimg(cvimg, center, scale, (384, 512), transform_only=True)
                    writer.append_data(cvimg)
            writer.close()


if __name__ == '__main__':
    main()