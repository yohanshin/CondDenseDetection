import os
import sys
sys.path.append("./")
import os.path as osp
import glob

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

from runs.tools import util_fn
from runs.tools.registry import load_checkpoint, load_model
from llib.viz import vis_fn
os.environ["OMP_NUM_THREADS"] = "8"

SA1B_ANNOTATION_NAMES = [
# 24 SA1B joints (in the order provided by OpenPose)
'Nose',
'Neck',
'RShoulder',
'RElbow',
'RWrist',
'LShoulder',
'LElbow',
'LWrist',
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
# 23 COCO+Feet joints (in the order provided by OpenPose)
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
sa1b_to_coco = [SA1B_ANNOTATION_NAMES.index(name) for name in COCO_NAMES]

THREEPO_RESULTS_PTH = "/large_experiments/3po/threepo_airstore/0528_threepod_pseudo/threepod_train.pkl"
SA1B_ANNOT_DIR = "/checkpoint/soyongshin/data/SA1B/2025-06-02"
SA1B_ANNOT_PTH = "/checkpoint/soyongshin/data/SA1B/2025-06-02.pkl"
RESULTS_DIR = "/checkpoint/soyongshin/results/CondDenseDetection/threepod"

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
    os.makedirs(os.path.join(results_dir, "vis"), exist_ok=True)

    # Load annotations
    SA1B_ANNOT_PTH = sorted(glob.glob(SA1B_ANNOT_DIR + "/*.pkl"))[cfg.inference_mod]
    sa1b_annots_list = joblib.load(SA1B_ANNOT_PTH)
    key_list = sa1b_annots_list['img_name']
    metadata_list = sa1b_annots_list['metadata']
    annotation_list = sa1b_annots_list['annotations']

    for key, metadata, annotations in tqdm(
            zip(key_list, metadata_list, annotation_list), 
            total=len(key_list),
            leave=False, 
            dynamic_ncols=True
        ):

        image_pth = metadata['filepath']
        basename = os.path.basename(image_pth)
        
        if 'gaia' in basename:
            dump_pth = os.path.join(results_dir, 'gaia', image_pth.split('/')[-3], basename.replace(".jpg", ".data.pyd"))
            vis_pth = os.path.join(results_dir, 'vis', 'gaia', image_pth.split('/')[-3], basename)
        elif basename.startswith('sa_'):
            dump_pth = os.path.join(results_dir, 'SA1B', basename.replace(".jpg", ".data.pyd"))
            vis_pth = os.path.join(results_dir, 'vis', 'SA1B', basename)
        elif 'egoexo' in image_pth.split('/')[-2]:
            dump_pth = os.path.join(results_dir, 'egoexo4d', basename.replace(".jpg", ".data.pyd"))
            vis_pth = os.path.join(results_dir, 'vis', 'egoexo4d', basename)
        else:
            dump_pth = os.path.join(results_dir, basename.replace(".jpg", ".data.pyd"))
            vis_pth = os.path.join(results_dir, 'vis', basename)
        os.makedirs(os.path.dirname(dump_pth), exist_ok=True)
        os.makedirs(os.path.dirname(vis_pth), exist_ok=True)
        
        if os.path.exists(dump_pth):
            continue
        
        n_people = len(annotations)
        image = cv2.imread(image_pth)
        cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        images = []
        conds = []
        cond_masks = []
        centers, scales = [], []
        cond_points = []
        for pid in range(n_people):
            annotation = annotations[pid]
            
            if annotation.get('rejection_reason', None) is not None:
                continue
            if annotation.get('person_rejection_reason', None) is not None:
                continue
            if isinstance(annotation['metadata'], dict):
                if annotation['metadata'].get('rejection_reason', None) is not None:
                    continue
                if annotation['metadata'].get('person_rejection_reason', None) is not None:
                    continue
                
            if len(annotation['keypoints']) == 0:
                continue

            keypoints = np.array(annotation['keypoints'])
            if keypoints.shape[0] == 17:
                cond = np.zeros((23, 2))    
                cond[:17] = keypoints
                cond_mask = np.zeros(len(cond)).astype(bool)
                cond_mask[-6:] = 1
            else:
                if keypoints.shape[0] != 24:
                    continue
                assert keypoints.shape[0] == 24
                cond = keypoints[sa1b_to_coco][:, :2]
                cond_mask = np.zeros(len(cond)).astype(bool)
            
            cond_points.append(np.concatenate((cond, 1 - cond_mask.astype(float)[:, None]), axis=-1))
            center, scale = util_fn.xyxy2cs(util_fn.kp2xyxy(keypoints[None], scale=1.3)[0][0], aspect_ratio=aspect_ratio, pixel_std=pixel_std, scale_factor=1.0)
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

        if len(images) == 0:
            continue
        
        with torch.no_grad():
            if "xCond" in cfg.model.name:
                pred = model._forward(torch.cat(images), cond=None, cond_mask=None)
            else:
                pred = model._forward(torch.cat(images), cond=torch.cat(conds), cond_mask=torch.cat(cond_masks))

        pred_joints2d = pred['joints2d'].cpu().numpy()
        pred_conf = pred['conf'].cpu().numpy()
        _pred_joints2d = []

        cmap = plt.get_cmap("tab10")
        n_available_people = len(images)
        for p_id in range(n_available_people):

            # Visualize condition
            cond = cond_points[p_id]
            cvimg = vis_fn.imshow_keypoints(cvimg, cond)

            center = centers[p_id]
            scale = scales[p_id]
            joints2d = pred_joints2d[p_id]
            if joints2d.shape[-1] == 2:
                joints2d = np.concatenate((joints2d, np.ones_like(joints2d[..., :1])), axis=-1)
            joints2d = util_fn.convert_kps_to_full_img(joints2d, center, scale, cfg.image_size)
            _pred_joints2d.append(joints2d)
            
            color = [int(c * 255) for c in cmap(p_id + 1)]
            radius = max(int(1.2 * scale.max()), 1)
            for xy in joints2d:
                x = int(xy[0])
                y = int(xy[1])
                cv2.circle(cvimg, (x, y), radius=radius, color=color, thickness=-1)

            vis_fn.imshow_bbox(cvimg, center, scale)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)

        results = {
            "predictions": [dict(
                pred_joints2d=_pred_joints2d[i],
                pred_conf=pred_conf[i],
                center=centers[i],
                scale=scales[i]

                ) for i in range(n_available_people)], 
            "key": key,
            "imagepth": image_pth
        }
        joblib.dump(results, dump_pth)
        cv2.imwrite(vis_pth, cvimg)
    

if __name__ == '__main__':
    main()