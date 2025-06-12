import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import pickle
import joblib
import imageio
import numpy as np
from tqdm import tqdm
from smplx import SMPL
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from configs import constants as _C
from runs.tools import util_fn
from runs.tools.registry import load_checkpoint, load_model

DATA_DIR = "/large_experiments/3po/data/images/3dpw"
RESULTS_DIR = "/checkpoint/soyongshin/results/CondDenseDetection/threedpw"
INIT_ATLAS_RESULTS_DIR = '/large_experiments/3po/data/atlas_nlf+wilor/threedpw'
SMPL2COCO_REG_PTH = "/large_experiments/3po/model/smpl/J_regressor_coco+feet.npy"
SMPL_MODEL_DIR = "/large_experiments/3po/model/smpl"

aspect_ratio = 3/4
pixel_std = 200
scale_factor = 1.0

def prepare_labels(annot, p_id, body_model):
    J_regressor = torch.from_numpy(np.load(SMPL2COCO_REG_PTH)).float().unsqueeze(0)
    
    # Gender
    gender = {'m': 'male', 'f': 'female'}[annot['genders'][p_id]]
            
    # SMPL parameters
    pose = torch.from_numpy(annot['poses'][p_id]).float()
    betas = torch.from_numpy(annot['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
    transl = torch.from_numpy(annot['trans'][p_id]).float()
    
    # Cam poses
    cam_intrinsics = annot['cam_intrinsics']
    cam_pose = torch.from_numpy(annot['cam_poses']).float()
    campose_valid = annot['campose_valid'][p_id].astype(bool)

    img_size = annot['cam_intrinsics'][:2, -1].astype(int) * 2
    joints2d = annot['poses2d'][p_id].transpose(0, 2, 1)
    
    # Construct SMPL output
    with torch.no_grad():
        output = body_model[gender](
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:],
            betas=betas,
            transl=transl
        )
    
    verts = output.vertices
    joints3d = torch.matmul(J_regressor, verts)
    joints3d_hom = torch.cat((joints3d, torch.ones_like(joints3d[..., :1])), dim=-1)
    joints3d_cam = (cam_pose @ joints3d_hom.mT).mT[..., :3]

    joints2d = (torch.from_numpy(cam_intrinsics).float().unsqueeze(0) @ torch.div(joints3d_cam, joints3d_cam[..., -1:]).mT).mT[..., :2]
    xyxys, bbox_valid = util_fn.kp2xyxy(joints2d.numpy(), img_size[::-1], scale=1.2)
    valid_mask = np.logical_and(campose_valid, bbox_valid)
    
    centers, scales = [], []
    for xyxy in xyxys:
        center, scale = util_fn.xyxy2cs(xyxy, aspect_ratio, pixel_std, scale_factor=1.0)
        centers.append(center)
        scales.append(scale)
    centers = np.stack(centers)
    scales = np.stack(scales)
    

    return dict(
        centers=centers[valid_mask],
        scales=scales[valid_mask],
        verts=verts[valid_mask],
        joints2d=joints2d[valid_mask],
        valid_frames=np.arange(len(pose))[valid_mask],
        cam_intrinsics=cam_intrinsics
    )

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

    body_model = {
        k: SMPL(SMPL_MODEL_DIR, gender=k, num_betas=10)
        for k in ["male", "female"]
    }

    for dset in (dbar := tqdm(['train', 'test', 'validation'], dynamic_ncols=True)):
        dbar.set_description_str(f'3DPW {dset} set')
        annot_fldr = os.path.join(DATA_DIR, 'sequenceFiles', dset)
        annot_file_names = sorted(os.listdir(annot_fldr))
        
        for annot_file_name in (pbar := tqdm(annot_file_names, dynamic_ncols=True)):
            sequence = annot_file_name.replace('.pkl', '')
            
            annot_file_pth = os.path.join(annot_fldr, annot_file_name)
            annot = pickle.load(open(annot_file_pth, 'rb'), encoding='latin1')
            num_people = len(annot['poses'])

            pred_joints2d_multiperson = []
            cond_joints2d_multiperson = []
            frames_multiperson = []
            xyxys_multiperson = []
            for p_id in range(num_people):
                save_pth = os.path.join(results_dir, f'{sequence}_P{p_id}.data.pyd')
                
                batch = prepare_labels(annot, p_id, body_model)
                image_pth_list = sorted(glob(os.path.join(DATA_DIR, 'imageFiles', sequence, "*.jpg")))
                image_pth_list = np.array(image_pth_list)[batch["valid_frames"]]
                
                img_tensors, conds, cond_masks = [], [], []
                for fi, (image_pth, center, scale, joints2d) in tqdm(
                    enumerate(zip(image_pth_list, batch["centers"], batch["scales"], batch["joints2d"])), 
                    total=len(image_pth_list),
                    desc="loading images", leave=False):
                    image = cv2.imread(image_pth)
                    cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img_tensor = util_fn.process_cvimg(cvimg.copy(), center, scale, cfg.image_size, frame=fi)
                    img_tensors.append(torch.from_numpy(img_tensor).float().to(device))

                    cond = joints2d[:, :2]
                    cond_mask = np.zeros(joints2d.shape[0]).astype(bool)
                    cond_mask[-6:] = 1   # Ignore feet 
                    cond = util_fn.process_cond(cond, center, scale, cfg.image_size)
                    cond_masks.append(torch.from_numpy(cond_mask).bool().to(device))
                    conds.append(torch.from_numpy(cond).float().to(device))

                img_tensors = torch.stack(img_tensors)
                conds = torch.stack(conds)
                cond_masks = torch.stack(cond_masks)
                
                full_pred_joints2d = []
                full_pred_conf = []

                for start in tqdm(range(0, img_tensors.shape[0], 64), desc="Inference ...", leave=False):
                    end = min(img_tensors.shape[0], start + 64)
                    with torch.no_grad():
                        if "xCond" in cfg.model.name:
                            pred = model._forward(img_tensors[start:end], cond=None, cond_mask=None)
                        else:
                            pred = model._forward(img_tensors[start:end], cond=conds[start:end], cond_mask=cond_masks[start:end])
                    
                    joints2d = pred["joints2d"].cpu().numpy()
                    conf = pred["conf"].cpu().numpy()

                    full_pred_joints2d.append(joints2d)
                    full_pred_conf.append(conf)
                
                full_pred_joints2d = np.concatenate(full_pred_joints2d)
                full_pred_conf = np.concatenate(full_pred_conf)
                full_pred_joints2d = np.concatenate((full_pred_joints2d, np.ones_like(full_pred_joints2d[..., :1])), axis=-1)
                for fi, (joints2d, center, scale) in enumerate(zip(full_pred_joints2d, batch["centers"], batch["scales"])):
                    _joints2d = util_fn.convert_kps_to_full_img(joints2d, center, scale, cfg.image_size)
                    full_pred_joints2d[fi] = _joints2d
                
                results = dict(
                    pred_joints2d=full_pred_joints2d,
                    pred_conf=full_pred_conf,
                    center=batch["centers"],
                    scale=batch["scales"],
                    frames=batch["valid_frames"]
                )
                joblib.dump(results, save_pth)

                pred_joints2d_multiperson.append(full_pred_joints2d)
                cond_joints2d_multiperson.append(batch["joints2d"])
                frames_multiperson.append(batch["valid_frames"])
                xyxys_multiperson.append(util_fn.cs2xyxy(batch["centers"], batch["scales"]))

            video_pth = os.path.join(results_dir, f'{sequence}.mp4')
            writer = imageio.get_writer(video_pth, fps=30, mode="I", format="FFMPEG", macro_block_size=None)
            cmap = plt.get_cmap("tab10")
            image_pth_list = sorted(glob(os.path.join(DATA_DIR, 'imageFiles', sequence, "*.jpg")))
            for img_i, image_pth in tqdm(enumerate(image_pth_list), total=len(image_pth_list), leave=False):
                cvimg = cv2.cvtColor(cv2.imread(image_pth), cv2.COLOR_BGR2RGB)
                for pi in range(num_people):
                    if img_i in frames_multiperson[pi]:
                        _i = np.where(frames_multiperson[pi] == img_i)[0][0].item()
                        joints2d = pred_joints2d_multiperson[pi][_i]
                        color = [int(c * 255) for c in cmap(pi+1)]
                        radius = 2
                        for xy in joints2d:
                            x = int(xy[0])
                            y = int(xy[1])
                            cv2.circle(cvimg, (x, y), radius=radius, color=color, thickness=-1)

                        # Draw condition input
                        joints2d = cond_joints2d_multiperson[pi][_i]
                        for xy in joints2d:
                            x = int(xy[0])
                            y = int(xy[1])
                            # cv2.circle(cvimg, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
                            cv2.circle(cvimg, (x, y), radius=4, color=color, thickness=-1)

                        # Draw bbox
                        xyxy = xyxys_multiperson[pi][_i]
                        pt1 = (int(xyxy[0]), int(xyxy[1]))
                        pt2 = (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(cvimg, pt1, pt2, color=(0, 0, 255), thickness=3)

                writer.append_data(cvimg)
            writer.close()


if __name__ == "__main__":
    main()