import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import imageio
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

DATA_DIR = "/large_experiments/3po/data/legion_test"
RESULTS_DIR = "/checkpoint/soyongshin/results/CondDenseDetection/legion_test"

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

    # Get detection model
    bbox_model = YOLO(_C.PATHS.YOLOV8_CKPT_PTH)
    
    # Get videos
    video_name_list = sorted(os.listdir(DATA_DIR))

    for video_name in video_name_list:
        video_pth = os.path.join(DATA_DIR, video_name)
        vidcap = cv2.VideoCapture(video_pth)

        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)


        out_video_pth = os.path.join(RESULTS_DIR, cfg.exp_name, video_name)
        os.makedirs(os.path.dirname(out_video_pth), exist_ok=True)
        writer = imageio.get_writer(out_video_pth, fps=fps, mode="I", format="FFMPEG", macro_block_size=None)
        pbar = tqdm(range(frame_count), dynamic_ncols=True, leave=False)
        
        while True:
            ret, frame = vidcap.read()
            if not ret: break
            cvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            bbox = bbox_model.predict(frame, 
                                      device=device, 
                                      classes=0, 
                                      conf=0.5, 
                                      save=False, 
                                      verbose=False
            )[0].boxes.xyxy.detach().cpu().numpy()

            if len(bbox)  == 0:
                writer.append_data(cvimg)
                pbar.update(1)
                continue
            bbox = bbox[0]

            center, scale = util_fn.xyxy2cs(bbox, aspect_ratio=aspect_ratio, pixel_std=pixel_std, scale_factor=scale_factor)
            scale = scale.max()
            img_tensor = util_fn.process_cvimg(cvimg.copy(), center, scale, cfg.image_size)
            img_tensor = torch.from_numpy(img_tensor).float().to(device)

            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0))

            pred_joints2d = pred['joints2d'].cpu().squeeze(0).numpy()
            if pred_joints2d.shape[-1] == 2:
                pred_joints2d = np.concatenate((pred_joints2d, np.ones_like(pred_joints2d[..., :1])), axis=-1)

            pred_joints2d = util_fn.convert_kps_to_full_img(pred_joints2d, center, scale, cfg.image_size)

            # Visualize bbox
            bbox = bbox.astype(int)
            cv2.rectangle(cvimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=3)

            # Visualize prediction
            for xy in pred_joints2d:
                x = int(xy[0])
                y = int(xy[1])
                cv2.circle(cvimg, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
            
            writer.append_data(cvimg)
            pbar.update(1)
        writer.close()


if __name__ == '__main__':
    main()