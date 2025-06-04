import os
import sys
sys.path.append("./")
import glob
import pickle
import argparse

import cv2
import torch
import joblib
import numpy as np
from smplx import SMPL
from pytorch3d import transforms as r

from configs import constants as _C

def is_dset(emdb_pkl_file, dset):
    target_dset = 'emdb' + dset
    with open(emdb_pkl_file, "rb") as f:
        data = pickle.load(f)
        return data[target_dset]

def preprocess(dset):
    skip = 5    # process to 6FPS 

    tt = lambda x: torch.from_numpy(x).float()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_pth = os.path.join(_C.PATHS.PARSED_DATA_DIR, f'emdb{dset}_6fps.pth')
    
    all_emdb_pkl_files = sorted(glob.glob(os.path.join(_C.PATHS.EMDB_DIR, "*/*/*_data.pkl")))
    emdb_sequence_roots = []
    both = []
    for emdb_pkl_file in all_emdb_pkl_files:
        if is_dset(emdb_pkl_file, dset):
            emdb_sequence_roots.append(os.path.dirname(emdb_pkl_file))
    
    smpl = {
        'neutral': SMPL(model_path=_C.PATHS.SMPL_MODEL_DIR),
        'male': SMPL(model_path=_C.PATHS.SMPL_MODEL_DIR, gender='male'),
        'female': SMPL(model_path=_C.PATHS.SMPL_MODEL_DIR, gender='female'),
    }

    dataset = dict(
        bboxes=[],
        imgpaths=[],
        verts2d=[],
    )
    for sequence in emdb_sequence_roots:
        subj, seq = sequence.split('/')[-2:]
        
        annot_pth = glob.glob(os.path.join(sequence, '*_data.pkl'))[0]
        annot = pickle.load(open(annot_pth, 'rb'))
        
        # Get ground truth data
        gender = annot['gender']
        masks = annot['good_frames_mask']
        poses_body = annot["smpl"]["poses_body"]
        poses_root = annot["smpl"]["poses_root"]
        trans = annot["smpl"]["trans"]
        betas = np.repeat(annot["smpl"]["betas"].reshape((1, -1)), repeats=annot["n_frames"], axis=0)
        extrinsics = annot["camera"]["extrinsics"]
        intrinsics = annot["camera"]["intrinsics"]
        width, height = annot['camera']['width'], annot['camera']['height']
        xyxys = annot['bboxes']['bboxes']
        imname_list = sorted(glob.glob(os.path.join(sequence, 'images/*')))
        imname_list = imname_list[::skip]
        
        # Build 3D human mesh
        with torch.no_grad():
            gt_body = smpl[gender](
                global_orient=torch.from_numpy(poses_root[::skip]).float(),
                body_pose=torch.from_numpy(poses_body[::skip]).float(),
                betas=torch.from_numpy(betas[::skip]).float(),
                transl=torch.from_numpy(trans[::skip]).float(),
            )
            verts_world = gt_body.vertices

        verts_world_hom = torch.cat((verts_world, torch.ones_like(verts_world[..., :1])), dim=-1).numpy()
        verts_cam = extrinsics[::skip] @ verts_world_hom.transpose(0, 2, 1)
        verts_cam_25d = np.divide(verts_cam[:, :3], verts_cam[:, 2:3])
        verts_img = intrinsics[None] @ verts_cam_25d
        verts_img = verts_img[:, :2].transpose(0, 2, 1)
        
        # Check visualization
        if False:
            img = cv2.imread(imname_list[100])
            for xy in verts_img[100]:
                img = cv2.circle(img, (int(xy[0]), int(xy[1])), color=(0, 255, 0), radius=2, thickness=-1)
            cv2.imwrite('test.png', img)

        # imname_list = sorted(['/'.join(imname.split('/')[-4:]) for imname in imname_list])
        dataset["bboxes"].append(xyxys[::skip].astype(np.float32))
        dataset["imgpaths"].append(np.array(imname_list))
        dataset["verts2d"].append(verts_img.astype(np.float32))

    for k, v in dataset.items():
        dataset[k] = np.concatenate(v)

    joblib.dump(dataset, save_pth)

        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="1", choices=['1', '2'], help='Data split')
    args = parser.parse_args()
    
    preprocess(args.split)