import os
import sys
import pickle
from glob import glob
from copy import deepcopy

import cv2
import torch
import joblib
import torchvision
import numpy as np
from tqdm import tqdm
from smplx import SMPL

from lib.renderer.renderer import Renderer
from utils import rotation as r


BODY_MODEL = 'smpl'
NUM_BETAS = 10
BODY_MODEL_PATH = f'/large_experiments/3po/model/smpl'
BASEDIR = '/large_experiments/3po/data/images/3dpw'

body_models = {
    gender: SMPL(BODY_MODEL_PATH, gender=gender, num_betas=10).eval() for gender in ['male', 'female']
}


def prepare_annot(annot, p_id, frame_id, map_to_cam_coord=False):
    """ There are two ways to convert world-coordinate SMPL annotation to camera coordinate.

    First, this will end-up with 
    
    """

    gender = {'m': 'male', 'f': 'female'}[annot['genders'][p_id]]
    poses_world = torch.from_numpy(annot['poses'][p_id][[frame_id]]).float()
    trans_world = torch.from_numpy(annot['trans'][p_id][[frame_id]]).float()
    betas = torch.from_numpy(annot['betas'][p_id][:10]).float().view(1, -1)

    intrinsic = torch.from_numpy(annot['cam_intrinsics']).float()
    extrinsic = torch.from_numpy(annot['cam_poses'][[frame_id]]).float()
    campose_valid = annot['campose_valid'][p_id][frame_id].astype(bool)

    if not campose_valid:
        return None

    smpl_params_world = dict(
        global_orient=poses_world[:, :3],
        body_pose=poses_world[:, 3:],
        betas=betas,
        transl=trans_world
    )
    
    if map_to_cam_coord:
        pelvis_6dof_world = torch.eye(4).unsqueeze(0).float()
        pelvis_6dof_world[:, :3, :3] = r.axis_angle_to_matrix(poses_world[:, :3])
        pelvis_6dof_world[:, :3, -1] = trans_world

        # Map world coordinate to camera coordinate
        pelvis_6dof_cam = extrinsic @ pelvis_6dof_world
        poses_cam = poses_world.clone()
        poses_cam[:, :3] = r.matrix_to_axis_angle(pelvis_6dof_cam[:, :3, :3])
        trans_cam_0 = pelvis_6dof_cam[:, :3, -1].clone() # This translation needs to be updated

        smpl_params_cam_0 = dict(
            global_orient=poses_cam[:, :3],
            body_pose=poses_cam[:, 3:],
            betas=betas,
            transl=trans_cam_0
        )

        # Adjust camera coordinate translation
        with torch.no_grad():
            smpl_output_world = body_models[gender](**smpl_params_world)
            smpl_output_cam_0 = body_models[gender](**smpl_params_cam_0)

        verts_cam_0 = smpl_output_cam_0.vertices
        verts_world = smpl_output_world.vertices
        verts_world_hom = torch.cat((verts_world, torch.ones_like(verts_world[..., :1])), dim=-1)
        verts_cam = (extrinsic @ verts_world_hom.mT).mT[..., :3]
        
        trans_cam_offset = verts_cam - verts_cam_0
        print(f'STD of transl offset: {trans_cam_offset.std(dim=1).mean():.3f}   ==> This needs to be very small')
        trans_cam_offset = trans_cam_offset.mean(1)
        trans_cam = trans_cam_0 + trans_cam_offset

        smpl_params_cam = dict(
            global_orient=poses_cam[:, :3],
            body_pose=poses_cam[:, 3:],
            betas=betas,
            transl=trans_cam
        )

        camera_params_cam = dict(
            intrinsic=intrinsic.clone(),
            extrinsic=torch.eye(4).unsqueeze(0)
        )

        return smpl_params_cam, camera_params_cam, gender

    else:
        camera_params_world = dict(
            intrinsic=intrinsic.clone(),
            extrinsic=extrinsic.clone()
        )

        return smpl_params_world, camera_params_world, gender
    

def render(smpl_params, camera_params, gender, background, save_pth, alpha=1.0):
    with torch.no_grad():
        output = body_models[gender](**smpl_params)
        verts = output.vertices.to('cuda').squeeze(0)
    
    H, W = background.shape[:2]
    E = camera_params['extrinsic'].clone()
    renderer = Renderer(W, H, focal_length=None, K=camera_params['intrinsic'], device='cuda', faces=body_models[gender].faces)
    renderer.cameras = renderer.create_camera(R=E[..., :3, :3], T=E[..., :3, -1:])
    renderer.create_lights()
    out_img = renderer.render_mesh(verts, background.copy(), alpha=alpha)
    cv2.imwrite(save_pth, out_img)

    return out_img

if __name__ == '__main__':
    p_id = 0
    frame_id = 755 # 137,  199, #360, #694, #755

    dset = 'train'
    annot_fldr = os.path.join(BASEDIR, 'sequenceFiles', dset)
    annot_file_name = 'courtyard_arguing_00.pkl'
    sequence = annot_file_name.replace('.pkl', '')
    output_fldr = 'outputs/prepare_gt/visualization/3dpw_atlas'
    output_pth = f'3dpw_imageFiles_{sequence}_image_{frame_id:05d}.jpg'

    import pdb; pdb.set_trace()

    annot_file_pth = os.path.join(annot_fldr, annot_file_name)
    annot = pickle.load(open(annot_file_pth, 'rb'), encoding='latin1')
    num_people = len(annot['poses'])
    num_frames = len(annot['img_frame_ids'])
    assert (annot['poses2d'][0].shape[0] == num_frames)

    image_dir = os.path.join(BASEDIR, 'imageFiles', sequence)
    image_path = sorted(glob(os.path.join(image_dir, '*.jpg')))[frame_id]
    
    org_img = cv2.imread(image_path)
    
    img_world, img_cam = org_img.copy(), org_img.copy()
    for p_id in range(num_people):
        smpl_params_world, camera_params_world, gender = prepare_annot(annot, p_id, frame_id, map_to_cam_coord=False)
        smpl_params_cam, camera_params_cam, gender = prepare_annot(annot, p_id, frame_id, map_to_cam_coord=True)
        
        img_world = render(smpl_params_world, camera_params_world, gender, img_world.copy(), f'{output_fldr}/world_{output_pth}', alpha=1.0)
        img_cam = render(smpl_params_cam, camera_params_cam, gender, img_cam.copy(), f'cam_{output_pth}', alpha=1.0)