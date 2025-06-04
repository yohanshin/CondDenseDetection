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

sys.path.remove("/private/home/xyang35/codes/3po")
sys.path.remove("/private/home/xyang35/codes/3po/promptable_3po")
sys.path.append("/private/home/soyongshin/code/projects/3po")
from lib.renderer.renderer import Renderer
from lib.keypoints_inference.misc.pose_utils import batch_compute_similarity_transform_torch
from utils import rotation as _r
from promptable_3po.models.modules.atlas import ATLAS


BASEDIR = '/large_experiments/3po/data/images/3dpw'

def atlas_forward(atlas, params):
    global_trans = params["global_trans"].reshape(1, -1)
    global_rot = params["global_rot"].reshape(1, -1)
    body_pose_params = params["body_pose_params"].reshape(1, -1)
    hand_pose_params = params["hand_pose_params"].reshape(1, -1)
    scale_params = params["scale_params"].reshape(1, -1)
    shape_params = params["shape_params"].reshape(1, -1)
    expr_params = params["expr_params"].reshape(1, -1)
    
    verts, j3d = atlas(
        global_trans=global_trans, 
        global_rot=global_rot,
        body_pose_params=body_pose_params,
        hand_pose_params=hand_pose_params,
        scale_params=scale_params,
        shape_params=shape_params,
        expr_params=expr_params,
        do_pcblend=True,
        return_keypoints=atlas.load_keypoint_mapping
    )

    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1
    return verts, j3d

def prepare_annot(results, annot, frame_id, map_to_world_coord=False):
    """ There are two ways to convert world-coordinate SMPL annotation to camera coordinate.

    First, this will end-up with 
    
    """

    atlas_params_cam = results['atlas_params']
    intrinsic = torch.from_numpy(annot['cam_intrinsics']).float()
    extrinsic = torch.from_numpy(annot['cam_poses'][frame_id]).float()
    
    if not map_to_world_coord:
        camera_params_cam = dict(
            intrinsic=intrinsic.clone(),
            extrinsic=torch.eye(4).unsqueeze(0)
        )
        return atlas_params_cam, camera_params_cam

    inv_extrinsic = torch.linalg.inv(extrinsic)
    rotation = torch.tensor([[[1, 0, 0], [0, 0, -1], [0, -1, 0]]]).float().to(device)
    global_rot_cam_euler = atlas_params_cam['global_rot']
    global_rot_cam_mat = rotation @ _r.euler_angles_to_matrix(global_rot_cam_euler, "ZYX").mT @ rotation.mT
    global_rot_world_mat = inv_extrinsic[:3, :3].unsqueeze(0).to(device=global_rot_cam_mat.device) @ global_rot_cam_mat
    global_rot_world_euler = _r.matrix_to_euler_angles(rotation @ global_rot_world_mat.mT @ rotation.mT, "ZYX")
    
    atlas_params_world = deepcopy(atlas_params_cam)
    atlas_params_world["global_rot"] = global_rot_world_euler
    atlas_params_world["global_trans"] = torch.zeros_like(atlas_params_world["global_trans"])

    verts_cam, j3d_cam = atlas_forward(atlas, atlas_params_cam)
    atlas_params_cam2 = deepcopy(atlas_params_cam)
    random_trans = torch.randn_like(atlas_params_cam['global_trans'])
    atlas_params_cam2['global_trans'] = atlas_params_cam['global_trans'] + random_trans
    verts_cam2, j3d_cam2 = atlas_forward(atlas, atlas_params_cam2)
    import pdb; pdb.set_trace()
    verts_world_wo_trans, j3d_world_wo_trans = atlas_forward(atlas, atlas_params_world)
    verts_world = (inv_extrinsic.unsqueeze(0).cuda() @ torch.cat((verts_cam, torch.ones_like(verts_cam[..., :1])), dim=-1).mT).mT[..., :3]
    trans_offset = verts_world - verts_world_wo_trans
    trans_offset = trans_offset.reshape(-1, 3)
    print(f"Trans offset STD: {trans_offset.std(0)}")
    trans_offset = trans_offset.mean(0)
    atlas_params_world["global_trans"] = trans_offset.view_as(atlas_params_world["global_trans"])
    
    camera_params_world = dict(
        intrinsic=intrinsic.clone(),
        extrinsic=extrinsic.unsqueeze(0)
    )
    return atlas_params_world, camera_params_world


def render(atlas, atlas_params, camera_params, background, save_pth, alpha=1.0):
    with torch.no_grad():
        verts, j3d = atlas_forward(atlas, atlas_params)
    
    H, W = background.shape[:2]
    E = camera_params['extrinsic'].clone()
    renderer = Renderer(W, H, focal_length=None, K=camera_params['intrinsic'], device='cpu', faces=atlas.faces.detach().cpu().numpy())
    renderer.cameras = renderer.create_camera(R=E[..., :3, :3], T=E[..., :3, -1:])
    renderer.create_lights()
    out_img = renderer.render_mesh(verts.cpu().reshape(-1, 3), background.copy(), alpha=alpha)
    cv2.imwrite(save_pth, out_img)

    return out_img

if __name__ == '__main__':
    p_id = 0
    frame_id = 360 # 137,  199, #360, #694, #755
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dset = 'train'
    annot_fldr = os.path.join(BASEDIR, 'sequenceFiles', dset)
    annot_file_name = 'courtyard_arguing_00.pkl'
    sequence = annot_file_name.replace('.pkl', '')
    output_fldr = 'outputs/prepare_gt/visualization/3dpw_atlas'
    output_pth = f'3dpw_imageFiles_{sequence}_image_{frame_id:05d}.jpg'
    annot_file_pth = os.path.join(annot_fldr, annot_file_name)
    annot = pickle.load(open(annot_file_pth, 'rb'), encoding='latin1')

    fitting_results_pth = f"/large_experiments/3po/data/atlas_refine_v2/ThreeDPW/{sequence}_P{p_id}.data.pyd"
    fitting_results = joblib.load(fitting_results_pth)[::2][frame_id]
    
    image_dir = os.path.join(BASEDIR, 'imageFiles', sequence)
    image_path = sorted(glob(os.path.join(image_dir, '*.jpg')))[frame_id]
    
    # Build ATLAS model
    # Load in ATLAS
    # Initialize ATLAS model
    num_shape_comps = 16
    num_scale_comps = 16
    num_hand_comps = 32 # PER HAND!
    num_expr_comps = 10
    lod = "lod3" # "smplx"
    asset_dir = "/large_experiments/3po/model/atlas/"
    atlas = ATLAS(asset_dir, 
                  num_shape_comps, 
                  num_scale_comps, 
                  num_hand_comps, 
                  num_expr_comps, 
                  lod=lod, 
                  load_keypoint_mapping=lod=="lod3",
                  verbose=True
    ).to(device)
    # atlas = None

    # Check the global rotation representation
    default_params = {k: torch.zeros_like(v) for k, v in fitting_results['atlas_params'].items()}
    rotated_params = {k: torch.zeros_like(v) for k, v in fitting_results['atlas_params'].items()}
    rotated_params['global_rot'] = torch.tensor([[-0.18, 1.23, -0.71]]).float().to(device)

    with torch.no_grad():
        rotated_verts, rotated_j3d = atlas_forward(atlas, rotated_params)
        default_verts, default_j3d = atlas_forward(atlas, default_params)
    
    default_j3d_hat, R, scale, trans = batch_compute_similarity_transform_torch(default_j3d, rotated_j3d, return_transformation=True)
    # rotation = torch.tensor([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]]).float().to(device)
    rotation = torch.tensor([[[1, 0, 0], [0, 0, -1], [0, -1, 0]]]).float().to(device)
    # for convention in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
    #     print(R @ rotation - _r.euler_angles_to_matrix(rotated_params['global_rot'], convention))
    import pdb; pdb.set_trace()
    for convention in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
        print(_r.euler_angles_to_matrix(rotated_params['global_rot'], convention))
        # print(rotation @ _r.euler_angles_to_matrix(rotated_params['global_rot'], convention).mT @ rotation.mT)
    # for convention in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
    #     print(R.mT - rotation @ _r.euler_angles_to_matrix(rotated_params['global_rot'], convention))
    # for convention in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
    #     print(R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], convention) @ rotation)
    
    # R - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'XYZ')
    # R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'XYZ')
    # R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'XZY')
    # R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'YXZ')
    # R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'YZX')
    # R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'ZXY')
    # R.mT - _r.euler_angles_to_matrix(rotated_params['global_rot'], 'ZYX')
    # R.mT - _r.axis_angle_to_matrix(rotated_params['global_rot'])

    org_img = cv2.imread(image_path)
    
    img_world, img_cam = org_img.copy(), org_img.copy()
    atlas_params_world, camera_params_world = prepare_annot(fitting_results, annot, frame_id, map_to_world_coord=True)
    atlas_params_cam, camera_params_cam = prepare_annot(fitting_results, annot, frame_id, map_to_world_coord=False)
    
    img_world = render(atlas, atlas_params_world, camera_params_world, img_world.copy(), f'{output_fldr}/world_{output_pth}', alpha=1.0)
    import pdb; pdb.set_trace()
    # img_cam = render(atlas, atlas_params_cam, camera_params_cam, img_cam.copy(), os.path.join(output_fldr, f'cam_{output_pth}'), alpha=1.0)