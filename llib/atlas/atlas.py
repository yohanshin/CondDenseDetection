import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn

from linear_blend_skinning_cuda import LinearBlendSkinningCuda

from .atlas_utils import load_pickle, batchXYZfrom6D, get_pose_feats_6d_batched_from_joint_params, SparseLinear


class ATLAS(nn.Module):
    def __init__(self,
                 model_data_dir,
                 num_shape_comps,
                 num_scale_comps,
                 num_hand_comps,
                 num_expr_comps,
                 lod="lod3",
                 load_keypoint_mapping=False,
                 verbose=False):
        super().__init__()

        # Set Class Fields
        self.model_data_dir = model_data_dir
        self.num_shape_comps = num_shape_comps
        self.num_scale_comps = num_scale_comps
        self.num_hand_comps = num_hand_comps
        self.num_expr_comps = num_expr_comps
        self.lod = lod.lower(); assert self.lod in ['smpl', 'smplx', 'lod3', 'lod4', 'lod5']
        if self.lod == 'smplx': print("WARNING! ATLAS-to-SMPLX mapping is suboptimal for face")
        self.load_keypoint_mapping = load_keypoint_mapping
        if self.load_keypoint_mapping: assert self.lod == 'lod3', "308 Keypoints only supported for LOD3"
        self.verbose = verbose

        # Load Model
        if self.verbose: print(f"Loading ATLAS from {model_data_dir} ...")
        model_dict = load_pickle(osp.join(model_data_dir, "params.pkl"))
        if self.verbose: print(f"Done loading pkl!")

        # Load Shape & Scale Bases
        self.shape_mean = nn.Parameter(model_dict['shape_mean'], requires_grad=False) # 115834 x 3
        self.shape_comps = nn.Parameter(model_dict['shape_comps'][:self.num_shape_comps], requires_grad=False) # num_shape_comps x 115834 x 3
        self.scale_mean = nn.Parameter(model_dict['scale_mean'], requires_grad=False) # 76
        self.scale_comps = nn.Parameter(model_dict['scale_comps'][:self.num_scale_comps], requires_grad=False) # num_scale_comps x 76

        # Load Faces
        self.faces = nn.Parameter(model_dict['faces'][self.lod], requires_grad=False) # F x 3

        # Load Pose Correctives
        if self.verbose: print(f"Loading Pose Correctives...")
        self.posedirs = nn.Sequential(
            SparseLinear(159 * 6, 159 * 24, sparse_mask=model_dict['posedirs_sparse_mask'], bias=False),
            nn.ReLU(),
            nn.Linear(159 * 24, 115834 * 3, bias=False))
        self.posedirs.load_state_dict(model_dict['posedirs_state_dict'])
        for p in self.posedirs.parameters():
            p.requires_grad = False

        # Load LBS Function
        if self.verbose: print("Loading LBS function...")
        self.lbs_fn = LinearBlendSkinningCuda(
            dict(lbs_skin_json_path=osp.join(self.model_data_dir, "skinning.json"),
                 lbs_config_txt_path=osp.join(self.model_data_dir, "pose.cfg")))

        # Update Skinning Weights in LBS Function
        self.lbs_fn.skin_weights = model_dict['skin_weights']
        self.lbs_fn.skin_indices = model_dict['skin_indices']

        # Load Hand PCA
        self._load_hand_prior()

        # Load Expressions
        self.exprdirs = nn.Parameter(model_dict['exprdirs'][:self.num_expr_comps], requires_grad=False)

        # Load Keypoint Mapping
        if self.load_keypoint_mapping:
            self.keypoint_mapping = nn.Parameter(model_dict['keypoint_mapping_lod3_dict']['keypoint_mapping'], requires_grad=False)
            self.general_expression_skeleton_kps_dict = model_dict['keypoint_mapping_lod3_dict']['general_expression_skeleton_kps_dict']
            self.keypoint_names_308 = model_dict['keypoint_mapping_lod3_dict']['keypoint_names_308']
            
        # Map from LOD5 to other LODs.
        if self.lod != "lod5":
            self._process_lod()

    def _process_lod(self):
        if self.verbose: print(f"Converting to {self.lod}...")
        lod_mapping_dict = load_pickle(osp.join(self.model_data_dir, "lod_mapping.pkl"))
        mapping_matrix = torch.sparse.FloatTensor(*lod_mapping_dict[self.lod])

        # Straightforward/naive barycentric mapping from LOD5 to lower for model parameters
        self.shape_mean = nn.Parameter(mapping_matrix @ self.shape_mean, requires_grad=self.shape_mean.requires_grad)
        self.shape_comps = nn.Parameter((mapping_matrix @ self.shape_comps.permute(1, 0, 2).flatten(1, 2))\
                                        .reshape(-1, self.num_shape_comps, 3).permute(1, 0, 2), 
                                        requires_grad=self.shape_comps.requires_grad)
        final_posedirs_mapping = self.posedirs[-1].weight # V*3 x C
        final_posedirs_mapping = (mapping_matrix @ final_posedirs_mapping.reshape(115834, 3, -1).permute(0, 2, 1).flatten(1, 2))\
                                 .reshape(mapping_matrix.shape[0], -1, 3).permute(0, 2, 1).flatten(0, 1)
        self.posedirs[-1].weight = nn.Parameter(final_posedirs_mapping, requires_grad=self.posedirs[-1].weight.requires_grad)
        self.exprdirs = nn.Parameter((mapping_matrix @ self.exprdirs.permute(1, 0, 2).flatten(1, 2))\
                                        .reshape(-1, self.num_expr_comps, 3).permute(1, 0, 2), 
                                        requires_grad=self.exprdirs.requires_grad)
        
        # Skin weights are a bit more finicky, since lower LOD might draw from >8 joints.
        skin_weights = torch.zeros(self.lbs_fn.skin_weights.shape[0], 161)
        skin_weights = torch.scatter(skin_weights, dim=1, index=self.lbs_fn.skin_indices, src=self.lbs_fn.skin_weights)
        new_skin_weights = mapping_matrix @ skin_weights
        new_skin_weights_sort = new_skin_weights.sort(dim=1, descending=True)
        self.lbs_fn.skin_weights = new_skin_weights_sort.values[:, :8]
        self.lbs_fn.skin_indices = new_skin_weights_sort.indices[:, :8]
        self.lbs_fn.skin_weights = self.lbs_fn.skin_weights / self.lbs_fn.skin_weights.sum(dim=1, keepdim=True)

        # Finally, adjust lbs_fn to accept lower # of vertices.
        ## So, update other lbs_fn params that contain any notion of # of vertices
        self.lbs_fn.nr_vertices = mapping_matrix.shape[0]
        self.lbs_fn.out_skinned_mesh = torch.zeros((0, self.lbs_fn.nr_vertices, 3), dtype=self.lbs_fn.dtype)
        self.lbs_fn.out_grad_vertices = torch.zeros((0, self.lbs_fn.nr_vertices, 3), dtype=self.lbs_fn.dtype)
        self.lbs_fn.out_jac_surface_to_params = torch.zeros((0, self.lbs_fn.nr_vertices, self.lbs_fn.nr_params, 3), dtype=self.lbs_fn.dtype)

    def _load_hand_prior(self):
        if self.verbose: print(f"Loading Hand Prior...")
        hand_prior_dict = load_pickle(osp.join(self.model_data_dir, "hand_prior.pkl"))

        self.hand_pose_mean = nn.Parameter(hand_prior_dict['hand_pose_mean'], requires_grad=False)
        self.hand_pose_comps = nn.Parameter(hand_prior_dict['hand_pose_comps'][:self.num_hand_comps], requires_grad=False)
        self.hand_joint_mask_left = hand_prior_dict['hand_joint_mask_left']
        self.hand_joint_mask_right = hand_prior_dict['hand_joint_mask_right']
        
        # Store a pinv from joint rotations to model params
        # Why do this? Because hand PCA is in 6D space, so need to map back to Euler for LBS.
        self.param_transform_rots_pinv = nn.Parameter(
            torch.linalg.pinv(self.lbs_fn.param_transform[:, :468].reshape(161, 7, 468)[:, 3:6, :].flatten(0, 1)), requires_grad=False)

        # Figure out which model params belong to hand rotations (the only thing we're replacing)
        def is_hand(name):
            return "thumb" in name or "index" in name or "middle" in name or "ring" in name or "pinky" in name
        self.hand_idxs = torch.LongTensor([idx
                                           for idx, name in enumerate(self.lbs_fn.model_param_names[:-76])
                                           if is_hand(name) and name[-3:] not in ['_tx', '_ty', '_tz']])

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 468

        # This drops in the hand poses from hand_pose_params (PCA 6D) into full_pose_params.
        # Split into left and right hands
        left_hand_params, right_hand_params = torch.split(
            hand_pose_params, [self.num_hand_comps, self.num_hand_comps], dim=1)

        # This is just a 1127 with only hand params inside
        dummy_joint_params = torch.zeros(len(full_pose_params), 1127).to(full_pose_params.device)
        dummy_joint_params[:, self.hand_joint_mask_left] = batchXYZfrom6D(
            (self.hand_pose_mean + torch.einsum('da,ab->db', left_hand_params, self.hand_pose_comps)).reshape(-1, 22, 6)).flatten(1, 2)
        dummy_joint_params[:, self.hand_joint_mask_right] = batchXYZfrom6D(
            (self.hand_pose_mean + torch.einsum('da,ab->db', right_hand_params, self.hand_pose_comps)).reshape(-1, 22, 6)).flatten(1, 2)
        dummy_joint_params_rots = dummy_joint_params.reshape(dummy_joint_params.shape[0], 161, 7)[:, :, 3:6].flatten(1, 2)

        # Now p-inv it to 468
        hand_pose_params_non_pca = dummy_joint_params_rots @ self.param_transform_rots_pinv[self.hand_idxs, :].T

        # Drop it in
        full_pose_params[:, self.hand_idxs] = hand_pose_params_non_pca.to(full_pose_params)

        return full_pose_params # B x 468

    def forward(self,
                global_trans,
                global_rot,
                body_pose_params,
                hand_pose_params,
                scale_params,
                shape_params,
                expr_params=None,
                return_keypoints=False,
                do_pcblend=True,
                return_joint_coords=False,
                scale_offsets=None,
                vertex_offsets=None):

        # Convert from scale and shape params to actual scales and vertices
        ## Add singleton batches in case...
        if len(scale_params.shape) == 1:
            scale_params = scale_params[None]
        if len(shape_params.shape) == 1:
            shape_params = shape_params[None]
        ## Convert scale...
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
        if scale_offsets is not None:
            scales = scales + scale_offsets
        ## Convert shape...
        template_verts = self.shape_mean[None, :, :] + torch.einsum('da,abc->dbc', shape_params, self.shape_comps)
        if vertex_offsets is not None:
            template_verts = template_verts + vertex_offsets
        ## Put in expressions
        if expr_params is not None:
            expression_offsets = torch.einsum('cab,dc->dab', self.exprdirs, expr_params)
            template_verts = template_verts + expression_offsets

        # Now, figure out the pose.
        ## 100 here is because it's more stable to optimize global translation in meters.
        ## LBS works in cm (global_scale is [1, 1, 1]).
        full_pose_params = torch.cat([global_trans * 100, global_rot, body_pose_params], dim=1) # B x 468
        ## Put in hands
        full_pose_params = self.replace_hands_in_pose(full_pose_params, hand_pose_params)
        ## Disallow joint translations. 
        ## This results in 99 body rotations, 132 hand rotations and 6 globals.
        pose_trans_mask = [name[-3:] in ["_tx", "_ty", "_tz"] and "_root_" not in name 
                           for name in self.lbs_fn.model_param_names[:468]]
        full_pose_params[:, pose_trans_mask] *= 0 # B x 468
        ## Get the 1127 joint params
        if scales.shape[0] == 1:
            scales = scales.squeeze(0)
        model_params = self.lbs_fn.assemblePoseAndScale(full_pose_params, scales)
        joint_params = self.lbs_fn.jointParamsFromModelParams(model_params)
        
        # Get pose correctives
        if do_pcblend:
            pose_6d_feats = get_pose_feats_6d_batched_from_joint_params(joint_params)
            pose_corrective_offsets = self.posedirs(pose_6d_feats).reshape(len(pose_6d_feats), -1, 3)
            template_verts = template_verts + pose_corrective_offsets

        # Finally, LBS
        # (
        #     out_joint_state_t,
        #     out_joint_state_r,
        #     out_joint_state_s,
        #     stash_joint_grad_t,
        #     stash_joint_grad_r,
        # ) = torch.ops.lbs_pytorch_ops.forward_kinematics_malloced(
        #     joint_params, self.lbs_fn.joint_parents, self.lbs_fn.joint_offset, self.lbs_fn.joint_rotation
        # )
        curr_skinned_verts, curr_joint_coords = self.lbs_fn.forwardFromJointParams(joint_params, template_verts)[:2]
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            assert self.load_keypoint_mapping
            model_vert_joints = torch.cat([curr_skinned_verts, curr_joint_coords], dim=1) # B x (num_verts + 161) x 3
            model_keypoints_pred = (self.keypoint_mapping @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)).reshape(308, -1, 3).permute(1, 0, 2)
            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]

        if len(to_return) == 1: return to_return[0]
        else: return tuple(to_return)