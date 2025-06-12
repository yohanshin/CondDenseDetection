import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn

from linear_blend_skinning_cuda import LinearBlendSkinningCuda

from .atlas_utils import (load_pickle, 
                          batch6DFromXYZ, 
                          batchXYZfrom6D, 
                          get_pose_feats_6d_batched_from_joint_params, 
                          SparseLinear, 
                          compact_cont_to_model_params_hand)

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
        # if self.load_keypoint_mapping: assert self.lod == 'lod3', "308 Keypoints only supported for LOD3"
        self.verbose = verbose

        # Load Model
        if self.verbose: print(f"Loading ATLAS from {model_data_dir} ...")
        model_dict = load_pickle(osp.join(model_data_dir, "params.pkl"))
        if self.verbose: print(f"Done loading pkl!")

        # Load Shape & Scale Bases
        self.shape_mean = nn.Parameter(model_dict['shape_mean'], requires_grad=False) # 115834 x 3
        self.shape_comps = nn.Parameter(model_dict['shape_comps'][:self.num_shape_comps], requires_grad=False) # num_shape_comps x 115834 x 3
        self.scale_mean = nn.Parameter(model_dict['scale_mean'], requires_grad=False) # 68
        assert self.num_scale_comps >= 18 and (self.num_scale_comps - 18) % 2 == 0, \
            "num_scale_comps must be at least 18, as each body scale has its own component, and num_scale_comps - 18 must be divisible by 2, since each hand then gets (num_scale_comps - 18) / 2 components."

        self.scale_comps = nn.Parameter(torch.cat([
            model_dict['scale_comps'][:18], # Full-body components
            model_dict['scale_comps'][18:18+25][:(self.num_scale_comps - 18) // 2], # Left hand components
            model_dict['scale_comps'][18+25:18+50][:(self.num_scale_comps - 18) // 2], # Left hand components
        ], dim=0), requires_grad=False) # num_scale_comps x 68

        # Load Faces
        self.faces = nn.Parameter(model_dict['faces'][self.lod], requires_grad=False) # F x 3

        # Load Pose Correctives
        if self.verbose: print(f"Loading Pose Correctives...")
        self.posedirs = None

        # Load LBS Function
        if self.verbose: print("Loading LBS function...")
        self.lbs_fn = LinearBlendSkinningCuda(
            dict(
                lbs_skin_json_path=osp.join(self.model_data_dir, 'trinity_texelheadsewn_body_lod0_skin_v4.0.dev7.json'),
                lbs_config_txt_path=osp.join(self.model_data_dir, 'compact_v6_0_withjaw_latest_25_06_05.model')
            ))

        # Load Hand PCA
        self._load_hand_prior()

        # Load Expressions
        self.exprdirs = nn.Parameter(model_dict['exprdirs'][:self.num_expr_comps], requires_grad=False)

        # Load Keypoint Mapping
        if self.load_keypoint_mapping:
            assert self.lod in ["lod3"]
            self.keypoint_mapping = nn.Parameter(load_pickle(osp.join(self.model_data_dir, "lod3_joint_to_kps_v4.pkl")), requires_grad=False)
            self.general_expression_skeleton_kps_dict = model_dict['keypoint_mapping_lod3_dict']['general_expression_skeleton_kps_dict']
            self.keypoint_names_308 = model_dict['keypoint_mapping_lod3_dict']['keypoint_names_308']
                
            
        # Map from LOD5 to other LODs.
        if self.lod != "lod5":
            self._process_lod()

        # Load parameter limits
        model_param_limits = []
        for model_param_name in self.lbs_fn.model_param_names:
            for limit in self.lbs_fn.limits:
                if limit['str'] == model_param_name:
                    model_param_limits.append([*limit['limits'], limit['weight']])
                    break
            if limit['str'] != model_param_name:
                assert "root" in model_param_name
                model_param_limits.append([0, 0, 0])
        self.model_param_limits = nn.Parameter(torch.FloatTensor(model_param_limits), requires_grad=False)

        # Make life easier by having a mask of flexibles
        self.flexible_model_params_mask = nn.Parameter(
            (self.model_param_limits[:, 0] == 0) 
            & (self.model_param_limits[:, 1] == 0) 
            & (self.model_param_limits[:, 2] != 0), requires_grad=False)

    def _process_lod(self):
        if self.verbose: print(f"Converting to {self.lod}...")
        lod_mapping_dict = load_pickle(osp.join(self.model_data_dir, "lod_mapping.pkl"))
        mapping_matrix = torch.sparse.FloatTensor(*lod_mapping_dict[self.lod])

        # Straightforward/naive barycentric mapping from LOD5 to lower for model parameters
        self.shape_mean = nn.Parameter(mapping_matrix @ self.shape_mean, requires_grad=self.shape_mean.requires_grad)
        self.shape_comps = nn.Parameter((mapping_matrix @ self.shape_comps.permute(1, 0, 2).flatten(1, 2))\
                                        .reshape(-1, self.num_shape_comps, 3).permute(1, 0, 2), 
                                        requires_grad=self.shape_comps.requires_grad)
        self.exprdirs = nn.Parameter((mapping_matrix @ self.exprdirs.permute(1, 0, 2).flatten(1, 2))\
                                        .reshape(-1, self.num_expr_comps, 3).permute(1, 0, 2), 
                                        requires_grad=self.exprdirs.requires_grad)
        
        # Skin weights are a bit more finicky, since lower LOD might draw from >8 joints.
        skin_weights = torch.zeros(self.lbs_fn.skin_weights.shape[0], 216)
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
        hand_prior_dict = load_pickle(osp.join(self.model_data_dir, "hand_pose_prior_compact.pkl"))

        self.hand_pose_mean = nn.Parameter(hand_prior_dict['hand_pose_mean'], requires_grad=False)
        self.hand_pose_comps = nn.Parameter(hand_prior_dict['hand_pose_comps'][:self.num_hand_comps], requires_grad=False)
        self.hand_joint_idxs_left = hand_prior_dict['hand_joint_idxs_left']
        self.hand_joint_idxs_right = hand_prior_dict['hand_joint_idxs_right']

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 139

        # This drops in the hand poses from hand_pose_params (PCA 6D) into full_pose_params.
        # Split into left and right hands
        left_hand_params, right_hand_params = torch.split(
            hand_pose_params, [self.num_hand_comps, self.num_hand_comps], dim=1)

        # Change from cont to model params
        left_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean + torch.einsum('da,ab->db', left_hand_params, self.hand_pose_comps))
        right_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean + torch.einsum('da,ab->db', right_hand_params, self.hand_pose_comps))

        # Drop it in
        full_pose_params[:, self.hand_joint_idxs_left] = left_hand_params_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_hand_params_model_params

        return full_pose_params # B x 207

    def forward(self,
                global_trans,
                global_rot,
                body_pose_params,
                hand_pose_params,
                scale_params,
                shape_params,
                expr_params=None,
                return_keypoints=False,
                do_pcblend=False,
                return_joint_coords=False,
                return_model_params=False,
                mask_flexibles=False,
                mask_flexibles_pose_only=False,
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
        ## 10 here is because it's more stable to optimize global translation in meters.
        ## LBS works in cm (global_scale is [1, 1, 1]).
        full_pose_params = torch.cat([global_trans * 10, global_rot, body_pose_params], dim=1) # B x 204
        ## Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(full_pose_params, hand_pose_params)
        ## Get the 1512 joint params
        if scales.shape[0] == 1:
            scales = scales.squeeze(0)
        ## Optinally mask out flexibles
        if mask_flexibles and mask_flexibles_pose_only:
            full_pose_params = full_pose_params * ~self.flexible_model_params_mask[None, :139]
        model_params = self.lbs_fn.assemblePoseAndScale(full_pose_params, scales)
        ## Optinally mask out flexibles
        if mask_flexibles and not mask_flexibles_pose_only:
            model_params = model_params * ~self.flexible_model_params_mask[None, :]
        joint_params = self.lbs_fn.jointParamsFromModelParams(model_params)

        # Finally, LBS
        curr_skinned_verts, curr_joint_coords = self.lbs_fn.forwardFromJointParams(joint_params, template_verts)[:2]
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            assert self.load_keypoint_mapping
            model_vert_joints = torch.cat([curr_skinned_verts, curr_joint_coords], dim=1) # B x (num_verts + 204) x 3
            model_keypoints_pred = (self.keypoint_mapping @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)).reshape(-1, model_vert_joints.shape[0], 3).permute(1, 0, 2)
            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]

        if len(to_return) == 1: return to_return[0]
        else: return tuple(to_return)