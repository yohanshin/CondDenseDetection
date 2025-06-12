import os
import os.path as osp
import pickle
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2
import imageio

def load_json(f):
    with open(f, "r") as ff:
        return json.load(ff)


def save_json(obj, f):
    with open(f, "w+") as ff:
        json.dump(obj, ff)

def load_pickle(f):
    with open(f, "rb") as ff:
        return pickle.load(ff)

def save_pickle(obj, f):
    with open(f, "wb+") as ff:
        pickle.dump(obj, ff)
        
def batch6DFromXYZ(r, return_9D=False):
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1])+[3, 3], dtype=r.dtype).to(r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy
    
    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result
        
# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L82
def batchXYZfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1) # ... x 3 x 3

    # Now get it into euler
    # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L412
    sy = torch.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
    singular = sy < 1e-6
    singular = singular.float()
        
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(-matrix[..., 2, 0], sy)
    z = torch.atan2(matrix[..., 1, 0],matrix[..., 0, 0])
    
    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0
        
    out_euler = torch.zeros_like(matrix[..., 0])
    out_euler[..., 0] = x * (1 - singular) + xs * singular
    out_euler[..., 1] = y * (1 - singular) + ys * singular
    out_euler[..., 2] = z * (1 - singular) + zs * singular
    
    return out_euler
    
def get_pose_feats_6d_batched_from_joint_params(joint_params):
    joint_euler_angles = joint_params.reshape(-1, 161, 7)[:, 2:, 3:6] # B x 159 x 3
    joint_6d_feat = batch6DFromXYZ(joint_euler_angles)
    joint_6d_feat[:, :, 0] -= 1 # so all 0 when no rotation.
    joint_6d_feat[:, :, 4] -= 1 # so all 0 when no rotation.
    joint_6d_feat = joint_6d_feat.flatten(1, 2)
    return joint_6d_feat

class SparseLinear(nn.Module):
    def __init__(self, in_channels, out_channels, sparse_mask, bias=True):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        
        self.sparse_indices = nn.Parameter(sparse_mask.nonzero().T, requires_grad=False) # 2 x K
        self.sparse_shape = sparse_mask.shape
        
        weight = torch.zeros(out_channels, in_channels)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.bias = None

        # Initialize!
        for out_idx in range(out_channels):
            # By default, self.weight is initialized with kaiming,
            # fan_in, linear default.
            # Here, the entire thing (even stuff that should be 0) are initialized,
            # only relevant stuff will be kept
            fan_in = sparse_mask[out_idx].sum()
            gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            weight[out_idx].uniform_(-bound, bound)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in)
                self.bias[out_idx:out_idx+1].uniform_(-bound, bound)
        self.sparse_weight = nn.Parameter(weight[self.sparse_indices[0], self.sparse_indices[1]])
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)
    
    def forward(self, x):
        curr_weight = torch.sparse_coo_tensor(self.sparse_indices, self.sparse_weight, self.sparse_shape)
        if self.bias is None:
            return (curr_weight @ x.T).T
            # return torch.sparse.mm(curr_weight, x.T).T
        else:
            return (curr_weight @ x.T).T + self.bias
            # return torch.sparse.mm(curr_weight, x.T).T + self.bias

    def __repr__(self):
        return f"SparseLinear(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None})"

class PosePriorVAE(nn.Module):
    def __init__(self, input_dim=198, feature_dim=512, latent_dim=32, dropout_prob=0.2, eps=1e-6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU())
        
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, input_dim))

        self.eps = eps

    def latent_to_462(self, x, jaw_params=None):
        nonhand_468 = torch.LongTensor([9,10,11,15,16,17,21,22,23,27,28,29,33,34,35,39,40,41,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71,75,76,77,81,82,83,87,88,89,93,94,95,99,100,101,105,106,107,111,112,113,117,118,119,123,124,125,129,130,131,135,136,137,141,142,143,147,148,149,153,154,155,159,160,161,165,166,167,171,172,173,177,178,179,183,184,185,189,190,191,195,196,197,465,466,467])
        nonhand_462 = nonhand_468 - 6
        sampled_poses = self.decoder(x)
        sampled_poses_euler = batchXYZfrom6D(sampled_poses.reshape(-1, 33, 6)).reshape(-1, 99)
        res = torch.zeros(len(x), 462).to(x.device)
        res[:, nonhand_462] = sampled_poses_euler
        if jaw_params is not None:
            res[:, [459, 460, 461]] = jaw_params
        return res

class BodyPoseGMMPrior(nn.Module):
    def __init__(self, model_data_dir, gmm_comps=32):
        super().__init__()

        assert gmm_comps in [8, 16, 32]
        (self.model_mu, self.model_var, self.model_pi, self.model_precision, self.model_logdet, self.model_logweights) = [
            nn.Parameter(tmp, requires_grad=False) 
            for tmp in load_pickle(osp.join(model_data_dir, "full_body_gmm_prior.pkl"))[gmm_comps]]
        
        self.log_2pi = self.model_mu.shape[2] * np.log(2. * math.pi)

    def forward(self, pose_params):
        sub_x_mu = pose_params[:, None, :] - self.model_mu #[N, sub_K, D]
        sub_x_mu_T_precision = (sub_x_mu.transpose(0, 1) @ self.model_precision).transpose(0, 2)
        sub_x_mu_T_precision_x_mu = (sub_x_mu_T_precision.squeeze(2) * sub_x_mu).sum(dim=2, keepdim=True) #[N, sub_K, 1]
        log_prob = sub_x_mu_T_precision_x_mu
        log_prob = log_prob + self.log_2pi
        log_prob = log_prob - self.model_logdet
        log_prob = log_prob * -0.5
        log_prob = log_prob + self.model_logweights
        log_prob = log_prob.squeeze(2)
        log_prob = log_prob.amax(dim=1)

        return -log_prob
        
class BodyPosePerJointGMMPrior(nn.Module):
    def __init__(self, model_data_dir, num_comps=4):
        super().__init__()
        
        assert num_comps in [1, 2, 4, 8]
        (self.model_mu, self.model_var, self.model_pi, self.model_precision, self.model_logdet, self.model_logweights) = \
        [
            nn.Parameter(torch.stack(tmp, dim=1), requires_grad=False) 
            for tmp in list(zip(*load_pickle(osp.join(model_data_dir, "per_joint_gmm_prior.pkl"))[num_comps]))
        ]

        self.log_2pi = self.model_mu.shape[3] * np.log(2. * math.pi)

    def forward(self, pose_params):
        pose_params = pose_params.reshape(-1, 33, 6)
        sub_x_mu = pose_params[:, :, None, :] - self.model_mu #[N, 33, sub_K, D]
        sub_x_mu_T_precision = torch.einsum('afbc,dfbce->afbde', sub_x_mu[:, :, :, :], self.model_precision[:, :, :, :, :])
        sub_x_mu_T_precision_x_mu = (sub_x_mu_T_precision.squeeze(3) * sub_x_mu).sum(dim=3, keepdim=True) # N x 33 x sub_K x 1
        log_prob = sub_x_mu_T_precision_x_mu
        log_prob = log_prob + self.log_2pi
        log_prob = log_prob - self.model_logdet.permute(1, 0, 2) # 33 x sub_K x 1
        log_prob = log_prob * -0.5
        log_prob = log_prob + self.model_logweights
        log_prob = log_prob.squeeze(3)
        # print(log_prob.argmax(dim=2))
        log_prob = log_prob.amax(dim=2)

        return -log_prob

def resize_image(image_array, scale_factor, interpolation=cv2.INTER_LINEAR):
    new_height = int(image_array.shape[0] // scale_factor)
    new_width = int(image_array.shape[1] // scale_factor)
    resized_image = cv2.resize(image_array, (new_width, new_height), interpolation=interpolation)
    
    return resized_image



def compact_cont_to_model_params_hand(hand_cont):
    # These are ordered by joint, not model params ^^
    assert hand_cont.shape[-1] == 54
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27
    # Mask of 3DoFs into hand_cont
    mask_cont_threedofs = torch.cat([torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_cont
    mask_cont_onedofs = torch.cat([torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])
    # Mask of 3DoFs into hand_model_params
    mask_model_params_threedofs = torch.cat([torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_model_params
    mask_model_params_onedofs = torch.cat([torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])

    # Convert hand_cont to eulers
    ## First for 3DoFs
    hand_cont_threedofs = hand_cont[..., mask_cont_threedofs].unflatten(-1, (-1, 6))
    hand_model_params_threedofs = batchXYZfrom6D(hand_cont_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_cont_onedofs = hand_cont[..., mask_cont_onedofs].unflatten(-1, (-1, 2)) #(sincos)
    hand_model_params_onedofs = torch.atan2(hand_cont_onedofs[..., -2], hand_cont_onedofs[..., -1])

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27).to(hand_cont)
    hand_model_params[..., mask_model_params_threedofs] = hand_model_params_threedofs
    hand_model_params[..., mask_model_params_onedofs] = hand_model_params_onedofs

    return hand_model_params

def compact_model_params_to_cont_hand(hand_model_params):
    # These are ordered by joint, not model params ^^
    assert hand_model_params.shape[-1] == 27
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27
    # Mask of 3DoFs into hand_cont
    mask_cont_threedofs = torch.cat([torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_cont
    mask_cont_onedofs = torch.cat([torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])
    # Mask of 3DoFs into hand_model_params
    mask_model_params_threedofs = torch.cat([torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order])
    # Mask of 1DoFs (including 2DoF) into hand_model_params
    mask_model_params_onedofs = torch.cat([torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order])

    # Convert eulers to hand_cont hand_cont
    ## First for 3DoFs
    hand_model_params_threedofs = hand_model_params[..., mask_model_params_threedofs].unflatten(-1, (-1, 3))
    hand_cont_threedofs = batch6DFromXYZ(hand_model_params_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_model_params_onedofs = hand_model_params[..., mask_model_params_onedofs]
    hand_cont_onedofs = torch.stack([hand_model_params_onedofs.sin(), hand_model_params_onedofs.cos()], dim=-1).flatten(-2, -1)

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_cont = torch.zeros(*hand_model_params.shape[:-1], 54).to(hand_model_params)
    hand_cont[..., mask_cont_threedofs] = hand_cont_threedofs
    hand_cont[..., mask_cont_onedofs] = hand_cont_onedofs

    return hand_cont