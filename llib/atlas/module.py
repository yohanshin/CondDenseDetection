import os.path as osp
import math
import numpy as np
import torch
import torch.nn as nn

from .utils import load_pickle, batchXYZfrom6D, batch6DFromXYZ


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
    
    @property
    def __name__(self,):
        return "BodyPoseVAE"

    def pose462_to_latent(self, x):
        nonhand_468 = torch.LongTensor([9,10,11,15,16,17,21,22,23,27,28,29,33,34,35,39,40,41,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71,75,76,77,81,82,83,87,88,89,93,94,95,99,100,101,105,106,107,111,112,113,117,118,119,123,124,125,129,130,131,135,136,137,141,142,143,147,148,149,153,154,155,159,160,161,165,166,167,171,172,173,177,178,179,183,184,185,189,190,191,195,196,197,465,466,467])
        nonhand_462 = nonhand_468 - 6
        nonhand_poses = x[..., nonhand_462].reshape(-1, 33, 3)
        input_vector = batch6DFromXYZ(nonhand_poses).reshape(-1, 198)
        res = self.encoder(input_vector)
        latent = self.mu(res)
        return latent

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

    @property
    def __name__(self,):
        return "BodyPoseGMM"

    def pose462_to_vector(self, x):
        nonhand_468 = torch.LongTensor([9,10,11,15,16,17,21,22,23,27,28,29,33,34,35,39,40,41,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71,75,76,77,81,82,83,87,88,89,93,94,95,99,100,101,105,106,107,111,112,113,117,118,119,123,124,125,129,130,131,135,136,137,141,142,143,147,148,149,153,154,155,159,160,161,165,166,167,171,172,173,177,178,179,183,184,185,189,190,191,195,196,197,465,466,467])
        nonhand_462 = nonhand_468 - 6
        nonhand_poses = x[..., nonhand_462].reshape(-1, 33, 3)
        input_vector = batch6DFromXYZ(nonhand_poses).reshape(-1, 198)
        return input_vector

    def forward(self, pose_params):
        if pose_params.size(-1) != self.model_mu.size(-1):
            pose_params = self.pose462_to_vector(pose_params)
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
