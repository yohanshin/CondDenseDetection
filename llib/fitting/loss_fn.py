import os
import os.path as osp

import torch
import numpy as np
from torch import nn

funcl2 = lambda x: torch.sum(x**2)
funcl1 = lambda x: torch.sum(torch.abs(x**2))



class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


class LossFunction(nn.Module):
    def __init__(self, verts_idxs, device, center_idxs=[11, 12], 
                 **kwargs):
        super().__init__()
        self.robustifier = GMoF(rho=100.0)
        
        self.verts_idxs = verts_idxs
        self.to(device)
    
    def reset_loss_weights(self, loss_dict):
        for key, val in loss_dict.items():
            if 'lw_' in key:
                setattr(self, key, float(val))

    def ldmks_loss(self, pred_verts, ldmks, scales, intrinsic, **kwargs):
        
        # Projection
        conf = ldmks[..., -1:] if ldmks.size(-1) == 3 else 1
        
        # pred_joints3d = torch.matmul(self.J_regressor, pred_verts)
        pred_ldmks_3d = torch.index_select(pred_verts, dim=1, index=self.verts_idxs)
        pred_ldmks = torch.matmul(intrinsic, torch.div(pred_ldmks_3d, pred_ldmks_3d[..., -1:]).mT).mT[..., :2]
        residual = (pred_ldmks - ldmks[..., :2]) * (conf ** 2)
        scales = scales.reshape(residual.shape[0], 1, 1)
        squared_res = self.robustifier(residual)
        squared_res = squared_res / ((1e-3 + scales))
        loss_per_person = squared_res.sum(dim=[1, 2])
        
        return loss_per_person.sum(), loss_per_person.detach().cpu().numpy()

    def regularize_loss(self, params, init_params):
        loss = 0
        losses_per_subject = []
        for k in init_params.keys():
            scale = 1e2 if k in ["shape_params", "scale_params"] else 1.0
            loss_per_subject = torch.nn.functional.l1_loss(params[k], init_params[k], reduction='none').sum(dim=-1)
            loss += loss_per_subject.sum() * scale
            losses_per_subject.append(loss_per_subject.detach().cpu().numpy())
        return loss, np.stack(losses_per_subject).sum(0)
    
    def pose_prior_loss(self, params, pose_prior):
        if "VAE" in pose_prior.__name__:
            loss_per_subject = torch.sum(params["latent_params"] ** 2, dim=-1)
        elif "GMM" in pose_prior.__name__:
            loss_per_subject = pose_prior(params["body_pose_params"])
        loss = loss_per_subject.sum()
        return loss, loss_per_subject.detach().cpu().numpy()
    
    def shape_prior_loss(self, params):
        shape_params = params["shape_params"]
        scale_params = params["scale_params"]
        loss_per_subject = torch.sum(shape_params ** 2, dim=-1) + torch.sum(scale_params ** 2, dim=-1)
        loss = loss_per_subject.sum()
        return loss, loss_per_subject.detach().cpu().numpy()

    def shape_consistency_loss(self, params):
        shape_params = params["shape_params"]
        scale_params = params["scale_params"]
        N_frames = shape_params.shape[0]

        loss = shape_params.std(0).mean() * N_frames + scale_params.std(0).mean() * N_frames
        
        return loss

    def smoothing_loss(self, verts, params):
        def smooth(est):
            "smooth body"
            interp = est.clone().detach()
            interp[1:-1] = (interp[:-2] + interp[2:])/2
            loss = funcl2(est[1:-1] - interp[1:-1])
            return loss/(est.shape[0] - 2)
        
        smoothing_loss = 0
        smoothing_loss += smooth(verts)
        for key in ["global_trans", "global_rot", "body_pose_params", "hand_pose_params"]:
            smoothing_loss += smooth(params[key])
        return smoothing_loss
    
    def forward(self, 
                pred_verts, 
                params, 
                init_params, 
                ldmks, 
                scales, 
                intrinsic, 
                pose_prior=None):
        
        losses = dict()
        losses_per_subj = dict()
        n_subj = pred_verts.shape[0]
        
        if hasattr(self, 'lw_ldmks'):
            loss, loss_per_subj = self.ldmks_loss(pred_verts, ldmks, scales, intrinsic)
            losses['ldmks'] = loss * self.lw_ldmks
            losses_per_subj['ldmks'] = loss_per_subj * self.lw_ldmks

        if hasattr(self, 'lw_reg'):
            loss, loss_per_subj = self.regularize_loss(params, init_params)
            losses['regularize'] = loss * self.lw_reg
            losses_per_subj['regularize'] = loss_per_subj * self.lw_reg
        
        if hasattr(self, 'lw_pose'):
            loss, loss_per_subj = self.pose_prior_loss(params, pose_prior)
            losses['pose_prior'] = loss * self.lw_pose
            losses_per_subj['pose_prior'] = loss_per_subj * self.lw_pose

        if hasattr(self, 'lw_shape'):
            loss, loss_per_subj = self.shape_prior_loss(params)
            losses['shape_prior'] = loss * self.lw_shape
            losses_per_subj['shape_prior'] = loss_per_subj * self.lw_shape

        if hasattr(self, 'lw_smoothing'):
            loss = self.smoothing_loss(pred_verts, params)
            losses['smoothing'] = loss * self.lw_smoothing
            
        if hasattr(self, 'lw_shape_consist'):
            loss = self.shape_consistency_loss(params)
            losses['shape_consist'] = loss * self.lw_shape_consist

        total_loss = sum(losses.values())
        self.loss_dict = {}
        for k, v in losses.items():
            if k in ['shape_consist', 'smoothing']:
                self.loss_dict[k] = f'{v.item():.1f}'
            else:
                self.loss_dict[k] = f'{v.item() / n_subj:.1f}'
        self.loss_per_subj = np.stack([v for v in losses_per_subj.values()], axis=-1).sum(-1)
        
        return total_loss