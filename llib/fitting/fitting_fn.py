from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm

from .loss_fn import LossFunction

class FittingFunction(object):
    def __init__(self, 
                 dtype=torch.float32,
                 device="cuda"):
        super(FittingFunction, self).__init__()

        self.dtype = dtype
        self.device = device

        
    def atlas_forward(self, atlas, params):
        global_trans = params["global_trans"]
        global_rot = params["global_rot"]
        body_pose_params = params["body_pose_params"]
        hand_pose_params = params["hand_pose_params"]
        scale_params = params["scale_params"]
        shape_params = params["shape_params"]
        expr_params = params["expr_params"]
        
        # verts, j3d = atlas(
        verts = atlas(
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
        return verts

    def forward_loop(self, 
                     atlas,
                     pose_prior,
                     params, 
                     init_params,
                     ldmks,
                     scales,
                     intrinsic,
                     loss_fn,
                     optimizer,
                     scheduler,
                     max_iters,
                     **kwargs):
        
        
        closure = self.create_fitting_closure(
                optimizer=optimizer,
                scheduler=scheduler,
                atlas=atlas,
                pose_prior=pose_prior,
                params=params,
                init_params=init_params,
                ldmks=ldmks,
                scales=scales,
                intrinsic=intrinsic,
                loss_fn=loss_fn,
        )
        
        loss_accm = []
        
        for i in (pbar := tqdm(range(max_iters), 
                               desc='Fitting ...', 
                               dynamic_ncols=True, 
                               leave=False)):
            loss = optimizer.step(closure)

            msg = f'Iter {i+1}  |  '
            for k, v in loss_fn.loss_dict.items():
                msg += f'{k}: {v}  |  '
            pbar.set_postfix_str(msg)
            loss_accm.append(loss_fn.loss_per_subj)
        
        return np.stack(loss_accm, axis=-1)
    
    def configure_optimizers(self, params, optim_kwargs):
        if optim_kwargs['params'] == 'all':
            optim_params = [params[key].requires_grad_(True) for key in params.keys()]
        else:
            optim_params = [params[key].requires_grad_(True) for key in optim_kwargs['params']]
        
        if optim_kwargs["optim_type"] == "adamw":
            optimizer = torch.optim.AdamW(
                optim_params, lr=optim_kwargs.get('lr', 1e-3), betas=(0.9, 0.999)
            )
        else:
            raise NotImplementedError(f"No optimizer type {optim_kwargs['optim_type']} implemented !")
        
        if "lr_scheduler" in optim_kwargs:
            if optim_kwargs['lr_scheduler'] == 'cosine':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=optim_kwargs['num_steps'], eta_min=0.0
                )
            else:
                raise NotImplementedError(f"No scheduler type {optim_kwargs['lr_scheduler']} implemented !")
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler
            

    def run_fitting(self, atlas, pose_prior, init_params, ldmks, intrinsic, scales, optim_kwargs_list, smplx2ldmks):
        # Copy initial ATLAS parameters
        params = deepcopy(init_params)
        
        # Construct loss function
        loss_fn = LossFunction(verts_idxs=smplx2ldmks, device=self.device)

        losses = []
        for optim_kwargs in optim_kwargs_list:
            optimizer, scheduler = self.configure_optimizers(params, optim_kwargs)
            loss_fn.reset_loss_weights(optim_kwargs)
            
            max_iters = optim_kwargs['num_steps']
            loss = self.forward_loop(
                atlas,
                pose_prior,
                params,
                init_params,
                ldmks,
                scales,
                intrinsic,
                loss_fn,
                optimizer,
                scheduler,
                max_iters
            )

            losses.append(loss)

        for k, v in params.items():
            params[k] = v.detach()

        return params, np.concatenate(losses, axis=1)


    def create_fitting_closure(self, 
                               optimizer,
                               scheduler,
                               atlas,
                               pose_prior,
                               params,
                               init_params,
                               ldmks,
                               scales,
                               intrinsic,
                               loss_fn):
        def closure(backward=True):
            if backward:
                optimizer.zero_grad()
            
            pred_verts = self.atlas_forward(atlas, params)
            total_loss = loss_fn(pred_verts=pred_verts,
                                 params=params,
                                 init_params=init_params,
                                 ldmks=ldmks,
                                 scales=scales,
                                 intrinsic=intrinsic,
                                 pose_prior=pose_prior
                                 )
            
            if backward:
                total_loss.backward(create_graph=False)
                optimizer.step()
                scheduler.step()
            
            return total_loss

        return closure