import os
import glob
import torch


def load_model(cfg, device):
    if "regressor" in cfg.model.name or "encdec" in cfg.model.name:
        from llib.trainer.regressor import DensLnmkRegressor as ModelCls
    elif "diffusion" in cfg.model.name:
        from llib.trainer.diffusion import DensLnmkDiffusion as ModelCls
    
    return ModelCls(cfg).to(device)


def load_checkpoint(model, ckpt_dir):
    checkpoint_pth_list = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    checkpoint_pth = checkpoint_pth_list[-1]    # The most recent one
    state_dict = torch.load(checkpoint_pth, weights_only=False)['state_dict']
    model.load_state_dict(state_dict, strict=False)
    return model