import os

import torch
import torch.distributed as dist
from .base_trainer import BaseTrainer
from ..models.backbone.vit import ViT
from ..models.losses import JointGNLLLoss
from ..models.head.diffusion_head import EncDecPerLandmark
# from ..models.head.regression_head import EncDecPerLandmark
from ..models.utils.visualization import compare_results_denseldmks2d

from ..models.diffusion import gaussian_diffusion
from ..models.diffusion.wrapper import ModelWrapper
from ..models.diffusion.utils import create_gaussian_diffusion
from ..models.diffusion.respace import SpacedDiffusionQuestDiff
from ..models.diffusion.fp16_utils import MixedPrecisionTrainer

class DensLnmkDiffusion(BaseTrainer):
    def __init__(self, cfg, viz_dir=None):
        super(DensLnmkDiffusion, self).__init__(cfg)

        backbone_cfg = {k: v for k, v in cfg.backbone.items() if not k in ['type', 'name']}
        self.backbone_cfg = backbone_cfg
        self.backbone = ViT(**backbone_cfg).to(self.device)
        
        decoder_cfg = {k: v for k, v in cfg.model['decoder'].items() if k != 'layer_name'}
        self.criterion = JointGNLLLoss(loss_weights=cfg.loss_weights)

        decoder = EncDecPerLandmark(**decoder_cfg)
        self.diffusion_train = create_gaussian_diffusion(cfg, 
                                                gd=gaussian_diffusion, 
                                                return_class=SpacedDiffusionQuestDiff, 
                                                device=self.device)
        self.diffusion_eval = create_gaussian_diffusion(cfg, 
                                                gd=gaussian_diffusion, 
                                                return_class=SpacedDiffusionQuestDiff, 
                                                device=self.device,
                                                eval=True)

        self.decoder = ModelWrapper(decoder, 
                                    self.diffusion_train, 
                                    self.diffusion_eval, 
                                    self.device)

        # self.decoder = EncDecPerLandmark(**decoder_cfg)
        
        self.viz_dir = viz_dir
        self.validation_outputs = []
        self.show_results = dict(images=[], preds=[], targets=[])

    def forward(self, x, batch):
        features = self.backbone(x)
        batch["features"] = features.clone()
        batch["pos_embed"] = self.backbone.pos_embed.clone()
        pred = self.decoder(batch)
        return pred

    def training_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        batch['repr_clean'] = target['joints'].clone().float()
        target_weights = target['target_weight'].to(self.device)

        pred = self(images, batch)
        target_weights = target_weights * pred['weights'].unsqueeze(-1).unsqueeze(-1)
        
        # Compute losses
        loss = self.criterion(pred, target, target_weights)
        
        steps = 10 if self.cfg.debug_mode else 2000
        train_path = os.path.join(self.viz_dir, 'train')
        os.makedirs(train_path, exist_ok=True)
        if self.global_step > 0 and self.global_step % steps == 0:
            with torch.no_grad():
                video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, train_path)
                self.wandb_video_log(video_path, 'train')
        if self.global_step > 0 and self.global_step % self.cfg.log_steps == 0:
            self.tensorboard_logging(loss, self.global_step, train=True)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        lr_backbone = optimizer.param_groups[-1]["lr"]
        
        self.log('train/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=images.size(0))
        self.log('train/lr_backbone', lr_backbone, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation step")
        if self.val_data_cfg['type'] == 'EMDB':
            target = batch
            images = target['image'].to(self.device)
            
            batch['repr_clean'] = torch.randn_like(target['joints'])
            target_weights = target['target_weight'].to(self.device)

            pred = self(images, batch)

            loss = self.criterion(pred, target, target_weights)

            self.tensorboard_logging(loss, self.global_step, train=False,)
            self.log('val/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True, batch_size=images.size(0))  # NOTE: , sync_dist=True MAYBE?
            self.log('val/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
            self.log('val/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
            
            self.show_results["images"].append(images[:1])
            self.show_results["preds"].append(pred["joints2d"][:1])
            self.show_results["targets"].append(target["joints"][:1])
            return loss

        val_output = {'val_loss': 0.0}  # Modified to store outputs
        self.validation_outputs.append(val_output)  # Store the output

        return val_output

    def on_validation_epoch_end(self):
        val_path = os.path.join(self.viz_dir, 'val')
        os.makedirs(val_path, exist_ok=True)
        with torch.no_grad():
            # images, pred, target = self.show_results
            images = torch.cat(self.show_results["images"])
            pred = {"joints2d": torch.cat(self.show_results["preds"])}
            target = {"joints": torch.cat(self.show_results["targets"])}
            video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, val_path)
            self.wandb_video_log(video_path, 'val')
        self.validation_outputs.clear()
        self.show_results = dict(images=[], preds=[], targets=[])