import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from llib.datasets.training.BEDLAM import BEDLAMDataset
from llib.datasets.validation.EMDB import EMDBDataset

class BaseTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseTrainer, self).__init__()

        # Default settings
        self.cfg = cfg
        self.optimizer_cfg = cfg.optimizer.optimizer
        self.lr_cfg = cfg.optimizer.lr_config
        self.train_data_cfg = cfg.data['train']
        self.val_data_cfg = cfg.data['val'] if 'val' in cfg.data else None

        self.backbone = None
        self.decoder = None
        self.integrator = None

    def configure_optimizers(self):
        param_groups = [
            {'params': self.decoder.parameters()},
            {'params': self.backbone.parameters(), 'lr': self.optimizer_cfg['lr_backbone']}
        ]

        if self.integrator is not None:
            param_groups.append(
                {'params': self.integrator.parameters()}
            )
        optimizer = eval(f'torch.optim.{self.optimizer_cfg["type"]}')(
            param_groups,
            lr=self.optimizer_cfg['lr'],
            betas=self.optimizer_cfg['betas'],
            weight_decay=self.optimizer_cfg['weight_decay']
        )

        # Learning rate scheduler
        if self.lr_cfg['warmup']:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=self.lr_cfg['warmup_ratio'], 
                end_factor=1.0, 
                total_iters=self.lr_cfg['warmup_iters']
            )

        if self.lr_cfg['policy'] == "step":
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=self.lr_cfg['milestones'], 
                gamma=self.lr_cfg['gamma'], 
            )
        elif self.lr_cfg['policy'] == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.lr_cfg['T_max'], 
                eta_min=self.lr_cfg['eta_min']
            )

        sequential = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            [warmup_scheduler, main_scheduler], 
            [self.lr_cfg['warmup_iters']]
        )
    
        return [optimizer], [{
            'scheduler': sequential,
            'interval': 'step',   # <-- batch-wise stepping
            'frequency': 1,
        }]

    def setup(self, stage=None):#prepare_data(self):
        if stage == 'fit' or stage is None:
            if self.train_data_cfg['type'] == 'BEDLAM':
                self.train_ds = BEDLAMDataset(**self.train_data_cfg)
                self.val_ds = EMDBDataset(**self.val_data_cfg)
                
            else:
                raise NotImplementedError(f"Dataset type {self.train_data_cfg['type']} not implemented yet!")

    def prepare_data(self):
        pass

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        if self.val_ds is None:
            return torch.utils.data.TensorDataset(torch.empty(0))
        else:
            return self.val_ds

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.samples_per_gpu,
            num_workers=self.cfg.workers_per_gpu,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        self.val_ds = self.val_dataset()
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=16,
            num_workers=min(self.cfg.workers_per_gpu, 2),
            pin_memory=True
        )

    def wandb_video_log(self, video_path, log_name='train') -> None:
        video = wandb.Video(video_path, format="mp4")
        if isinstance(self.loggers, WandbLogger):
            self.loggers.experiment.log({f"{log_name}/video": video})
        elif isinstance(self.loggers, list):
            for l in self.loggers:
                if isinstance(l, WandbLogger):
                    l.experiment.log({f"{log_name}/video": video})

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, losses, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)