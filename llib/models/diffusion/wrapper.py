from copy import deepcopy
import torch
import numpy as np
from .resample import create_named_schedule_sampler


class ModelWrapper(torch.nn.Module):
    def __init__(self, decoder, diffusion_train=None, diffusion_eval=None, device=None, **kwargs):
        super().__init__()
        
        self.decoder = decoder
        self.diffusion_train = diffusion_train
        self.diffusion_eval = diffusion_eval

        self.configure_schedule_sampler()
        self.device = device


    def forward(self, batch):
        if self.training:
            return self.train_step(batch)
        else:
            return self.inference(batch)

    def configure_schedule_sampler(self, ):
        if self.diffusion_train is None: return
        
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion_train)

    def train_step(self, batch):
        t, weights = self.schedule_sampler.sample(batch['repr_clean'].shape[0], self.device)
        model_output = self.diffusion_train.training_step(model=self.decoder, batch=batch, t=t, noise=None)

        pred = {"joints2d": torch.cat((model_output["joints2d"], model_output["uncertainty"]), dim=-1), 
                "weights": weights}
        return pred

    def inference(self, batch):
        # TODO: Implement multiple predictions

        model_output = self.diffusion_eval.eval_step(model=self.decoder, batch=batch)
        pred = {"joints2d": model_output}
        return pred