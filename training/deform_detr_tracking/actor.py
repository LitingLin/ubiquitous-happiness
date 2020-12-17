import torch
import sys
import math


class DETRTrackingActor:
    def __init__(self, model, criterion, optimizer, lr_scheduler, distributed_samplers=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        self.distributed_samplers = distributed_samplers

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, samples, targets):
        outputs = self.model(samples)
        loss, loss_stats_reduced_unscaled, loss_stats_reduced_scaled = self.criterion(outputs, targets)

        losses_reduced_scaled = sum(loss_stats_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_stats_reduced_unscaled)
            sys.exit(1)

        self.loss = loss
        return {'loss': loss_value, **loss_stats_reduced_unscaled, **loss_stats_reduced_scaled}

    def new_epoch(self):
        self.epoch += 1
        self.lr_scheduler.step()
        if self.distributed_samplers is not None:
            for distributed_sampler in self.distributed_samplers:
                distributed_sampler.set_epoch(self.epoch)

    def get_epoch(self):
        return self.epoch

    def backward(self, max_norm):
        self.optimizer.zero_grad()
        self.loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()
        del self.loss
        self.loss = None
        return {'lr': self.optimizer.param_groups[0]["lr"]}

    def state_dict(self):
        return {'model': self.get_model().state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': self.epoch}

    def load_state_dict(self, state, model_only=False):
        self.get_model().load_state_dict(state['model'])
        if not model_only:
            self.optimizer.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            self.epoch = state['epoch']

            if self.distributed_samplers is not None:
                for distributed_sampler in self.distributed_samplers:
                    distributed_sampler.set_epoch(self.epoch)

    def to(self, device):
        self.model.to(device)
        self.criterion.to(device)

    def train(self):
        self.model.train()
        self.criterion.train()

    def eval(self):
        self.model.eval()
        self.criterion.eval()

    def get_model(self):
        if not isinstance(self.model, torch.nn.Module):
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model
        return model_without_ddp

    def n_parameters(self):
        n_parameters = sum(p.numel() for p in self.get_model().parameters() if p.requires_grad)
        return n_parameters
