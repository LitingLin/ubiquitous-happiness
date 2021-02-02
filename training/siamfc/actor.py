import torch
import sys
import math


class SiamFCTrainingActor:
    def __init__(self, model, criterion, optimizer, lr_scheduler, param_init_fn, epoch_changed_event_signal_slots=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        self.epoch_changed_event_signal_slots = epoch_changed_event_signal_slots
        self.param_init_fn = param_init_fn

    def reset_parameters(self):
        self.param_init_fn(self.get_model())

    def forward(self, samples, targets):
        outputs = self.model(samples)
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        self.loss = loss
        return {'loss': loss_value}

    def _emit_signal_epoch_changed(self):
        if self.epoch_changed_event_signal_slots is not None:
            for epoch_changed_signal_slot in self.epoch_changed_event_signal_slots:
                epoch_changed_signal_slot.set_epoch(self.epoch)

    def new_epoch(self):
        self.epoch += 1
        self.lr_scheduler.step()
        self._emit_signal_epoch_changed()

    def get_epoch(self):
        return self.epoch

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
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
            self._emit_signal_epoch_changed()

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
        if self.model.__class__.__name__ == 'DistributedDataParallel':
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model
        return model_without_ddp

    def n_parameters(self):
        n_parameters = sum(p.numel() for p in self.get_model().parameters() if p.requires_grad)
        return n_parameters
