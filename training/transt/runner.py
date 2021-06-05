import torch
import sys
import math


class TransTRunner:
    def __init__(self, model, criterion, optimizer, lr_scheduler, epoch_changed_event_slots=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        self.epoch_changed_event_slots = epoch_changed_event_slots
        self.number_of_samples = 0

    def forward(self, samples, targets, is_positives):
        outputs = self.model(samples)
        loss, loss_value, loss_stats_reduced_unscaled, loss_stats_reduced_scaled = self.criterion(outputs, targets)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_stats_reduced_unscaled)
            sys.exit(1)

        self.loss = loss

        return {'loss': loss_value, **loss_stats_reduced_unscaled, **loss_stats_reduced_scaled, 'pos_samples_ratio': torch.sum(is_positives) / len(is_positives)}

    def _on_epoch_changed(self):
        if self.epoch_changed_event_slots is not None:
            for event_slot in self.epoch_changed_event_slots:
                event_slot.set_epoch(self.epoch)

    def move_next_epoch(self):
        self.epoch += 1
        self.lr_scheduler.step()
        self._on_epoch_changed()

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
        return {'model': self.get_model().state_dict(), 'version': 1}, \
               {'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': self.epoch, 'version': 1}

    def load_state_dict(self, model_state, training_state):
        assert model_state['version'] == 1
        self.get_model().load_state_dict(model_state['model'])
        if training_state is not None:
            self.optimizer.load_state_dict(training_state['optimizer'])
            self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
            self.epoch = training_state['epoch']
            self._on_epoch_changed()

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
