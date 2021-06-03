import torch
import sys
import math


class TransTRunner:
    def __init__(self, model, criterion, optimizer, lr_scheduler, data_loader_train, data_loader_val):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val

    def forward(self, samples, targets):
        outputs = self.model(samples)
        loss, loss_value, loss_stats_reduced_unscaled, loss_stats_reduced_scaled = self.criterion(outputs, targets)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_stats_reduced_unscaled)
            sys.exit(1)

        self.loss = loss
        return {'loss': loss_value, **loss_stats_reduced_unscaled, **loss_stats_reduced_scaled}

    def _synchronize_dataloader_state(self):
        self.data_loader_train.synchronize(self.epoch)
        self.data_loader_val.synchronize(self.epoch)

    def move_next_epoch(self):
        self.epoch += 1
        self.lr_scheduler.step()
        self._synchronize_dataloader_state()

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
                'train_data_loader': self.data_loader_train.get_state(),
                'val_data_loader': self.data_loader_val.get_state(),
                'epoch': self.epoch, 'version': 1}

    def load_state_dict(self, model_state, training_state):
        assert model_state['version'] == 1
        self.get_model().load_state_dict(model_state['model'])
        if training_state is not None:
            self.optimizer.load_state_dict(training_state['optimizer'])
            self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
            self.data_loader_train.load_state(training_state['train_data_loader'])
            self.data_loader_val.load_state(training_state['val_data_loader'])
            self.epoch = training_state['epoch']
            self._synchronize_dataloader_state()

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
