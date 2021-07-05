import torch
import sys
import math


class TransTRunner:
    def __init__(self, model, criterion, optimizer, lr_scheduler,
                 grad_max_norm=None,
                 stage_2_data_processor=None,
                 additional_stateful_objects=None, begin_training_event_slots=None, stop_training_event_slot=None,
                 epoch_changed_event_slots=None, statistics_collectors=None, multi_stage_handlers=None,
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_max_norm = grad_max_norm
        self.stage_2_data_processor = stage_2_data_processor
        self.epoch = 0
        self.number_of_samples = 0
        self.additional_stateful_objects = additional_stateful_objects
        self.begin_training_event_slots = begin_training_event_slots
        self.stop_training_event_slot = stop_training_event_slot
        self.epoch_changed_event_slots = epoch_changed_event_slots
        self.statistics_collectors = statistics_collectors
        self.multi_stage_handlers = multi_stage_handlers

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.begin_training_event_slots is not None:
            for slot in self.begin_training_event_slots:
                slot.start()

    def stop(self):
        if self.stop_training_event_slot is not None:
            for slot in self.stop_training_event_slot:
                slot.stop()

    def forward(self, samples, targets, miscellanies_host, miscellanies_device, _):
        if self.stage_2_data_processor is not None:
            samples = self.stage_2_data_processor(samples, miscellanies_host, miscellanies_device)
        outputs = self.model(*samples)
        loss, loss_value, loss_stats = self.criterion(outputs, targets)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_stats)
            sys.exit(1)

        self.loss = loss

        positive_samples = miscellanies_host['is_positive_sample']

        return {'loss': loss_value, **loss_stats, 'pos_samples_ratio': torch.sum(positive_samples) / len(positive_samples)}

    def _on_epoch_changed(self):
        if self.epoch_changed_event_slots is not None:
            for event_slot in self.epoch_changed_event_slots:
                event_slot.set_epoch(self.epoch)

    def move_to_next_epoch(self):
        if self.statistics_collectors is not None:
            print(f'Epoch {self.epoch} statistics:')
            for statistics_collector_name, statistics_collector in self.statistics_collectors.items():
                print('----------------------------')
                print(f'{statistics_collector_name}:')
                print(statistics_collector.get_status())
                print('----------------------------')
        self.epoch += 1
        self.lr_scheduler.step()
        self._on_epoch_changed()
        if self.multi_stage_handlers is not None:
            for handler in self.multi_stage_handlers:
                handler.on_epoch_changed(self.epoch, self)

    def get_epoch(self):
        return self.epoch

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        if self.grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
        self.optimizer.step()
        del self.loss
        self.loss = None
        return {'lr': self.optimizer.param_groups[0]["lr"]}

    def state_dict(self):
        training_state_dict = {'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': self.epoch, 'version': 1}
        if self.additional_stateful_objects is not None:
            for state_name, stateful_object in self.additional_stateful_objects.items():
                training_state_dict[state_name] = stateful_object.get_state()
        return {'model': self.get_model().state_dict(), 'version': 1}, training_state_dict

    def load_state_dict(self, model_state, training_state):
        assert model_state['version'] == 1
        self.get_model().load_state_dict(model_state['model'])
        if training_state is not None:
            self.optimizer.load_state_dict(training_state['optimizer'])
            self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
            self.epoch = training_state['epoch']
            if self.additional_stateful_objects is not None:
                for state_name, stateful_object in self.additional_stateful_objects.items():
                    stateful_object.set_state(training_state[state_name])
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
