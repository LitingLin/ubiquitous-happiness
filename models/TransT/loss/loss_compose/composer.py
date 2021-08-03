import torch

from miscellanies.torch.distributed import is_dist_available_and_initialized
from miscellanies.torch.distributed.reduce_mean import reduce_mean_


class LinearWeightScheduler:
    def __init__(self, init_weight, ultimate_weight, begin_step, end_step, per_iter):
        self.init_weight = init_weight
        self.ultimate_weight = ultimate_weight
        self.begin_step = begin_step
        self.end_step = end_step
        self.weight_step_size = (ultimate_weight - init_weight) / (end_step - begin_step)
        self.per_iter = per_iter
        self.position = 0

    def get_state(self):
        return self.position

    def set_state(self, state):
        self.position = state

    def forward(self, loss):
        if self.position < self.begin_step:
            weight = self.init_weight
        elif self.position > self.end_step:
            weight = self.ultimate_weight
        else:
            weight = self.init_weight + (self.position - self.begin_step) * self.weight_step_size
        return loss * weight, loss.detach(), weight

    def on_next_iter(self):
        if self.per_iter:
            self.position += 1

    def on_next_epoch(self):
        if not self.per_iter:
            self.position += 1


class ConstantWeightScheduler:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, loss):
        return loss * self.weight, loss.detach(), self.weight


class LossComposer:
    def __init__(self, loss_weight_schedulers, display_names, display_prefix, display_postfix):
        self.loss_weight_schedulers = loss_weight_schedulers
        self.display_names = display_names
        self.display_prefix = display_prefix
        self.display_postfix = display_postfix

    def __call__(self, losses):
        loss_list = []
        detached_loss_list = []
        weight_list = []
        for index, loss_weight_scheduler in enumerate(self.loss_weight_schedulers):
            loss, loss_detached, weight = loss_weight_scheduler.forward(losses[index])
            loss_list.append(loss)
            detached_loss_list.append(loss_detached)
            weight_list.append(weight)

        weighted_loss = sum(loss_list)
        if is_dist_available_and_initialized():
            detached_loss_list = torch.stack(detached_loss_list, dim=0)
            reduce_mean_(detached_loss_list)
            detached_loss_list = detached_loss_list.cpu()

        loss_stats_dict = {}
        unscaled_loss_stats_dict = {}

        for detached_loss, weight, display_name in zip(detached_loss_list, weight_list, self.display_names):
            detached_loss = detached_loss.cpu().item()
            loss_stats_dict[display_name] = detached_loss * weight
            unscaled_loss_stats_dict[display_name + '_unscaled'] = detached_loss

        loss_value = sum(loss_stats_dict.values())
        loss_stats_dict.update(unscaled_loss_stats_dict)

        return weighted_loss, loss_value, loss_stats_dict

    def get_state(self):
        state = []
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'get_state'):
                state.append(loss_weight_scheduler.get_state())
        return state

    def set_state(self, state):
        pos = 0
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'set_state'):
                loss_weight_scheduler.set_state(state[pos])
                pos += 1

    def on_next_iter(self):
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'on_next_iter'):
                loss_weight_scheduler.on_next_iter()

    def on_next_epoch(self):
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'on_next_epoch'):
                loss_weight_scheduler.on_next_epoch()
