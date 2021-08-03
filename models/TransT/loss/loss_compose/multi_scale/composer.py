def _sum_compose(losses):
    for loss, loss_value, loss_dict


class MultiScaleLossComposer:
    def __init__(self, compose_fn, single_scale_loss_composers):
        self.single_scale_loss_composers = single_scale_loss_composers
        self.compose_fn = compose_fn

    def __call__(self, losses):
        return self.compose_fn(losses)

    def get_state(self):
        state = []
        for single_scale_loss_composer in self.single_scale_loss_composers:
            state.append(single_scale_loss_composer.get_state())
        return state

    def set_state(self, states):
        for state, single_scale_loss_composer in zip(states, self.single_scale_loss_composers):
            single_scale_loss_composer.set_state(state)

    def on_next_iter(self):
        for single_scale_loss_composer in self.single_scale_loss_composers:
            single_scale_loss_composer.on_next_iter()

    def on_next_epoch(self):
        for single_scale_loss_composer in self.single_scale_loss_composers:
            single_scale_loss_composer.on_next_epoch()
