import time
from fvcore.nn import FlopCountAnalysis, flop_count_table


class TrackerEfficiencyAssessor:
    def __init__(self, model, pseudo_data_source, batch):
        self.model = model
        self.pseudo_data_source = pseudo_data_source
        self.batch = batch

    def _test_fps(self, batch):
        is_train = self.model.training
        if is_train:
            self.model.eval()
        for _ in range(3):
            z, x = self.pseudo_data_source.get_train(batch)
            init_begin_time = time.perf_counter()
            z_feat = self.model.template(z)
            track_begin_time = time.perf_counter()
            self.model.track(z_feat, x)
            track_end_time = time.perf_counter()

        if is_train:
            self.model.train()

        return batch / (track_begin_time - init_begin_time), batch / (track_end_time - track_begin_time)

    def get_batch(self):
        return self.batch

    def test_fps(self):
        return self._test_fps(1)

    def test_fps_batched(self):
        return self._test_fps(self.batch)

    def get_flop_count_table(self):
        return flop_count_table(FlopCountAnalysis(self.model, self.pseudo_data_source.get_train(1)))
