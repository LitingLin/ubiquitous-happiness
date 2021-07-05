from miscellanies.Viewer.qt5_viewer import Qt5Viewer
from miscellanies.simple_prefetcher import SimplePrefetcher


class _DataWrapper:
    def __init__(self, data_loader, stage_2_data_processor, visualizer_data_adaptor):
        self.data_loader = data_loader
        self.stage_2_data_processor = stage_2_data_processor
        self.data_adaptor = visualizer_data_adaptor

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        samplers, targets, miscellanies_host, miscellanies_device, miscellanies_element = next(self.data_loader_iter)
        if self.stage_2_data_processor is not None:
            samplers = self.stage_2_data_processor(samplers, miscellanies_host, miscellanies_device)
            miscellanies_device = None
        return self.data_adaptor.on_data((samplers, targets, miscellanies_host, miscellanies_device, miscellanies_element))


class _VisualizerDataPrefetcher:
    def __init__(self, data_loader, enable_prefetch = True):
        self.data_loader = data_loader
        self.enable_prefetch = enable_prefetch
        self._begin()

    def _begin(self):
        data_loader = self.data_loader
        if self.enable_prefetch:
            data_loader = SimplePrefetcher(data_loader)
        self.data_loader_iter = iter(data_loader)

    def get_next(self):
        try:
            data = next(self.data_loader_iter)
        except StopIteration:
            self._begin()
            data = next(self.data_loader_iter)
        return data


class DataPreprocessingVisualizer:
    def __init__(self, data_loader, stage_2_data_processor, visualizer_data_adaptor):
        self.visualizer_data_adaptor = visualizer_data_adaptor
        self.viewer: Qt5Viewer = visualizer_data_adaptor.on_create()
        self.timer = self.viewer.new_timer()
        self.timer.set_callback(self._on_timer_timeout)
        self.viewer.get_control_region().new_integer_spin_box('interval(ms): ', 1, 1000, int(1000 / 30), self._set_new_timer_interval)
        self.viewer.get_control_region().new_button('start', self._start_button_clicked)
        self.viewer.get_control_region().new_button('stop', self._stop_button_clicked)
        self.viewer.get_control_region().new_button('step', self._on_timer_timeout)

        self.data_loader = _VisualizerDataPrefetcher(_DataWrapper(data_loader, stage_2_data_processor, visualizer_data_adaptor))

    def _set_new_timer_interval(self, interval: int):
        self.timer.set_interval(interval)

    def _start_button_clicked(self):
        self.timer.start()

    def _stop_button_clicked(self):
        self.timer.stop()

    def _on_timer_timeout(self):
        data = self.data_loader.get_next()
        self.visualizer_data_adaptor.on_draw(data)

    def run(self):
        return self.viewer.run_event_loop()
