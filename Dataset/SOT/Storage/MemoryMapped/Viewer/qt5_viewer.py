from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped, SingleObjectTrackingDatasetSequence_MemoryMapped
from Dataset.Base.Common.Viewer.qt5_viewer import draw_object
from Miscellaneous.Viewer.old_qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
import random
from Miscellaneous.simple_prefetcher import SimplePrefetcher


__all__ = ['SOTDatasetQt5Viewer']


class DatasetSequenceImageLoader:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence[index]
        pixmap = QPixmap()
        assert pixmap.load(frame.get_image_path())
        return pixmap, frame


class SOTDatasetQt5Viewer:
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped):
        self.dataset = dataset
        self.viewer = Qt5Viewer()
        self.viewer.setWindowTitle(self.dataset.get_name())

        if self.dataset.has_category_id_name_map():
            self.category_id_color_map = {}
            for category_id in self.dataset.get_category_id_name_map().keys():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_id_color_map[category_id] = QColor(color[0], color[1], color[2], int(0.5 * 255))
        else:
            self.category_id_color_map = None

        sequence_names = []
        for sequence in self.dataset:
            sequence_names.append(sequence.get_name())

        self.viewer.addList(sequence_names, self._sequenceSelectedCallback)
        self.viewer.setTimerCallback(self._timerTimeoutCallback)

    def _sequenceSelectedCallback(self, index: int):
        self.sequence = SimplePrefetcher(DatasetSequenceImageLoader(self.dataset[index]))
        self._stopTimer()
        self._startTimer()

    def _startTimer(self):
        self.sequence_iter = iter(self.sequence)
        self.viewer.startTimer()

    def _timerTimeoutCallback(self):
        try:
            image, frame = next(self.sequence_iter)
        except StopIteration:
            self._stopTimer()
            return
        painter = self.viewer.getPainter(image)

        with painter:
            draw_object(painter, frame, frame, self.sequence.iterable.sequence, None, self.category_id_color_map, self.dataset, self.dataset)
        painter.update()

    def _stopTimer(self):
        self.viewer.stopTimer()

    def run(self):
        return self.viewer.runEventLoop()
