from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from Dataset.MOT.Base.sequence import MultipleObjectTrackingDatasetSequenceView
from Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor, QPen
from PyQt5.QtCore import Qt
from Utils.simple_prefetcher import SimplePrefetcher
import random


class DatasetSequenceImageLoader:
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequenceView):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence.getFrame(index)
        pixmap = QPixmap()
        assert pixmap.load(frame.getImagePath())
        return pixmap, frame


class MOTDatasetViewer:
    def __init__(self, dataset: MultipleObjectTrackingDataset):
        self.dataset = dataset
        self.viewer = Qt5Viewer()

        self.category_name_color_map = {}
        for category_name in self.dataset.category_names:
            color = [random.randint(0, 255) for _ in range(3)]
            self.category_name_color_map[category_name] = QColor(color[0], color[1], color[2], int(0.5 * 255))

        sequence_names = []
        for sequence in self.dataset:
            sequence_names.append(sequence.getName())

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
            for object_ in frame:
                category_name = object_.getCategoryName()
                bounding_box = object_.getBoundingBox()
                if bounding_box is not None:
                    is_present = object_.getAttributeIsPresent()
                    pen = QPen(self.category_name_color_map[category_name])
                    if is_present is False:
                        pen.setStyle(Qt.DashDotDotLine)
                    painter.setPen(pen)
                    painter.drawBoundingBox(bounding_box)
                    object_id = object_.getObjectId()
                    painter.drawLabel('{}-{}'.format(object_id, category_name), (bounding_box[0], bounding_box[1]), self.category_name_color_map[category_name])
        painter.update()

    def _stopTimer(self):
        self.viewer.stopTimer()

    def run(self):
        return self.viewer.runEventLoop()
