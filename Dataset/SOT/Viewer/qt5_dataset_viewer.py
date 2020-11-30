from Dataset.SOT.Base.dataset import SingleObjectTrackingDataset
from Dataset.SOT.Base.sequence import SingleObjectTrackingDatasetSequenceViewer
from Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor, QPen
from PyQt5.QtCore import Qt
import random
from Utils.simple_prefetcher import SimplePrefetcher


class DatasetSequenceImageLoader:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequenceViewer):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence.getFrame(index)
        pixmap = QPixmap()
        assert pixmap.load(frame.getImagePath())
        return pixmap, frame


class SOTDatasetQt5Viewer:
    def __init__(self, dataset: SingleObjectTrackingDataset):
        self.dataset = dataset
        self.attribute_has_category = dataset.hasAttributeCategory()
        self.viewer = Qt5Viewer()
        self.viewer.setWindowTitle(self.dataset.getName())

        if self.attribute_has_category:
            self.category_name_color_map = {}
            for category_name in self.dataset.getCategoryNameList():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_name_color_map[category_name] = QColor(color[0], color[1], color[2], int(0.5 * 255))

        sequence_names = []

        if isinstance(dataset, SingleObjectTrackingDataset):
            for sequence in self.dataset:
                sequence_names.append(sequence.getName())
        else:
            for index in range(len(self.dataset)):
                sequence_names.append(str(index))

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
        if frame.hasAttributes():
            bounding_box = frame.getBoundingBox()
            if bounding_box is not None:
                if self.attribute_has_category:
                    category_name = self.sequence.iterable.sequence.getCategoryName()
                    color = self.category_name_color_map[category_name]
                else:
                    color = QColor(255, 0, 0, int(0.5 * 255))
                pen = QPen(color)
                is_present = frame.getAttributeIsPresent()
                if is_present is False:
                    pen.setStyle(Qt.DashDotDotLine)
                with painter:
                    painter.setPen(pen)
                    painter.drawBoundingBox(bounding_box)
                    if self.attribute_has_category:
                        painter.drawLabel(category_name, (bounding_box[0], bounding_box[1]), color)
        painter.update()

    def _stopTimer(self):
        self.viewer.stopTimer()

    def run(self):
        return self.viewer.runEventLoop()
