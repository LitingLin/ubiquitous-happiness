from Dataset.Detection.Base.SingleObject.dataset import SingleObjectDetectionDataset
from Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
import random


class SingleObjectDetectionDatasetQt5Viewer:
    def __init__(self, dataset: SingleObjectDetectionDataset):
        self.dataset = dataset
        self.viewer = Qt5Viewer()

        self.category_name_color_map = {}
        for category_name in self.dataset.getCategoryNameList():
            color = [random.randint(0, 255) for _ in range(3)]
            self.category_name_color_map[category_name] = QColor(color[0], color[1], color[2], int(0.5 * 255))

        image_names = []
        if isinstance(dataset, SingleObjectDetectionDataset):
            for image in self.dataset:
                image_names.append(image.getName())
        else:
            for index in range(len(self.dataset)):
                image_names.append(str(index))

        self.viewer.addList(image_names, self._sequenceSelectedCallback)

    def _sequenceSelectedCallback(self, index: int):
        image = self.dataset[index]
        pixmap = QPixmap()
        assert pixmap.load(image.getImagePath())
        painter = self.viewer.getPainter(pixmap)
        with painter:
            category_name = image.getCategoryName()
            bounding_box = image.getBoundingBox()
            painter.setPen(self.category_name_color_map[category_name])
            painter.drawBoundingBox(bounding_box)
            painter.drawLabel(category_name, (bounding_box[0], bounding_box[1]), self.category_name_color_map[category_name])
        painter.update()

    def run(self):
        return self.viewer.runEventLoop()
