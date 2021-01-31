from Dataset.Base.Image.dataset import ImageDataset
from Dataset.Base.Common.Viewer.qt5_viewer import draw_object
from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
import random


class ImageDatasetQt5Viewer:
    def __init__(self, dataset: ImageDataset):
        self.dataset = dataset
        self.viewer = Qt5Viewer()

        if self.dataset.has_category_id_name_map():
            self.category_id_color_map = {}
            for category_id in self.dataset.get_category_id_name_map().keys():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_id_color_map[category_id] = QColor(color[0], color[1], color[2], int(0.5 * 255))
        else:
            self.category_id_color_map = None

        image_names = []
        for index in range(len(self.dataset)):
            image_names.append(str(index))

        self.viewer.addList(image_names, self._imageSelectedCallback)

    def _imageSelectedCallback(self, index: int):
        image = self.dataset[index]
        pixmap = QPixmap()
        assert pixmap.load(image.get_image_path())
        painter = self.viewer.getPainter(pixmap)
        if len(image) > 0:
            with painter:
                for object_ in image:
                    draw_object(painter, object_, object_, None, None, self.category_id_color_map,
                                self.dataset)
        painter.update()

    def run(self):
        return self.viewer.runEventLoop()