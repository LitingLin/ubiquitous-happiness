from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QListWidget, QListWidgetItem, QGridLayout
from PyQt5.QtCore import QTimer, Qt, pyqtSlot, QObject, QPoint, QPointF, QRect, QRectF, QSize
from PyQt5.QtGui import QResizeEvent, QPixmap, QPainter, QPen, QPolygon, QPolygonF, QBrush, QColor
from typing import List

# https://stackoverflow.com/questions/8211982/qt-resizing-a-qlabel-containing-a-qpixmap-while-keeping-its-aspect-ratio
class _QLabel_auto_scaled_pixmap(QLabel):
    def __init__(self, *args):
        super(_QLabel_auto_scaled_pixmap, self).__init__(*args)
        self.setMinimumSize(1,1)
        self.setScaledContents(False)
        self.unscaled_pixmap = None

    def setPixmap(self, qPixmap: QPixmap):
        self.unscaled_pixmap = qPixmap
        super(_QLabel_auto_scaled_pixmap, self).setPixmap(self._get_scaled_pixmap())

    def resizeEvent(self, qResizeEvent: QResizeEvent):
        super().resizeEvent(qResizeEvent)
        if self.unscaled_pixmap is not None:
            super(_QLabel_auto_scaled_pixmap, self).setPixmap(self._get_scaled_pixmap())

    def _get_scaled_pixmap(self):
        width = self.width()
        height = self.height()
        return self.unscaled_pixmap.scaled(width, height, Qt.KeepAspectRatio)

    def sizeHint(self):
        w = self.width()
        if self.unscaled_pixmap is None:
            h = self.height()
        else:
            h = (float(self.unscaled_pixmap.height()) * w) / self.unscaled_pixmap.width()
        return QSize(w, h)


def _create_canvas(parent_layout, n_vertical_canvas, n_horizontal_canvas):
    canvas_layout = QGridLayout()

    canvases = []

    for i_row in range(n_vertical_canvas):
        for i_col in range(n_horizontal_canvas):
            canvas = _QLabel_auto_scaled_pixmap()
            size_policy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            size_policy.setHorizontalStretch(1)
            size_policy.setVerticalStretch(1)
            canvas.setSizePolicy(size_policy)
            canvas_layout.addWidget(canvas, i_row, i_col)
            canvases.append(canvas)

    parent_layout.addLayout(canvas_layout)

    return canvases


class _Timer:
    def __init__(self, parent):
        timer = QTimer(parent)
        timer.timeout.connect(self._on_timeout)
        self.timer = timer
        self.callback = None

    def set_interval(self, msec: int):
        self.timer.setInterval(msec)

    def set_callback(self, callback):
        self.callback = callback

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def _on_timeout(self):
        self.callback()


class _CanvasPainter:
    def __init__(self, image: QPixmap):
        self.painter = QPainter(image)

    def __enter__(self):
        return self

    def draw_point(self, x, y):
        self.painter.drawPoint(QPoint(x, y))

    def set_pen(self, pen: QPen):
        self.painter.setPen(pen)

    def draw_label(self, text: str, position: List[int], color: QColor):
        assert len(position) == 2
        fontMetrics = self.painter.fontMetrics()
        rendered_object_info_size = fontMetrics.tightBoundingRect(text)
        self.painter.fillRect(QRectF(position[0], position[1] - rendered_object_info_size.height(),
                                     rendered_object_info_size.width(), rendered_object_info_size.height()),
                              QBrush(color))
        pen = self.painter.pen()
        self.painter.setPen(QColor(255 - color.red(), 255 - color.green(), 255 - color.blue()))
        self.painter.drawText(QPointF(position[0], position[1]), text)
        self.painter.setPen(pen)

    def draw_polygon(self, polygon: List):
        self.painter.drawPolygon(_CanvasPainter._list_to_polygon(polygon))

    def draw_rect(self, rect: List):
        assert len(rect) == 4
        self.painter.drawRect(_CanvasPainter._list_to_rect(rect))

    @staticmethod
    def _list_to_rect(rect: List):
        if all([isinstance(v, int) for v in rect]):
            return QRect(rect[0], rect[1], rect[2], rect[3])
        else:
            return QRectF(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))

    @staticmethod
    def _list_to_polygon(polygon: List):
        assert len(polygon) % 2 == 0
        pointList = []
        if all([isinstance(v, int) for v in polygon]):
            for index, value in polygon:
                if index % 2 == 0:
                    x = value
                else:
                    y = value
                    pointList.append(QPoint(x, y))
        else:
            for index, value in polygon:
                if index % 2 == 0:
                    x = value
                else:
                    y = value
                    pointList.append(QPointF(float(x), float(y)))

        return pointList

    def end_draw(self):
        self.painter.end()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.painter.end()


class _Canvas:
    def __init__(self, label):
        self.label = label
        self.image = None

    def set_background(self, qPixmap: QPixmap):
        self.image = qPixmap

    def create_empty(self, width: int, height: int):
        self.image = QPixmap(width, height)

    def get_painter(self):
        assert self.image is not None, "create canvas first"
        return _CanvasPainter(self.image)

    def update(self):
        self.label.setPixmap(self.image)
        self.label.fitPixmapToWidgetSize()


class _LayoutWidgetCreator:
    def __init__(self, layout):
        self.layout = layout

    def new_label(self, text: str):
        label = QLabel()
        if text is not None:
            label.setText(text)
        self.layout.addWidget(label)
        return label

    def new_button(self, text: str, callback):
        button = QPushButton()
        if text is not None:
            button.setText(text)
        if callback is not None:
            button.clicked.connect(callback)
        self.layout.addWidget(button)
        return button

    def new_list(self, string_list, callback):
        listWidget = QListWidget()
        listWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        if string_list is not None:
            for string in string_list:
                QListWidgetItem(string, listWidget)
        if callback is not None:
            listWidget.currentRowChanged.connect(callback)
        self.layout.addWidget(listWidget)
        return listWidget


class Qt5Viewer:
    def __init__(self, argv=[], n_vertical_canvas=1, n_horizontal_canvas=1):
        app = QApplication(argv)

        window = QDialog()
        window.setWindowState(Qt.WindowMaximized)
        window.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        window.setWindowTitle('Viewer')

        main_layout = QVBoxLayout()
        window.setLayout(main_layout)
        content_widget_layout = QHBoxLayout()

        self.canvas_labels = _create_canvas(content_widget_layout, n_vertical_canvas, n_horizontal_canvas)
        self.n_vertical_canvas = n_vertical_canvas
        self.n_horizontal_canvas = n_horizontal_canvas

        main_layout.addLayout(content_widget_layout)

        control_widget_layout = QVBoxLayout()
        content_widget_layout.addLayout(control_widget_layout)

        self.content_widget_layout = content_widget_layout
        self.control_widget_layout = control_widget_layout

        self.window = window
        self.app = app

    def set_title(self, title: str):
        self.window.setWindowTitle(title)

    def run_event_loop(self):
        self.window.show()
        return self.app.exec_()

    def new_timer(self):
        return _Timer(self.window)

    def get_canvas(self, index_of_vertical=0, index_of_horizontal=0):
        index = index_of_vertical * self.n_horizontal_canvas + index_of_horizontal
        return _Canvas(self.canvas_labels[index])

    def close(self):
        self.window.close()

    def get_content_region(self):
        return _LayoutWidgetCreator(self.content_widget_layout)

    def get_control_region(self):
        return _LayoutWidgetCreator(self.control_widget_layout)
