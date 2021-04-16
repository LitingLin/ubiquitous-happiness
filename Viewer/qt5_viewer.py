from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QListWidget, QListWidgetItem
from PyQt5.QtCore import QTimer, Qt, pyqtSlot, QObject, QPoint, QPointF, QRect, QRectF
from PyQt5.QtGui import QResizeEvent, QPixmap, QPainter, QPen, QPolygon, QPolygonF, QBrush, QColor
from typing import List


class _QLabel_autoFitToWidgetSize(QLabel):
    def resizeEvent(self, qResizeEvent: QResizeEvent):
        super().resizeEvent(qResizeEvent)
        self.fitPixmapToWidgetSize()

    def fitPixmapToWidgetSize(self):
        p = self.pixmap()
        if p is not None:
            width = self.width()
            height = self.height()
            self.setPixmap(p.scaled(width, height, Qt.KeepAspectRatio))


class _QtPainter:
    def __init__(self, label: _QLabel_autoFitToWidgetSize, image: QPixmap=None, width: int=None, height: int=None):
        self.label = label
        if image is not None:
            self.image = image
        elif width is not None and height is not None:
            self.image = QPixmap(width, height)

    def setCanvas(self, image: QPixmap):
        self.image = image

    def setEmptyCanvas(self, width: int, height: int):
        self.image = QPixmap(width, height)

    def __enter__(self):
        assert self.image is not None, "set canvas first"
        self.painter = QPainter(self.image)

    def drawPoint(self, x, y):
        self.painter.drawPoint(QPoint(x, y))

    def setPen(self, pen: QPen):
        self.painter.setPen(pen)

    def drawLabel(self, text: str, position: List[int], color: QColor):
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

    def drawBoundingBox(self, bounding_box):
        if len(bounding_box) == 4:
            self.drawRect(bounding_box)
        else:
            self.drawPolygon(bounding_box)

    def drawPolygon(self, polygon: List):
        self.painter.drawPolygon(_QtPainter._listToPolygon(polygon))

    def drawRect(self, rect: List):
        assert len(rect) == 4
        self.painter.drawRect(_QtPainter._listToRect(rect))

    @staticmethod
    def _listToRect(rect: List):
        if all([isinstance(v, int) for v in rect]):
            return QRect(rect[0], rect[1], rect[2], rect[3])
        else:
            return QRectF(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))

    @staticmethod
    def _listToPolygon(polygon: List):
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.painter.end()

    def update(self):
        assert self.image is not None, 'you need to set canvas'
        self.label.setPixmap(self.image)
        self.label.fitPixmapToWidgetSize()


class Qt5Viewer:
    def __init__(self, argv=[]):
        app = QApplication(argv)

        window = QDialog()
        window.setWindowState(Qt.WindowMaximized)
        window.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        window.setWindowTitle('Viewer')

        mainLayout = QVBoxLayout()
        window.setLayout(mainLayout)
        contentLayout = QHBoxLayout()

        canvasLabel = _QLabel_autoFitToWidgetSize()
        canvasLabel.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        contentLayout.addWidget(canvasLabel)

        mainLayout.addLayout(contentLayout)

        customLayout = QVBoxLayout()
        contentLayout.addLayout(customLayout)

        self.canvasLabel = canvasLabel
        self.contentLayout = contentLayout
        self.app = app
        self.window = window
        self.customLayout = customLayout
        self.stuffs = []
        self.timer = QTimer()
        self.timer.timeout.connect(self._onTimerTimeOut)

    def setWindowTitle(self, title: str):
        self.window.setWindowTitle(title)

    def runEventLoop(self):
        self.window.show()
        return self.app.exec_()

    def startTimer(self):
        self.timer.start()

    def stopTimer(self):
        self.timer.stop()

    def setTimerInterval(self, msec: int):
        self.timer.setInterval(msec)

    def setTimerCallback(self, callback):
        self.callback = callback

    def addButton(self, text=None, callback=None):
        button = QPushButton()
        if text is not None:
            button.setText(text)
        if callback is not None:
            button.clicked.connect(callback)
        self.customLayout.addWidget(button)
        return button

    def addList(self, list:List[str]=None, callback=None):
        listWidget = QListWidget()
        listWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        if list is not None:
            for v in list:
                QListWidgetItem(v, listWidget)
        if callback is not None:
            class Proxy:
                def __init__(self, listWidget: QListWidget, callback):
                    self.listWidget = listWidget
                    self.callback = callback

                def setSelectionChangedCallback(self, callback):
                    self.callback = callback

                def onCallback(self):
                    indices = self.listWidget.selectedIndexes()
                    if len(indices) == 0:
                        return

                    self.callback(indices[0].row())
            proxy = Proxy(listWidget, callback)
            listWidget.itemSelectionChanged.connect(proxy.onCallback)
            self.stuffs.append(proxy)
        self.contentLayout.addWidget(listWidget)

    def getPainter(self, canvas: QPixmap = None, width: int=None, height: int=None):
        return _QtPainter(self.canvasLabel, canvas, width, height)

    def newCanvas(self, canvas: QPixmap = None, width: int=None, height: int=None):
        canvasLabel = _QLabel_autoFitToWidgetSize()
        canvasLabel.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.contentLayout.addWidget(canvasLabel)
        return _QtPainter(canvasLabel, canvas, width, height)

    def _onTimerTimeOut(self):
        self.callback()

    def close(self):
        self.window.close()

    def addLabel(self, text: str):
        label = QLabel()
        label.setText(text)
        self.customLayout.addWidget(label)
        return label
