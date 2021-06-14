import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PlotDrawer:
    def __init__(self, ):
        self.fig, self.ax = plt.subplots(1)

    def switch_backend(self, backend='Qt5Agg'):
        plt.switch_backend(backend)

    def drawImageWithBoundingBox(self, image, bounding_box, color=(1, 0, 0), linewidth=1, linestyle='solid'):
        self.drawImage(image)
        self.drawBoundingBox(bounding_box, color, linewidth, linestyle)

    def drawImageWithBoundingBoxAndLabel(self, image, bounding_box, label, color=(1, 0, 0), linewidth=1, linestyle='solid'):
        self.drawImage(image)
        self.drawBoundingBox(bounding_box, color, linewidth, linestyle)
        self.drawText(label, (bounding_box[0], bounding_box[1]), color, linestyle)

    def drawPoint(self, x_array, y_array, color=(1, 0, 0), size=10):
        self.ax.scatter(x_array, y_array, c=color, s=size)

    def waitKey(self):
        plt.waitforbuttonpress()

    def pause(self, value):
        plt.pause(value)

    def drawImage(self, image):
        self.ax.imshow(image)

    def drawBoundingBox(self, bounding_box, color=(1, 0, 0), linewidth=1, linestyle='solid'):
        self.ax.add_patch(
            patches.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2], bounding_box[3], linewidth=linewidth,
                              linestyle=linestyle,
                              edgecolor=color, facecolor='none'))
    # linestyle: ['solid'|'dashed'|'dashdot'|'dotted']
    def drawBoundingBoxAndLabel(self, bounding_box, label, color=(1, 0, 0), linewidth=1, linestyle='solid'):
        self.drawBoundingBox(bounding_box, color, linewidth, linestyle)
        self.drawText(label, (bounding_box[0], bounding_box[1]), color, linestyle)

    def drawText(self, string, position, color=(1, 0, 0), edgestyle='solid'):
        self.ax.text(position[0], position[1], string, horizontalalignment='left', verticalalignment='bottom',
                     bbox={'facecolor': color, 'alpha': 0.5, 'boxstyle': 'square,pad=0', 'linewidth': 0,
                           'linestyle': edgestyle})

    def update(self):
        self.fig.canvas.draw()

    def clear(self):
        self.ax.clear()

    def setTitle(self, text):
        self.ax.set_title(text)
