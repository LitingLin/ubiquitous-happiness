from Utils.vot import VOT, Rectangle
import numpy as np
from Utils.decode_image import decode_image_file


def run_vot(tracker):
    handle = VOT("rectangle")
    initial_bbox = handle.region()
    initial_bbox = np.array([initial_bbox.x, initial_bbox.y, initial_bbox.width, initial_bbox.height], dtype=np.float)

    initial_path = handle.frame()
    if initial_path is None:
        return

    initial_image = decode_image_file(initial_path)
    tracker.initialize(initial_image, initial_bbox)

    while True:
        path = handle.frame()
        if path is None:
            return
        image = decode_image_file(path)
        predicted_bbox = tracker.track(image)
        handle.report(Rectangle(x=predicted_bbox[0], y=predicted_bbox[1], width=predicted_bbox[2], height=predicted_bbox[3]))
