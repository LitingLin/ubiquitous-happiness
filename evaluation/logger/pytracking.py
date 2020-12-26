import numpy as np
import os


class PyTrackingLogger:
    def __init__(self, output_path=None):
        self.output_path = output_path

    def log_sequence_result(self, name: str, predicted_bboxes: np.ndarray, **kwargs):
        print(f'Sequence: {name}')
        print(f'FPS: {kwargs["fps"]}')
        predicted_bboxes = predicted_bboxes.copy()
        predicted_bboxes[:, 0] += 1
        predicted_bboxes[:, 1] += 1
        if self.output_path is not None:
            np.savetxt(os.path.join(self.output_path, '{}.txt'.format(name)), predicted_bboxes, delimiter='\t',
                       fmt='%d')
