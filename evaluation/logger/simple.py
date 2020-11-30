import numpy as np
import os


class SimpleEvaluationLogger:
    def __init__(self, output_path: str=None):
        self.output_path = output_path

    def log_sequence_result(self, name: str, predicted_bboxes: np.ndarray, **kwargs):
        string = name + ': '
        for item_key, item_value in kwargs.items():
            string += '{}={}, '.format(item_key, item_value)
        string = string[: -2]
        print(string)
        if self.output_path is not None:
            with open(os.path.join(self.output_path, '{}-stats.txt'.format(name)), 'w') as f:
                f.write(string)
            np.savetxt(os.path.join(self.output_path, '{}-pred.txt'.format(name)), predicted_bboxes, delimiter='\t', fmt='%d')
