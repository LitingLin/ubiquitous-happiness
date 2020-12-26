from ..method.ope import run_one_pass_evaluation_on_sequence
from ..criterion.otb.success_plot import SuccessPlotCriterion
from ..criterion.otb.precision_plot import PrecisionPlotCriterion
import numpy as np
import time


def run_otb_evaluation(dataset, tracker, logger):
    aucs = []
    length = []
    for sequence in dataset:
        start = time.perf_counter()
        gt, predicted = run_one_pass_evaluation_on_sequence(sequence, tracker)
        end = time.perf_counter()
        length_of_sequence = len(sequence)
        time.perf_counter()
        success_plot_criterion = SuccessPlotCriterion(gt, predicted)
        precision_plot_criterion = PrecisionPlotCriterion(gt, predicted)
        success_auc = success_plot_criterion.auc()
        aucs.append(success_auc)
        length.append(len(sequence))
        logger.log_sequence_result(sequence.getName(), predicted, fps=length_of_sequence / (end - start), success_auc=success_auc, precision=precision_plot_criterion.at(20))

    aucs = np.array(aucs)
    return np.mean(aucs)
