import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)

from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from Dataset.SOT.FactorySeeds.OTB100 import OTB100_Seed
from algorithms.tracker.deform_tracker.builder import build_deform_tracker
from data.tracking.simple_dataloader import SimpleTrackingDatasetIterator
from evaluation.experiment.run_otb_evaluation import run_otb_evaluation
from Dataset.Filter.DataCleaner_BoundingBox import DataCleaner_BoundingBox
from Dataset.Filter.DataCleaner_Integrity import DataCleaner_Integrity
from workarounds.all import apply_all_workarounds
from evaluation.logger.simple import SimpleEvaluationLogger
import argparse


if __name__ == '__main__':
    apply_all_workarounds()
    parser = argparse.ArgumentParser(description='Run tracker on OTB100 dataset.')
    parser.add_argument('network_config_path', type=str, help='Path to network config')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('--otb100_path', type=str, default=None, help="Path to OTB100 dataset.")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save results.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args()
    tracker = build_deform_tracker(args.network_config_path, args.weight_path, args.device)
    dataset = SingleObjectTrackingDatasetFactory(OTB100_Seed(args.otb100_path)).constructMemoryMapped(
        [DataCleaner_BoundingBox(), DataCleaner_Integrity()])
    dataset = SimpleTrackingDatasetIterator(dataset, True)

    logger = SimpleEvaluationLogger(args.output_path)

    auc = run_otb_evaluation(dataset, tracker, logger)
    print('success_auc={}'.format(auc))
