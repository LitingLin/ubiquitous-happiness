from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from Dataset.SOT.FactorySeeds.OTB100 import OTB100_Seed
from factory.tracker import TrackerFactory
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
    parser.add_argument('otb100_path', type=str, help="Path to OTB100 dataset.")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save results.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args()
    tracker = TrackerFactory.create(device=args.device)
    dataset = SingleObjectTrackingDatasetFactory(OTB100_Seed(args.otb100_path)).construct_memory_mapped(
        [DataCleaner_BoundingBox(), DataCleaner_Integrity()])
    dataset = SimpleTrackingDatasetIterator(dataset, False)

    logger = SimpleEvaluationLogger(args.output_path)

    auc = run_otb_evaluation(dataset, tracker, logger)
    print('success_auc={}'.format(auc))
