import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_path)
default_config_path = os.path.join(root_path, 'config', 'detr_tracking_variants', 'encoder_shared_params_decoder_cross_attn_resnet50_decoder_no_z_mask_new_aug')

from algorithms.tracker.detr_tracking_variants.encoder_shared_params_decoder_cross_attn_resnet50_decoder_no_z_mask_new_aug.builder import build_detr_tracker
from evaluation.evaluator.got10k.run_evaluation import run_evaluation_on_tracker
from workarounds.all import apply_all_workarounds
import argparse


if __name__ == '__main__':
    apply_all_workarounds()
    parser = argparse.ArgumentParser(description='Run tracker on OTB GOT10k LaSOT.')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('--net-config', type=str, default=os.path.join(default_config_path, 'network.yaml'),
                        help='Path to network config')
    parser.add_argument('--output-path', type=str, default=None, help="Path to save results.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    parser.add_argument('--visualize', action='store_true', help='Visualize the tracking procedure')
    args = parser.parse_args()
    tracker = build_detr_tracker(args.net_config, args.weight_path, args.device)

    result_path = os.path.join(args.output_path, 'results')
    os.makedirs(result_path, exist_ok=True)
    report_path = os.path.join(args.output_path, 'reports')
    os.makedirs(report_path, exist_ok=True)

    run_evaluation_on_tracker(tracker, tracker.get_name(), True, result_path, report_path, args.visualize)
