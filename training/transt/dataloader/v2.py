from data.tracking.methods.TransT.training.builder import build_transt_data_processor
from data.tracking.builder.siamfc.stateful.data_loader import build_siamfc_sampling_dataloader
from data.performance.cuda_prefetcher import TensorFilteringByIndices


def build_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                      val_dataset_config_path: str):
    processor, collate_fn = build_transt_data_processor(network_config, train_config)
    return build_siamfc_sampling_dataloader(args, train_config, train_dataset_config_path, val_dataset_config_path, processor, processor, collate_fn, TensorFilteringByIndices((0, 1, 3)))
