from .pipeline import SimpleDataPipelineOrganizer_NoZMask
from data.detr_tracking_variants.processor import DETRTrackingProcessor
from data.tracking.processor.curation import SiamFCLikeCurationExemplarBlankBackgroundProcessor
from data.tracking.processor.resizing import RandomResizing_KeepingAspect_Processor


def build_processor(network_config: dict, train_config: dict):
    siamfc_curation_context = network_config['backbone']['siamfc']['context']
    siamfc_curation_exemplar_size = network_config['backbone']['siamfc']['exemplar_size']
    train_min_instance_size = train_config['train']['data']['instance_size']['min']
    train_max_instance_size = train_config['train']['data']['instance_size']['max']
    val_min_instance_size = train_config['val']['data']['instance_size']['min']
    val_max_instance_size = train_config['val']['data']['instance_size']['max']
    train_processor = DETRTrackingProcessor(
        SiamFCLikeCurationExemplarBlankBackgroundProcessor(siamfc_curation_context, siamfc_curation_exemplar_size),
        RandomResizing_KeepingAspect_Processor(train_min_instance_size, train_max_instance_size),
        SimpleDataPipelineOrganizer_NoZMask())
    val_processor = DETRTrackingProcessor(
        SiamFCLikeCurationExemplarBlankBackgroundProcessor(siamfc_curation_context, siamfc_curation_exemplar_size),
        RandomResizing_KeepingAspect_Processor(val_min_instance_size, val_max_instance_size),
        SimpleDataPipelineOrganizer_NoZMask())

    return train_processor, val_processor
