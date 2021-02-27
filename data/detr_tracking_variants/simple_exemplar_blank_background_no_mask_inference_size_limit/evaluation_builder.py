from data.detr_tracking_variants.processor import DETRTrackingEvaluationProcessor
from data.tracking.processor.curation import SiamFCLikeCurationExemplar_MaskGenerating_Processor
from data.tracking.processor.resizing import SizeLimited_MinMax_KeepingAspect_Image_Processor
from data.detr_tracking_variants.common.post_processor import PostProcessor_ImageToTorchImagenetNormalizationMaskGenerating, PostProcessor_ImageToTorchImagenetNormalizationNoAnnotation
from data.detr_tracking_variants.common.pre_processor import PreProcessor_BoundingBoxToNumpyToXYWH


def build_evaluation_processor(network_config: dict):
    siamfc_curation_context = network_config['backbone']['siamfc']['context']
    siamfc_curation_exemplar_size = network_config['backbone']['siamfc']['exemplar_size']
    instance_size_limit_min = network_config['backbone']['siamfc']['instance_size_limit']['min']
    instance_size_limit_max = network_config['backbone']['siamfc']['instance_size_limit']['max']

    return DETRTrackingEvaluationProcessor(
        SiamFCLikeCurationExemplar_MaskGenerating_Processor(siamfc_curation_context, siamfc_curation_exemplar_size),
        SizeLimited_MinMax_KeepingAspect_Image_Processor(instance_size_limit_min, instance_size_limit_max),
        PostProcessor_ImageToTorchImagenetNormalizationMaskGenerating(),
        PostProcessor_ImageToTorchImagenetNormalizationNoAnnotation(),
        PreProcessor_BoundingBoxToNumpyToXYWH()
    )
