from data.operator.bbox.spatial.utility.aligned.normalize_v2 import BoundingBoxNormalizationHelper
from data.types.bounding_box_format import BoundingBoxFormat


def _get_bounding_box_normalization_helper(network_config: dict):
    if 'bounding_box_normalization_protocol' in network_config['data']:
        return BoundingBoxNormalizationHelper(network_config['data']['bounding_box_normalization_protocol']['interval'], network_config['data']['bounding_box_normalization_protocol']['range'])
    else:
        return BoundingBoxNormalizationHelper('[]', [0, 1])


def _get_bounding_box_format(network_config: dict):
    if 'bounding_box_format' in network_config['head']:
        return BoundingBoxFormat[network_config['head']['bounding_box_format']]
    else:
        return BoundingBoxFormat.CXCYWH
