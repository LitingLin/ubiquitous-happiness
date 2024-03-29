from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
from data.types.bounding_box_format import BoundingBoxFormat


def recover_bounding_box_from_normalized(label, search_region_size, bbox_normalizer):
    label = label.tolist()

    bbox = bbox_normalizer.denormalize(label, search_region_size)
    return bbox


class TransTBoundingBoxPostProcessor:
    def __init__(self, search_region_size, bbox_size_limit_min_wh, bbox_size_limit_in_curated_image,
                 bbox_normalizer,
                 input_format=BoundingBoxFormat.XYXY):
        self.search_region_size = search_region_size
        self.bbox_size_limit_min_wh = bbox_size_limit_min_wh
        self.bbox_size_limit_in_curated_image = bbox_size_limit_in_curated_image
        assert input_format in (BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH)
        self.input_format = input_format
        self.bbox_normalizer = bbox_normalizer

    def __call__(self, bbox_normalized, image_size, curation_parameter):
        curation_parameter = curation_parameter.tolist()
        curation_scaling, curation_source_center_point, curation_target_center_point = curation_parameter
        bbox = recover_bounding_box_from_normalized(bbox_normalized, self.search_region_size, self.bbox_normalizer)
        if self.input_format == BoundingBoxFormat.CXCYWH:
            from data.operator.bbox.spatial.cxcywh2xyxy import bbox_cxcywh2xyxy
            bbox = bbox_cxcywh2xyxy(bbox)

        if self.bbox_size_limit_in_curated_image:
            internal_bbox = bbox
            if self.bbox_size_limit_min_wh[0] >= 0 and self.bbox_size_limit_min_wh[1] >= 0:
                from data.operator.bbox.spatial.xyxy2cxcywh import bbox_xyxy2cxcywh
                from data.operator.bbox.spatial.cxcywh2xyxy import bbox_cxcywh2xyxy
                internal_bbox = list(bbox_xyxy2cxcywh(internal_bbox))

                if internal_bbox[2] < self.bbox_size_limit_min_wh[0]:
                    internal_bbox[2] = self.bbox_size_limit_min_wh[0]
                if internal_bbox[3] < self.bbox_size_limit_min_wh[1]:
                    internal_bbox[3] = self.bbox_size_limit_min_wh[1]

                internal_bbox = bbox_cxcywh2xyxy(internal_bbox)
            internal_bbox = bbox_scale_and_translate(internal_bbox, [1.0 / curation_scaling_ for curation_scaling_ in
                                                                     curation_scaling],
                                                     curation_target_center_point, curation_source_center_point)
            bbox = bbox_scale_and_translate(bbox, [1.0 / curation_scaling_ for curation_scaling_ in curation_scaling],
                                            curation_target_center_point, curation_source_center_point)

            return bbox, self._get_target_bbox_state(internal_bbox, image_size)
        else:

            bbox = bbox_scale_and_translate(bbox, [1.0 / curation_scaling_ for curation_scaling_ in curation_scaling],
                                            curation_target_center_point, curation_source_center_point)

            return bbox, self._get_target_bbox_state(bbox, image_size)

    def _get_target_bbox_state(self, bbox, image_size):
        w, h = image_size
        from data.operator.bbox.spatial.xyxy2cxcywh import bbox_xyxy2cxcywh
        from data.operator.bbox.spatial.cxcywh2xyxy import bbox_cxcywh2xyxy
        from data.operator.bbox.spatial.utility.aligned.image import bounding_box_fit_in_image_boundary, \
            get_image_bounding_box
        bbox = bounding_box_fit_in_image_boundary(bbox, (w, h))
        image_boundary = get_image_bounding_box((w, h))

        if self.bbox_size_limit_min_wh[0] >= 0 and self.bbox_size_limit_min_wh[1] >= 0:
            if self.bbox_size_limit_min_wh[0] < (image_boundary[2] - image_boundary[0]) and self.bbox_size_limit_min_wh[1] < (
                    image_boundary[3] - image_boundary[1]):
                bbox = list(bbox_xyxy2cxcywh(bbox))

                if bbox[2] < self.bbox_size_limit_min_wh[0]:
                    bbox[2] = self.bbox_size_limit_min_wh[0]
                if bbox[3] < self.bbox_size_limit_min_wh[1]:
                    bbox[3] = self.bbox_size_limit_min_wh[1]

                valid_bbox_center_image_boundary = [image_boundary[0] + self.bbox_size_limit_min_wh[0] / 2,
                                                    image_boundary[1] + self.bbox_size_limit_min_wh[1] / 2,
                                                    image_boundary[2] - self.bbox_size_limit_min_wh[0] / 2,
                                                    image_boundary[3] - self.bbox_size_limit_min_wh[1] / 2]
                if bbox[0] < valid_bbox_center_image_boundary[0]:
                    bbox[0] = valid_bbox_center_image_boundary[0]
                if bbox[1] < valid_bbox_center_image_boundary[1]:
                    bbox[1] = valid_bbox_center_image_boundary[1]
                if bbox[0] > valid_bbox_center_image_boundary[2]:
                    bbox[0] = valid_bbox_center_image_boundary[2]
                if bbox[1] > valid_bbox_center_image_boundary[3]:
                    bbox[1] = valid_bbox_center_image_boundary[3]
                bbox = bbox_cxcywh2xyxy(bbox)
        return bbox
