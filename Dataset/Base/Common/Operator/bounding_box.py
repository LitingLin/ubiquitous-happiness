from Dataset.Type.bounding_box_format import BoundingBoxFormat


def convert_bounding_box_format(bounding_box, source_format: BoundingBoxFormat, target_format: BoundingBoxFormat, strict = True):
    if source_format == BoundingBoxFormat.XYWH:
        if target_format == BoundingBoxFormat.XYWH:
            return bounding_box
        elif target_format == BoundingBoxFormat.XYXY:
            return bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]
        elif target_format == BoundingBoxFormat.Quadrilateral:
            return bounding_box[0], bounding_box[1], bounding_box[0], bounding_box[1] + bounding_box[3], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3], bounding_box[0] + bounding_box[2], bounding_box[1]
        else:
            raise NotImplementedError
    elif source_format == BoundingBoxFormat.XYXY:
        if target_format == BoundingBoxFormat.XYWH:
            return bounding_box[0], bounding_box[1], bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]
        elif target_format == BoundingBoxFormat.XYXY:
            return bounding_box
        elif target_format == BoundingBoxFormat.Quadrilateral:
            return bounding_box[0], bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2], bounding_box[3], bounding_box[2], bounding_box[1]
        else:
            raise NotImplementedError
    elif source_format == BoundingBoxFormat.Quadrilateral:
        if target_format == BoundingBoxFormat.Quadrilateral:
            return bounding_box
        elif target_format == BoundingBoxFormat.XYXY or target_format == BoundingBoxFormat.XYWH:
            if strict:
                raise RuntimeError(f'from {source_format} to {target_format} is not allowed in strict mode')
            x_min = min((bounding_box[0], bounding_box[2], bounding_box[4], bounding_box[6]))
            x_max = max((bounding_box[0], bounding_box[2], bounding_box[4], bounding_box[6]))
            y_min = min((bounding_box[1], bounding_box[3], bounding_box[5], bounding_box[7]))
            y_max = max((bounding_box[1], bounding_box[3], bounding_box[5], bounding_box[7]))
            if target_format == BoundingBoxFormat.XYWH:
                return x_min, y_min, x_max - x_min, y_max - y_min
            else:
                return x_min, y_min, x_max, y_max
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def fit_in_image_boundary_xyxy(bounding_box, image_size):
    w, h = image_size
    return max(0, min(w, bounding_box[0])), max(0, min(h, bounding_box[1])), max(0, min(w, bounding_box[2])), max(0, min(h, bounding_box[3]))


def fit_in_image_boundary_quadrilateral(bounding_box, image_size):
    w, h = image_size
    return max(0, min(w, bounding_box[0])), max(0, min(h, bounding_box[1])), max(0, min(w, bounding_box[2])), max(0, min(h, bounding_box[3])), max(0, min(w, bounding_box[4])), max(0, min(h, bounding_box[5])), max(0, min(w, bounding_box[6])), max(0, min(h, bounding_box[7]))


def fit_in_image_boundary(bounding_box, bounding_box_format: BoundingBoxFormat, image_size):
    if bounding_box_format == BoundingBoxFormat.XYWH:
        bounding_box = (bounding_box[0], bounding_box[1], bounding_box[2] + bounding_box[0], bounding_box[3] + bounding_box[1])
        fit_in_image_boundary_xyxy(bounding_box, image_size)
        return bounding_box[0], bounding_box[1], bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]
    elif bounding_box_format == BoundingBoxFormat.XYXY:
        return fit_in_image_boundary_xyxy(bounding_box, image_size)
    elif bounding_box_format == BoundingBoxFormat.Quadrilateral:
        return fit_in_image_boundary_quadrilateral(bounding_box, image_size)
    else:
        raise NotImplementedError


def check_bounding_box_validity_by_intersection_over_image(bounding_box, bounding_box_format, image_size):
    w, h = image_size
    if bounding_box_format == BoundingBoxFormat.Quadrilateral:
        from shapely.geometry import Polygon
        A = Polygon(((bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (bounding_box[4], bounding_box[5]), (bounding_box[6], bounding_box[7])))
        B = Polygon(((0, 0), (w, 0), (w, h), (0, h)))
        return A.intersection(B).area > 0
    elif bounding_box_format == BoundingBoxFormat.XYXY or bounding_box_format == BoundingBoxFormat.XYWH:
        if bounding_box_format == BoundingBoxFormat.XYWH:
            bounding_box = [bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]]
        inter_x1 = max(bounding_box[0], 0)
        inter_y1 = max(bounding_box[1], 0)
        inter_x2 = min(bounding_box[2], w)
        inter_y2 = min(bounding_box[3], h)

        return inter_y2 - inter_y1 > 0 and inter_x2 - inter_x1 > 0
    else:
        raise NotImplementedError
