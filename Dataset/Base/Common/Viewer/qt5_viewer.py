from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen


def draw_object(painter, bounding_box_accessor, category_id_accessor, object_id_accessor, bounding_box_format, category_id_color_map, category_id_name_map_accessor):
    if category_id_name_map_accessor.has_category_id_name_map():
        category_id_name_map = category_id_name_map_accessor.get_category_id_name_map()
    else:
        category_id_name_map = None
    if bounding_box_accessor.has_bounding_box():
        category_id = None
        if category_id_accessor is not None:
            if isinstance(category_id_accessor, (list, tuple)):
                for c_category_id_accessor in category_id_accessor:
                    if c_category_id_accessor.has_category_id():
                        category_id = c_category_id_accessor.get_category_id()
                        break
            else:
                if category_id_accessor.has_category_id():
                    category_id = category_id_accessor.get_category_id()
        bounding_box = bounding_box_accessor.get_bounding_box()
        if bounding_box_format is None:
            bounding_box, bounding_box_format, bounding_box_validity_flag = bounding_box
        else:
            bounding_box_validity_flag = bounding_box_accessor.get_bounding_box_validity_flag()

        if category_id is None or category_id_color_map is None:
            color = QColor(255, 0, 0, int(0.5 * 255))
        else:
            color = category_id_color_map[category_id]
        pen = QPen(color)
        if bounding_box_validity_flag is False:
            pen.setStyle(Qt.DashDotDotLine)
        painter.setPen(pen)
        if bounding_box_format.name == 'XYWH':
            painter.drawRect(bounding_box)
        elif bounding_box_format.name == 'XYXY':
            painter.drawRect(bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
        else:
            painter.drawPolygon(bounding_box)
        label_string = []
        if object_id_accessor is not None and ((not hasattr(object_id_accessor, 'has_id') and hasattr(object_id_accessor, 'get_id')) or object_id_accessor.has_id()):
            label_string.append(str(object_id_accessor.get_id()))
        if not (category_id is None or category_id_color_map is None or category_id_name_map is None):
            label_string.append(category_id_name_map[category_id])
        painter.drawLabel('-'.join(label_string), (bounding_box[0], bounding_box[1]), color)
