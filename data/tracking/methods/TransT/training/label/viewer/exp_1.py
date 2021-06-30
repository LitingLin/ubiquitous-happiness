class


def data_process_callback(data):
    (z_image_batch, x_image_batch), \
    (num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch,
     target_class_label_vector_batch, target_bounding_box_label_matrix_batch), \
    is_positives, context = data
    assert context is None

