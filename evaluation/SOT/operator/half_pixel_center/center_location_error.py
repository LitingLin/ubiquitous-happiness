def calculate_center_location_error_torch_vectorized(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * pred_bb[:, 2:]
    anno_center = anno_bb[:, :2] + 0.5 * anno_bb[:, 2:]

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center
