import numpy as np
import torch

def mask_valid_boxes(boxes):
    w = boxes[:, 2]
    h = boxes[:, 3]
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    mask = (w > 2) & (h > 2) & (ar < 30)
    
    return mask

def constraint_theta(bboxes):
    keep_dim = False
    if len(bboxes.shape)==1:
        keep_dim = True
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.unsqueeze(0)
        if isinstance(bboxes, np.ndarray):
            bboxes = np.expand_dims(bboxes, 0)
        else: bboxes = [bboxes]
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.grad_fn is False, 'Modifying variables to be calc. grad is not allowed.'
    assert (bboxes[:, 4] >= -90).all() and (bboxes[:, 4] <= 90).all(), 'theta must between (-90, 90)'

    for box in bboxes:
        if box[4] > 45.0:
            box[2], box[3] = box[3], box[2]
            box[4] -= 90
        elif box[4] < -45.0:
            box[2], box[3] = box[3], box[2]
            box[4] += 90
        elif abs(box[4]) == 45:
            if box[2] > box[3]:
                box[4] = -45
            else:
                box[4] = 45
                box[2], box[3] = box[3], box[2]
    if keep_dim:
        return bboxes[0]
    else:
        return bboxes
