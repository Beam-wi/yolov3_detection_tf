import copy
import torch
import numpy as np


def nms(boxes, scores, iou_threshold, offset=0, cat=False):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the \
            same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    def bbox_iou(bbox_a_, bbox_b_, offset=0):
        """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
        Parameters
        ----------
        bbox_a : numpy.ndarray true box
            An ndarray with shape :math:`(N, 4)`.
        bbox_b : numpy.ndarray predict box
            An ndarray with shape :math:`(M, 4)`.
        offset : float or int, default is 0
            The ``offset`` is used to control the whether the width(or height) is computed as
            (right - left + ``offset``).
            Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
        Returns
        -------
        numpy.ndarray
            An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
            bounding boxes in `bbox_a` and `bbox_b`.
            EP:
                gt = np.array([[2352, 2560, 2469, 2675, 0], [20, 40, 50, 60, 0], [1948, 1853, 2064, 1967, 0]])
                pr = np.array([[2357, 2565, 2474, 2680, 0], [1948, 1853, 2064, 1967, 0], [22, 42, 48, 58, 0],
                               [948, 853, 1064, 967, 0], [1948, 1853, 2064, 1967, 0]])
            return
                [[0.84441398 0.         0.         0.         0.        ]
                 [0.         0.         0.69333333 0.         0.        ]
                 [0.         1.         0.         0.         1.        ]]
                每行对应一个gt目标，只有一个值表示这个目标预测正确，都为0表示漏检，多个值表示误报了。
        """
        bbox_a = bbox_a_.cpu().numpy()
        bbox_b = bbox_b_.cpu().numpy()
        if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
            raise IndexError("Bounding boxes axis 1 must have at least length 4")

        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

        area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
        area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
        return torch.from_numpy(area_i / (area_a[:, None] + area_b - area_i))

    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)
    boxes+=offset
    num = scores.shape[0]
    matrix = bbox_iou(boxes, boxes, offset=offset)
    det_num = {x:0 for x in range(num)}
    for i in range(num):
        gt_ind = np.where(copy.deepcopy(torch.gt(matrix[:, i], iou_threshold)).cpu().numpy()==True)[0].tolist() # 大于
        gt_iou = gt_ind[copy.deepcopy(scores[gt_ind]).cpu().argmax().item()] # 大于
        lt_iou = np.where(copy.deepcopy(torch.lt(matrix[:, i], iou_threshold)).cpu().numpy()==True)[0].tolist()  # 小于
        eq_iou = np.where(copy.deepcopy(torch.eq(matrix[:, i], iou_threshold)).cpu().numpy()==True)[0].tolist()  # 等于
        lt_iou.extend(eq_iou)
        lt_iou.append(gt_iou)
        for j in lt_iou:
            det_num[j] += 1
    inds = [k for k, v in det_num.items() if v==num]
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1) if cat else [boxes[inds], scores[inds].reshape(-1, 1)]
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds