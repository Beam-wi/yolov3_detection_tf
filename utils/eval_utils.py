# coding: utf-8

from __future__ import division, print_function

import numpy as np
from collections import Counter
from shapely.geometry import Polygon

from utils.nms_utils import cpu_nms, gpu_nms


def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou


def evaluate_on_cpu(y_pred, y_true, num_classes, calc_now=True, score_thresh=0.5, iou_thresh=0.5):
    # y_pred -> [None, 13, 13, 255],
    #           [None, 26, 26, 255],
    #           [None, 52, 52, 255],

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        if true_boxes.shape[0] > 0:
            box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
            true_boxes[:, 0:2] = box_centers - box_sizes / 2.
            true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = cpu_nms(pred_boxes, pred_confs * pred_probs, num_classes,
                                                      score_thresh=score_thresh, iou_thresh=iou_thresh)

        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue
        if true_boxes.shape[0] <= 0:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if not match_idx in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred, y_true, num_classes, calc_now=True, score_thresh=0.5, iou_thresh=0.5): #0.5, 0.5
    # y_pred -> [None, 13, 13, 255],
    #           [None, 26, 26, 255],
    #           [None, 52, 52, 255],

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count} 每类有多少正确的,推测
    pred_labels_dict = {i: 0 for i in range(num_classes)} #每类预测了多少,推测
    true_positive_dict = {i: 0 for i in range(num_classes)} #每类有多少,实际

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        if true_boxes.shape[0] > 0:
            box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
            true_boxes[:, 0:2] = box_centers - box_sizes / 2.
            true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = sess.run(gpu_nms_op,
                                                       feed_dict={pred_boxes_flag: pred_boxes,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if not match_idx in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def bboxIou(bbox_a, bbox_b, label_a=None, label_b=None,
            scoresDet=None, scoresCls=None, iou_thr=0.01, offset=0, pool=None):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    返回iou矩阵和预测框正确、错误、漏检字典。
    同类别暂时无法做nms去重，所以默认已做过才行。
    Parameters
    ----------
    bbox_a : numpy.ndarray predict.
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray ground truth.
        An ndarray with shape :math:`(M, 4)`.
    label_a : list, the label of bbox_a. element is str or int.
    label_b : list, the label of bbox_b. element is str or int.
    scoresDet : list, the det scores for predict.
    scoresCls : list, the cls scores for predict.
    iou_thr: float, judge if the pre is true, must appear with label_a and
        label_b.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height)
        is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are
        in ``[0, 1]``.
    Returns
    -------
    iou_matrix: numpy.ndarray
            An ndarray with shape :math:`(N, M)` indicates IOU between each
            pairs of bounding boxes in `bbox_a` and `bbox_b`.
    result_dict: return dict, if label_a != None and label_b != None.
                {'T': {'b': np.ndarray(n, 4),
                       'l': [n]},
                 'M': {'b': np.ndarray(m, 4),
                       'l': [m]},
                 'W': {'b': np.ndarray(l, 4),   # predict box
                       'l': [l],    # predict label
                       'tl': [l],   # true label
                       'sd': [l],   # det score
                       'sc': [l]}   # cls score
                }
    """
    def filter_candidate(candidate, iou_matrix):
        dist_candidate = np.empty((0, 2), dtype=np.int)
        for i in range(iou_matrix.shape[-1]):
            repeat_cand = candidate[candidate[:, 1] == i]
            if not repeat_cand.shape[0]: continue
            repeat_matrix = iou_matrix[repeat_cand[:, 0], i]
            coordinate = repeat_cand[np.argmax(repeat_matrix)][np.newaxis, :]
            dist_candidate = np.concatenate((dist_candidate, coordinate), axis=0)

        return dist_candidate

    def compute_tmw(iou_matrix):
        # true, wrong
        candidate = np.vstack(np.where(iou_matrix >= iou_thr)).T
        candidate = filter_candidate(candidate, iou_matrix)
        for once in candidate:
            if label_a[once[0]] == label_b[once[1]]:
                result_dict['T']['b'] = np.vstack((result_dict['T']['b'], np.expand_dims(bbox_a[once[0]], axis=0)))
                result_dict['T']['l'].append(label_a[once[0]])
            else:
                result_dict['W']['b'] = np.vstack((result_dict['W']['b'], np.expand_dims(bbox_a[once[0]], axis=0)))
                result_dict['W']['l'].append(label_a[once[0]])  # pre label
                result_dict['W']['tl'].append(label_b[once[1]])  # true label
                if isinstance(scoresDet, np.ndarray):
                    result_dict['W']['sd'].append(scoresDet[once[0]])  # d score
                if isinstance(scoresCls, np.ndarray):
                    result_dict['W']['sc'].append(scoresCls[once[0]])  # c score

        # wrong
        r_sum, c_sum = iou_matrix.max(-1), iou_matrix.max(0)  #
        r_select = list(set(np.where(r_sum < iou_thr)[0].tolist()).intersection(set(np.where(r_sum > 0)[0].tolist())))
        c_select = np.argmax(iou_matrix, axis=-1)
        result_dict['W']['b'] = np.vstack((result_dict['W']['b'], bbox_a[r_select]))
        result_dict['W']['l'].extend([label_a[w] for w in r_select])
        result_dict['W']['tl'].extend([label_b[c_select[w]] for w in r_select])
        result_dict['W']['sd'].extend([scoresDet[w] for w in r_select])
        result_dict['W']['sc'].extend([scoresCls[w] for w in r_select])

        r_equal = np.where(r_sum==0)[0].tolist()
        result_dict['W']['b'] = np.vstack((result_dict['W']['b'], bbox_a[r_equal]))
        result_dict['W']['l'].extend([label_a[w] for w in r_equal])
        result_dict['W']['tl'].extend(['background' for _ in r_equal])
        result_dict['W']['sd'].extend([scoresDet[w] for w in r_equal])
        result_dict['W']['sc'].extend([scoresCls[w] for w in r_equal])
        # miss
        result_dict['M']['b'] = np.vstack((result_dict['M']['b'], bbox_b[c_sum < iou_thr]))
        result_dict['M']['l'].extend([label_b[w] for w in np.where(c_sum < iou_thr)[0]])

        return iou_matrix, result_dict

    def pickout_tmw(iou_matrix):
        # wrong
        result_dict['W']['b'] = np.vstack((result_dict['W']['b'], bbox_a))
        result_dict['W']['l'].extend(label_a)
        result_dict['W']['tl'] = ['background' for _ in range(bbox_a.shape[0])]
        result_dict['W']['sd'].extend(scoresDet)
        result_dict['W']['sc'].extend(scoresCls)
        # miss
        result_dict['M']['b'] = np.vstack((result_dict['M']['b'], bbox_b))
        result_dict['M']['l'].extend(label_b)

        return iou_matrix, result_dict

    assert isinstance(bbox_a, np.ndarray), "bbox_a must be np.ndarray."
    assert isinstance(bbox_b, np.ndarray), "bbox_b must be np.ndarray."
    assert bbox_a.shape[0] == len(label_a), "bbox_a num not match label's."
    assert bbox_b.shape[0] == len(label_b), "bbox_b num not match label's."

    pointNum = bbox_a.shape[-1]
    iou_matrix = calIouMatrix(bbox_a, bbox_b, offset=0) if pointNum == 4 else calIouMatrixRotate(bbox_a, bbox_b, pool)

    result_dict = {x: {'b': np.empty((0, pointNum)), 'l': []} for x in ['T', 'M', 'W']}
    result_dict['W']['tl'], result_dict['W']['sd'], result_dict['W']['sc'] = list(), list(), list()
    if not isinstance(label_a, list) and not isinstance(label_b, list):
        return iou_matrix  # return iou matrix
    elif bbox_a.shape[0] > 0 and bbox_b.shape[0]>0:
        return compute_tmw(iou_matrix)
    else:
        return pickout_tmw(iou_matrix)


def calIouMatrix(bbox_a, bbox_b, offset=0):
    """计算iou矩阵，包括旋转框和矩形框都兼容
    bbox_a : numpy.ndarray predict. An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray ground truth. An ndarray with shape :math:`(M, 4)`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def calIouMatrixRotate(p, g, pool):
    """计算旋转框iou矩阵
    p: 预测样本框 numpy.ndarray predict. An ndarray with shape :math:`(N, 8)`.
    g: 真实样本框 numpy.ndarray predict. An ndarray with shape :math:`(M, 8)`.
    """
    def rotate_inter(poly_data):
        p_, g_ = poly_data['p'], poly_data['g']
        if not g_.is_valid or not p_.is_valid:
            return 0
        return Polygon(g_).intersection(Polygon(p_)).area

    if p.shape[1] < 4 or g.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    p_num, g_num = p.shape[0], g.shape[0]
    p_poly = {f"{i}": Polygon(x[0].reshape((4, 2))) for i, x in enumerate(zip(*(iter(p),)*1))}
    g_poly = {f"{j}": Polygon(x[0].reshape((4, 2))) for j, x in enumerate(zip(*(iter(g),)*1))}
    data = [{'p': p_poly[f"{i_}"], 'g': g_poly[f"{j_}"]} for j_ in range(g_num) for i_ in range(p_num)] if pool is not None else pool
    inter_matrix = np.array([[rotate_inter({'p': p_poly[f"{i_}"], 'g': g_poly[f"{j_}"]}) for j_ in range(g_num)] for i_ in range(p_num)]) if pool is None else np.array(pool.map(rotate_inter, data)).reshape((p_num, g_num))
    p_area = np.array([p_poly[f"{ii}"].area for ii in range(p_num)])
    g_area = np.array([g_poly[f"{jj}"].area for jj in range(g_num)])
    if np.prod(inter_matrix.shape):
        return inter_matrix / (p_area[:, None] + g_area -inter_matrix)
    else:
        return inter_matrix


def rotateIou(p, g):
    """计算旋转框的iou
    p: 预测样本框
    g: 真实样本框
    """
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union
