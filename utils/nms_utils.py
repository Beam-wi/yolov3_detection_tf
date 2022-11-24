# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf


# def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
#     """
#     Perform NMS on GPU using TensorFlow.
#
#     params:
#         boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
#         scores: tensor of shape [1, 10647, num_classes], score=conf*prob
#         num_classes: total number of classes
#         max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
#         score_thresh: if [ highest class probability score < score_threshold]
#                         then get rid of the corresponding box
#         iou_thresh: real value, "intersection over union" threshold used for NMS filtering
#     """
#
#     boxes_list, label_list, score_list = [], [], []
#     max_boxes = tf.constant(max_boxes, dtype='int32')
#
#     # since we do nms for single image, then reshape it
#     boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
#     score = tf.reshape(scores, [-1, num_classes])
#
#     # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
#     mask = tf.greater_equal(score, tf.constant(score_thresh))
#     # Step 2: Do non_max_suppression for each class
#     for i in range(num_classes):
#         # Step 3: Apply the mask to scores, boxes and pick them out
#         filter_boxes = tf.boolean_mask(boxes, mask[:,i])
#         filter_score = tf.boolean_mask(score[:,i], mask[:,i])
#         nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
#                                                    scores=filter_score,
#                                                    max_output_size=max_boxes,
#                                                    iou_threshold=iou_thresh, name='nms_indices')
#         label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
#         boxes_list.append(tf.gather(filter_boxes, nms_indices))
#         score_list.append(tf.gather(filter_score, nms_indices))
#
#     boxes = tf.concat(boxes_list, axis=0)
#
#     score = tf.concat(score_list, axis=0)
#     label = tf.concat(label_list, axis=0)
#
#     return boxes, score, label


def gpu_nms(boxes, scores, labels, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """
    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    scores = tf.reshape(scores, [-1, 1])
    labels = tf.reshape(labels, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(scores, tf.constant(score_thresh))

    # Step 2: Do non_max_suppression for each class
    # Step 3: Apply the mask to scores, boxes and pick them out
    filter_boxes = tf.boolean_mask(boxes, mask[:, 0])
    filter_scores = tf.boolean_mask(scores, mask)
    filter_labels = tf.boolean_mask(labels, mask[:, 0])
    nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                               scores=filter_scores,
                                               max_output_size=max_boxes,
                                               iou_threshold=iou_thresh, name='nms_indices')
    label_list.append(
        tf.argmax(tf.nn.softmax(tf.gather(filter_labels, nms_indices), axis=-1), axis=-1)
        if num_classes > 1 else tf.zeros_like(tf.gather(filter_scores, nms_indices), 'int32'))
    boxes_list.append(tf.gather(filter_boxes, nms_indices))
    score_list.append(tf.gather(filter_scores, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)

    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def gpu_nms_one_class(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    num_box = tf.cast(tf.shape(boxes)[0], tf.int32)
    #需要根据类别数据调整
    # labels = tf.concat([tf.zeros([num_box, 1], dtype=tf.int32),
    #                     tf.ones([num_box, 1], dtype=tf.int32),
    #                     tf.ones([num_box, 1], dtype=tf.int32) * 2,
    #                     tf.ones([num_box, 1], dtype=tf.int32) * 3,
    #                     tf.ones([num_box, 1], dtype=tf.int32) * 4,
    #                     tf.ones([num_box, 1], dtype=tf.int32) * 5,
    #                     tf.ones([num_box, 1], dtype=tf.int32) * 6,
    #                     tf.ones([num_box, 1], dtype=tf.int32) * 7,
    #                     ], axis=1)
    labels = tf.concat([tf.ones([num_box, 1], dtype=tf.int32) * i for i in range(num_classes)], axis=1)

    boxes = tf.tile(boxes, [1, num_classes])
    boxes = tf.reshape(boxes, [-1, 4])
    score = tf.reshape(scores, [-1, 1])
    labels = tf.reshape(labels, [-1, 1])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each clas
    for i in range(1):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        filter_labels = tf.boolean_mask(labels[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        # label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        label_list.append(tf.gather(filter_labels, nms_indices))
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)

    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label



def gpu_nms_1(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    boxes_rate = tf.constant([[5472 / 2720, 3648 / 1824, 5472 / 2720, 3648 / 1824]], tf.float32)
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    boxes = boxes * boxes_rate
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)

    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def gpu_nms_2(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    boxes_rate = tf.constant([[5472/3360, 3648 / 2240, 5472 / 3360, 3648 / 2240]], tf.float32)
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    boxes = boxes * boxes_rate
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)

    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0: 
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: 
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def gpu_nms_combine(boxes_1, scores_1, boxes_2, scores_2, num_classes, max_boxes=50, score_thresh_1=0.5, score_thresh_2=0.5, iou_thresh=0.5, area_thresh=1.):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """
    def broadcast_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes_1_rate = tf.constant([[5472/2720, 3648/1824, 5472/2720, 3648/1824]], tf.float32)
    boxes_1 = tf.reshape(boxes_1, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    boxes_1 = boxes_1 * boxes_1_rate
    score_1 = tf.reshape(scores_1, [-1, num_classes])
    boxes_2_rate = tf.constant([[5472/3360, 3648 / 2240, 5472 / 3360, 3648 / 2240]], tf.float32)
    boxes_2 = tf.reshape(boxes_2, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    boxes_2 = boxes_2 * boxes_2_rate
    score_2 = tf.reshape(scores_2, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask_1 = tf.greater_equal(score_1, tf.constant(score_thresh_1))
    mask_2 = tf.greater_equal(score_2, tf.constant(score_thresh_2))

    # Step 2: Do non_max_suppression for each class
    boxes_combine = tf.concat([boxes_1, boxes_2], 0)
    score_combine = tf.concat([score_1, score_2], 0)
    mask_combine = tf.concat([mask_1, mask_2], 0)
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes_combine, mask_combine[:,i])
        filter_score = tf.boolean_mask(score_combine[:,i], mask_combine[:,i])

        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    #
    # mg_box_xy = (boxes_list[1][..., 0:2] + boxes_list[1][..., 2:4]) / 2
    # mg_box_wh = boxes_list[1][..., 2:4] - boxes_list[1][..., 0:2]
    # yx_box_xy = (boxes_list[0][..., 0:2] + boxes_list[0][..., 2:4]) / 2
    # yx_box_wh = boxes_list[0][..., 2:4] - boxes_list[0][..., 0:2]
    # iou = broadcast_iou(mg_box_xy, mg_box_wh, yx_box_xy, yx_box_wh)
    # best_iou = tf.reduce_max(iou, axis=-1)
    # object_mask = tf.cast(best_iou < 0.5, tf.float32)
    # object_mask = tf.expand_dims(object_mask, -1)
    #
    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)
    boxes_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])
    mask_area = tf.greater_equal(boxes_area, tf.constant(area_thresh))
    boxes = tf.boolean_mask(boxes, mask_area)
    score = tf.boolean_mask(score, mask_area)
    label = tf.boolean_mask(label, mask_area)

    return boxes, score, label


def gpu_nms_new(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5, area_thresh=1.):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """
    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes_rate = tf.constant([[5472/3008, 3648/2016, 5472/3008, 3648/2016]], tf.float32)
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    boxes = boxes * boxes_rate
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))

    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])

        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    #
    # mg_box_xy = (boxes_list[1][..., 0:2] + boxes_list[1][..., 2:4]) / 2
    # mg_box_wh = boxes_list[1][..., 2:4] - boxes_list[1][..., 0:2]
    # yx_box_xy = (boxes_list[0][..., 0:2] + boxes_list[0][..., 2:4]) / 2
    # yx_box_wh = boxes_list[0][..., 2:4] - boxes_list[0][..., 0:2]
    # iou = broadcast_iou(mg_box_xy, mg_box_wh, yx_box_xy, yx_box_wh)
    # best_iou = tf.reduce_max(iou, axis=-1)
    # object_mask = tf.cast(best_iou < 0.5, tf.float32)
    # object_mask = tf.expand_dims(object_mask, -1)
    #
    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)
    boxes_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])
    mask_area = tf.greater_equal(boxes_area, tf.constant(area_thresh))
    boxes = tf.boolean_mask(boxes, mask_area)
    score = tf.boolean_mask(score, mask_area)
    label = tf.boolean_mask(label, mask_area)

    return boxes, score, label