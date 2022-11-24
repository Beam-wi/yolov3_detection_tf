# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf
slim = tf.contrib.slim

from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer, darknet53_body_prun, mobilenetv2, mobilenetv3, mobilenetv3_add_zoom_factor
from utils.focal_loss import focal_loss
import sys

class yolov3(object):

    def __init__(self, class_num, anchors, batch_norm_decay=0.9):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
                         # [30, 61], [62, 45], [59,  119],
                         # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }
        
        with slim.arg_scope([slim.conv2d, slim.batch_norm],reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')
                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list())
                    #
                    # inter1 = slim.conv2d(inter1, inter1.get_shape().as_list()[3], 3,
                    #             stride=1, biases_initializer=tf.zeros_initializer())
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list())

                    # inter2 = slim.conv2d(inter2, inter2.get_shape().as_list()[3], 3,
                    #                      stride=1, biases_initializer=tf.zeros_initializer())
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.shape.as_list()[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map 
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits


    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.shape.as_list()[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        
        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        # center_x = center_x * 2176/2720
        # center_y = center_y * 1459/1824
        # width = width * 2176/2720
        # height = height * 1459/1824

        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def predict_1(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.shape.as_list()[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        # center_x = center_x / 1.25
        # center_y = center_y / 1.25

        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def predict_2(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.shape.as_list()[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        center_x = center_x * 1.234
        center_y = center_y * 1.228

        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        '''
        
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)
        #pred_boxes,反馈到原始图片上的中心和长宽, conf,prob没有激活函数

        ###########
        # get mask
        ###########
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]  #是否含有目标的概率.只有目标在enchor中心,且iou最大的enchor为1,其余为0
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box

        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

        num_picture = tf.cast(tf.shape(feature_map_i)[0], tf.int32)

        def loop_body(id, ignore_mask):
            valid_true_boxes = tf.boolean_mask(y_true[id:id + 1][..., 0:4],
                                               tf.cast(object_mask[id:id + 1][..., 0], 'bool'))
            valid_true_box_xy = valid_true_boxes[:, 0:2]
            valid_true_box_wh = valid_true_boxes[:, 2:4]
            pred_box_xy = pred_boxes[id:id + 1][..., 0:2]
            pred_box_wh = pred_boxes[id:id + 1][..., 2:4]
            iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask_ = tf.cast(best_iou < 0.5, tf.float32)
            ignore_mask_ = tf.expand_dims(ignore_mask_, -1)
            ignore_mask = ignore_mask.write(id, ignore_mask_[0])
            return id + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda id, *args: id < num_picture, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()


        # valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))#真实的目标数据

        # shape: [V, 2]
        #valid_true_box_xy = valid_true_boxes[:, 0:2]
        #valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        #iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        #best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask
        #ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        #ignore_mask = tf.expand_dims(ignore_mask, -1) #anchor 里面是否含有物体,anchor和任意的object的iou大于0.5,表示含有物体

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N

        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask  #目标在anchor内,且iou最大的anchor为1,其余为0
        conf_neg_mask = (1 - object_mask) * ignore_mask #
        # conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_pos = conf_pos_mask * focal_loss(pred_conf_logits,  object_mask, alpha=0.25, gamma=2)
        conf_loss_neg = conf_neg_mask * focal_loss(pred_conf_logits,  object_mask, alpha=0.25, gamma=2)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # conf_loss = conf_loss_pos + conf_loss_neg
        # use_focal_loss = True
        # if use_focal_loss:
        #     alpha = 1.0
        #     gamma = 2.0
        #     # TODO: alpha should be a mask array if needed
        #     focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
        #     conf_loss *= focal_mask
        # conf_loss = tf.reduce_sum(conf_loss) / N

        # shape: [N, 13, 13, 3, 1]
        # class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        # class_loss = tf.reduce_sum(class_loss) / N
        class_loss_pos = conf_pos_mask * focal_loss(pred_prob_logits, y_true[..., 5:], alpha=0.25, gamma=2)
        # class_loss_neg = conf_neg_mask * focal_loss(pred_prob_logits, y_true[..., 5:], alpha=0.25, gamma=2)
        # class_loss = tf.reduce_sum(class_loss_pos + class_loss_neg) / N
        class_loss = tf.reduce_sum(class_loss_pos) / N

        # xy_loss = tf.clip_by_value(xy_loss, -100.0, 100.0)
        # wh_loss = tf.clip_by_value(wh_loss, -100.0, 100.0)
        # conf_loss = tf.clip_by_value(conf_loss, -100.0, 100.0)
        # class_loss = tf.clip_by_value(class_loss, -100.0, 100.0)
        return xy_loss, wh_loss, conf_loss * 100.0, class_loss * 100.0

    
    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]



    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
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


class yolov3_trt(object):

    def __init__(self, class_num, anchors, backbone_name, train_with_two_feature_map=False, batch_norm_decay=0.9):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
        # [30, 61], [62, 45], [59,  119],
        # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        if train_with_two_feature_map:
            self.anchors = self.anchors[:6]
        self.backbone_name = backbone_name
        self.batch_norm_decay = batch_norm_decay
        self.train_with_two_feature_map = train_with_two_feature_map
        self.rgb_factor = tf.constant([[[[0.2989, 0.5870, 0.1140]]]]) #rgb转灰度图转换因子

    def forward(self, inputs, is_training=False, train_with_gray=True, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.relu6(x)):
                with tf.variable_scope('darknet53_body'):
                    # 转为灰度图
                    if train_with_gray:
                        inputs = inputs * self.rgb_factor
                        inputs = tf.reduce_sum(inputs, axis=-1)
                        inputs = tf.expand_dims(inputs, -1)
                        inputs = tf.tile(inputs, [1, 1, 1, 3])
                    if self.backbone_name == "darknet53":
                        routes = darknet53_body(inputs, self.train_with_two_feature_map)
                    elif self.backbone_name == "darknet53_prun":
                        routes = darknet53_body_prun(inputs, self.train_with_two_feature_map)
                    elif self.backbone_name == "mobilenetv2":
                        routes = mobilenetv2(inputs, self.train_with_two_feature_map, is_training)
                    elif self.backbone_name == "mobilenetv3":
                        routes = mobilenetv3(inputs, self.train_with_two_feature_map, is_training)
                    elif self.backbone_name == "mobilenetv3_add_zoom_factor":
                        routes = mobilenetv3_add_zoom_factor(inputs, self.train_with_two_feature_map, is_training)
                    else:
                        print("backbone name is not right, it is mast in [darknet53, darknet53_prun, mobilenetv2, mobilenetv3, mobilenetv3_add_zoom_factor]")
                        sys.exit()

                with tf.variable_scope('yolov3_head'):
                    if not self.train_with_two_feature_map:
                        route_1, route_2, route_3 = routes  # (?, 128, 128, 256) (?, 64, 64, 512) (?, 32, 32, 1024)
                        inter1, net = yolo_block(route_3, 512)  # (?, 32, 32, 512) (?, 32, 32, 1024)
                        feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                    stride=1, normalizer_fn=None,
                                                    activation_fn=None, biases_initializer=tf.zeros_initializer())  # (?, 32, 32, 18)
                        feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')  # (?, 32, 32, 18)
                        inter1 = conv2d(inter1, 256, 1)  # (?, 32, 32, 256)
                        inter1 = upsample_layer(inter1, route_2.get_shape().as_list())  # (?, 64, 64, 256)
                        #
                        # inter1 = slim.conv2d(inter1, inter1.get_shape().as_list()[3], 3,
                        #             stride=1, biases_initializer=tf.zeros_initializer())
                        concat1 = tf.concat([inter1, route_2], axis=3)  # (?, 64, 64, 768)

                        inter2, net = yolo_block(concat1, 256)  # (?, 64, 64, 256) (?, 64, 64, 512)
                        feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                    stride=1, normalizer_fn=None,
                                                    activation_fn=None, biases_initializer=tf.zeros_initializer())  # (?, 64, 64, 18)
                        feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')  # (?, 64, 64, 18)

                        inter2 = conv2d(inter2, 128, 1)  # (?, 64, 64, 128)
                        inter2 = upsample_layer(inter2, route_1.get_shape().as_list())  # (?, 128, 128, 128)

                        # inter2 = slim.conv2d(inter2, inter2.get_shape().as_list()[3], 3,
                        #                      stride=1, biases_initializer=tf.zeros_initializer())
                        concat2 = tf.concat([inter2, route_1], axis=3)  # (?, 128, 128, 384)

                        _, feature_map_3 = yolo_block(concat2, 128)  # (?, 128, 128, 128) (?, 128, 128, 256)
                        feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                    stride=1, normalizer_fn=None,
                                                    activation_fn=None, biases_initializer=tf.zeros_initializer())  # (?, 128, 128, 18)
                        feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')  # (?, 128, 128, 18)

                        return feature_map_1, feature_map_2, feature_map_3  #   # (?, 32, 32, 18) (?, 64, 64, 18) (?, 128, 128, 18)
                    else:
                        route_1, route_2 = routes
                        inter2, net = yolo_block(route_2, 256)
                        feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                    stride=1, normalizer_fn=None,
                                                    activation_fn=None, biases_initializer=tf.zeros_initializer())
                        feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                        inter2 = conv2d(inter2, 128, 1)
                        inter2 = upsample_layer(inter2, route_1.get_shape().as_list())

                        # inter2 = slim.conv2d(inter2, inter2.get_shape().as_list()[3], 3,
                        #                      stride=1, biases_initializer=tf.zeros_initializer())
                        concat2 = tf.concat([inter2, route_1], axis=3)

                        _, feature_map_3 = yolo_block(concat2, 128)
                        feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                    stride=1, normalizer_fn=None,
                                                    activation_fn=None, biases_initializer=tf.zeros_initializer())
                        feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

                        return feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.shape.as_list()[1:3]  # [32, 32]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)  # (2,)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        # box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = feature_map[:, :, :, :, 0:2]  # (?, 32, 32, 3, 2)
        box_sizes = feature_map[:, :, :, :, 2:4]  # (?, 32, 32, 3, 2)
        conf_logits = feature_map[:, :, :, :, 4:5]  # (?, 32, 32, 3, 1)
        prob_logits = feature_map[:, :, :, :, 5:5+self.class_num]  # (?, 32, 32, 3, 1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)  # (32,)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)  # (32,)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)  # (32, 32) (32, 32)
        x_offset = tf.reshape(grid_x, (-1, 1))  # (1024, 1)
        y_offset = tf.reshape(grid_y, (-1, 1))  # (1024, 1)
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  # (1024, 2)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)  # (32, 32, 1, 2)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset  # (?, 32, 32, 3, 2)
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        # boxes = tf.concat([box_centers, box_sizes], axis=-1)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]  步长下标
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        if self.train_with_two_feature_map:
            feature_map_2, feature_map_3 = feature_maps
            feature_map_anchors = [
                                   (feature_map_2, self.anchors[3:6]),
                                   (feature_map_3, self.anchors[0:3])]
        else:
            feature_map_1, feature_map_2, feature_map_3 = feature_maps
            feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                                   (feature_map_2, self.anchors[3:6]),
                                   (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.shape.as_list()[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        # center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        center_x = boxes[:, :, 0:1]
        center_y = boxes[:, :, 1:2]
        width = boxes[:, :, 2:3]
        height = boxes[:, :, 3:4]

        # center_x = center_x * 2176/2720
        # center_y = center_y * 1459/1824
        # width = width * 2176/2720
        # height = height * 1459/1824

        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        '''

        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)  # [13, 13, 1, 2]步长下标  [N, 13, 13, 3, 4] [N, 13, 13, 3, 1] [N, 13, 13, 3, class_num]
        # pred_boxes,反馈到原始图片上的中心和长宽, conf,prob没有激活函数

        ###########
        # get mask
        ###########
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]  # 是否含有目标的概率.只有目标在enchor中心,且iou最大的enchor为1,其余为0
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box

        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

        num_picture = tf.cast(tf.shape(feature_map_i)[0], tf.int32)  # ()

        def loop_body(id, ignore_mask):
            valid_true_boxes = tf.boolean_mask(y_true[id:id + 1][..., 0:4],  # y_true[id:id + 1][..., 0:4]: (1, 32, 32, 3, 4)
                                               tf.cast(object_mask[id:id + 1][..., 0], 'bool'))  # (1, 32, 32, 3, 1) 返回 (m, 4) m为置信度为True的那个像素点个数
            valid_true_box_xy = valid_true_boxes[:, 0:2]  # (m, 2)
            valid_true_box_wh = valid_true_boxes[:, 2:4]  # (m, 2)
            pred_box_xy = pred_boxes[id:id + 1][..., 0:2]  # (1, 32, 32, 3, 2)
            pred_box_wh = pred_boxes[id:id + 1][..., 2:4]  # (1, 32, 32, 3, 2)
            iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)  # (1, 32, 32, 3, m) 把预测的每个像素点都与存在的真实box算iou
            best_iou = tf.reduce_max(iou, axis=-1)  # (1, 32, 32, 3) 每个像素找到最大iou的那个真实box对应下标
            ignore_mask_ = tf.cast(best_iou < 0.5, tf.float32)  # (1, 32, 32, 3)
            ignore_mask_ = tf.expand_dims(ignore_mask_, -1)  # (1, 32, 32, 3, 1)
            ignore_mask = ignore_mask.write(id, ignore_mask_[0])
            return id + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda id, *args: id < num_picture, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()  # (?, 32, 32, 3, 1)

        # valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))#真实的目标数据

        # shape: [V, 2]
        # valid_true_box_xy = valid_true_boxes[:, 0:2]
        # valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        # iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        # best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask
        # ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        # ignore_mask = tf.expand_dims(ignore_mask, -1) #anchor 里面是否含有物体,anchor和任意的object的iou大于0.5,表示含有物体

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset  # (N, 32, 32, 3, 2)
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset  # (N, 32, 32, 3, 2)

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                    y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N

        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask  # 目标在anchor内,且iou最大的anchor为1,其余为0
        conf_neg_mask = (1 - object_mask) * ignore_mask  #
        # conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_pos = conf_pos_mask * focal_loss(pred_conf_logits, object_mask, alpha=0.25, gamma=2)
        conf_loss_neg = conf_neg_mask * focal_loss(pred_conf_logits, object_mask, alpha=0.25, gamma=2)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # conf_loss = conf_loss_pos + conf_loss_neg
        # use_focal_loss = True
        # if use_focal_loss:
        #     alpha = 1.0
        #     gamma = 2.0
        #     # TODO: alpha should be a mask array if needed
        #     focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
        #     conf_loss *= focal_mask
        # conf_loss = tf.reduce_sum(conf_loss) / N

        # shape: [N, 13, 13, 3, 1]
        # class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        # class_loss = tf.reduce_sum(class_loss) / N
        class_loss_pos = conf_pos_mask * focal_loss(pred_prob_logits, y_true[..., 5:], alpha=0.25, gamma=2)
        # class_loss_neg = conf_neg_mask * focal_loss(pred_prob_logits, y_true[..., 5:], alpha=0.25, gamma=2)
        # class_loss = tf.reduce_sum(class_loss_pos + class_loss_neg) / N
        class_loss = tf.reduce_sum(class_loss_pos) / N

        # xy_loss = tf.clip_by_value(xy_loss, -100.0, 100.0)
        # wh_loss = tf.clip_by_value(wh_loss, -100.0, 100.0)
        # conf_loss = tf.clip_by_value(conf_loss, -100.0, 100.0)
        # class_loss = tf.clip_by_value(class_loss, -100.0, 100.0)
        return xy_loss, wh_loss, conf_loss * 100.0, class_loss * 100.0

    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        if not self.train_with_two_feature_map:
            anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]
        else:
            anchor_group = [self.anchors[3:6], self.anchors[0:3]]

        if self.train_with_two_feature_map:
            y_true = y_true[1:]
        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
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


class darknet_plus(object):

    def __init__(self, class_num, batch_norm_decay=0.9):
        self.batch_norm_decay = batch_norm_decay
        self.class_num = class_num

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.relu6(x)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    downsample_1 = conv2d(route_1, 128, 3, strides=2)
                    concat1 = tf.concat([downsample_1, route_2], axis=3)
                    downsample_2 = conv2d(concat1, 128, 3, strides=2)
                    concat2 = tf.concat([downsample_2, route_3], axis=3)
                    out_1 = conv2d(concat2, 128, 3, strides=2)
                    out_2 = conv2d(out_1, 128, 3, strides=1)
                    out_3 = conv2d(out_2, 128, 3, strides=1)
                    out_4 = tf.layers.flatten(out_3)
                    out_5 = tf.layers.dense(out_4, 128, activation=lambda x: tf.nn.relu6(x),
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(), use_bias=True)

                    logits_output = tf.layers.dense(out_5, self.class_num,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    bias_initializer=tf.zeros_initializer(), use_bias=False)

            # return logits_output, out_5, route_1, out_1, downsample_1, concat1, downsample_2, concat2, route_2, route_3
            return logits_output, out_5

