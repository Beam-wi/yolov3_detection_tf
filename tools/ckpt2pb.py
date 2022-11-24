# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import argparse
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.model import darknet_plus, yolov3_trt
from utils.model import darknet_plus


def det_ckpt2pb(model_path, export_path, num_class, anchors,
                img_size=[1, 2016, 3008, 3], use_predict=True):
    tf.reset_default_graph()
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, img_size, name='input_data')
        input_shape = tf.placeholder(tf.float32, [2], name='feature_shape')
        feature_shape = tf.placeholder(tf.float32, [2], name='feature_shape')
        # x_rang = tf.placeholder(tf.float32, [img_size[2]], name='x_rang')
        # y_rang = tf.placeholder(tf.float32, [img_size[1]], name='y_rang')
        # iou_thresh = tf.placeholder(tf.float32, [], name='iou_thresh')
        # score_thresh = tf.placeholder(tf.float32, [], name='score_thresh')
        yolo_model = yolov3_trt(num_class, anchors, 'darknet53')
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
            # pred_feature_maps = yolo_model.forward(input_data, input_shape, False)
        if use_predict:
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(
                pred_feature_maps)

        # pred_scores = pred_confs * pred_probs
        # boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class,
        #                                 max_boxes=150, score_thresh=0.1,
        #                                 nms_thresh=0.2)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        if not use_predict:
            feature_map_1, feature_map_2, feature_map_3 = pred_feature_maps
            print(feature_map_1.shape)
            print(feature_map_2.shape)
            print(feature_map_3.shape)

            # 变量及输出的 tensor 信息
            inp0 = tf.saved_model.utils.build_tensor_info(input_data)
            # inp1 = tf.saved_model.utils.build_tensor_info(input_shape)
            # inp2 = tf.saved_model.utils.build_tensor_info(feature_shape)
            # inp3 = tf.saved_model.utils.build_tensor_info(x_rang)
            # inp4 = tf.saved_model.utils.build_tensor_info(y_rang)
            # inp1 = tf.saved_model.utils.build_tensor_info(input1)
            out0 = tf.saved_model.utils.build_tensor_info(feature_map_1)  # name="feature_map_1"
            out1 = tf.saved_model.utils.build_tensor_info(feature_map_2)  # name="feature_map_2"
            out2 = tf.saved_model.utils.build_tensor_info(feature_map_3)  # name="feature_map_3"

        else:
            print(pred_boxes.shape)
            print(pred_confs.shape)
            print(pred_probs.shape)

            boxes_result = tf.identity(pred_boxes, name='boxes_result')
            confs_result = tf.identity(pred_confs, name='confs_result')
            probs_result = tf.identity(pred_probs, name='probs_result')

            # 变量及输出的 tensor 信息
            inp0 = tf.saved_model.utils.build_tensor_info(input_data)
            # inp1 = tf.saved_model.utils.build_tensor_info(input_shape)
            # inp2 = tf.saved_model.utils.build_tensor_info(feature_shape)
            # inp3 = tf.saved_model.utils.build_tensor_info(x_rang)
            # inp4 = tf.saved_model.utils.build_tensor_info(y_rang)
            # inp1 = tf.saved_model.utils.build_tensor_info(input1)
            out0 = tf.saved_model.utils.build_tensor_info(boxes_result)
            out1 = tf.saved_model.utils.build_tensor_info(confs_result)
            out2 = tf.saved_model.utils.build_tensor_info(probs_result)
        # 输入输出签名
        sign = tf.saved_model.signature_def_utils.build_signature_def(
            # inputs={'input0': inp0, 'input1': inp1, 'input2': inp2, 'input3': inp3, 'input4': inp4},
            inputs={'input0': inp0},
            outputs={'output0': out0, 'output1': out1, 'output2': out2, },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # 模型要被 tensorrtserver 运行，必须以如下的方式保存
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: sign},
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        builder.save()
        print("export end!!")


def cls_ckpt2pb(model_path, export_path, class_num, img_size=(1, 192, 192, 3)):
    tf.reset_default_graph()
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, img_size, name='input_data')
        yolo_model_classfy = darknet_plus(class_num)
        with tf.variable_scope('yolov3_classfication'):
            logits, center_feature = yolo_model_classfy.forward(input_data,
                                                                is_training=False)  # (1, 5) (1, 128)

        saver = tf.train.Saver(
            var_list=tf.contrib.framework.get_variables_to_restore(
                include=["yolov3_classfication"]))
        # saver.restore(sess, "Q:/ai_model_ckpt/sjht/20200804_model_v4.5/sjht_common/yolov3_model/20200802_sjht_common_33_57lassnum.ckpt")
        saver.restore(sess, model_path)

        print(logits.shape)
        print(center_feature.shape)

        class_result = tf.identity(logits, name='class_result')
        feature_result = tf.identity(center_feature, name='feature_result')

        # 变量及输出的 tensor 信息
        inp = tf.saved_model.utils.build_tensor_info(input_data)
        out0 = tf.saved_model.utils.build_tensor_info(class_result)
        out1 = tf.saved_model.utils.build_tensor_info(feature_result)
        # 输入输出签名
        sign = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': inp},
            outputs={'output0': out0, 'output1': out1},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        # builder = tf.saved_model.builder.SavedModelBuilder("/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/pb/pb")
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # 模型要被 tensorrtserver 运行，必须以如下的方式保存
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: sign},
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        builder.save()
        print("export end!!")


if __name__ == '__main__':
    # det
    # # inserts
    # model_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210311_dy/ckpt/20210105_whht_v1.2_model-step_22000_loss_28.189247_lr_5e-07"
    # export_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210311_dy/pb/pb_5"
    # class_name_path = "yolo3Tensorflow/data/inserts/sjht_classnames_big_class.txt"
    # anchor_path = "yolo3Tensorflow/data/inserts/sjht_anchors.txt"
    # num_class = len(read_class_names(class_name_path))
    # anchors = parse_anchors(anchor_path)
    # img_size = [None, None, None, 3]
    # # img_size = [1, 1600, 2048, 3]
    # det_ckpt2pb(model_path, export_path, num_class, anchors, img_size=img_size, use_predict=False)

    model_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/ckpt_model/det/4/202107201517_detection_model_default_name"
    export_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/pb/det/4"
    num_class = 1
    anchors = np.reshape(np.asarray([15.00, 30.00, 19.00, 19.00, 30.00, 15.00, 25.00, 50.00, 36.00, 36.00, 50.00, 25.00, 43.00, 86.00, 60.00, 60.00, 86.00, 43.00], np.float32), [-1, 2]) * 2.5
    img_size = (1, 2016, 3008, 3)
    # img_size = [1, 1600, 2048, 3]
    det_ckpt2pb(model_path, export_path, num_class, anchors, img_size=img_size, use_predict=True)


    # cls
    # # model_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20210317/ckpt/20201209_whht_v1.0_1000_5cassnum.ckpt"
    # # export_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20210317/pb/16"
    # # img_size = (1, 2016, 3008, 3)
    # model_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/ckpt_model/cls/4/202107191741_classify_model_default_name"
    # export_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/pb/cls/4"
    # img_size = (1, 192, 192, 3)
    # class_num = 6
    # cls_ckpt2pb(model_path, export_path, class_num, img_size=img_size)


"""
workpieces
--anchor_path
./data/workpieces/sjht_anchors.txt
--new_size
(768,768)
--class_name_path
./data/workpieces/sjht_num_class.txt
--restore_path
/mnt/data/bihua/data/model/yolo3Tensorflow/ckpt/SJHT_1211_model-epoch_72_step_537206_loss_0.1375_lr_5.6467e-06
--save_path
/mnt/data/bihua/data/model/yolo3Tensorflow/pb/pb


inserts
--anchor_path
./data/inserts/sjht_anchors.txt
--new_size
(3008,2016)
--class_name_path
./data/inserts/sjht_classnames_big_class.txt
--restore_path
/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210105/ckpt/20210105_whht_v1.2_model-step_22000_loss_28.189247_lr_5e-07
--save_path
/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210105/pb/pb
"""
