# coding: utf-8

# from __future__ import division, print_function

import tensorflow as tf
import numpy as np
from utils.nms_utils import gpu_nms

import cv2
import time
import json
from utils.model import darknet_plus, yolov3_trt


def inference_one_img_cls(model_path, image_path, classNum):
    tf.reset_default_graph()
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, 192, 192, 3],
                                    name='input_data')
        yolo_model_classfy = darknet_plus(classNum)
        with tf.variable_scope('yolov3_classfication'):
            logits, center_feature, route_1, out_1, downsample_1, concat1, downsample_2, concat2, route_2, route_3 = yolo_model_classfy.forward(input_data,
                                                                is_training=False)  # (1, 5) (1, 128)

        saver = tf.train.Saver(
            var_list=tf.contrib.framework.get_variables_to_restore(
                include=["yolov3_classfication"]))
        # saver.restore(sess, "Q:/ai_model_ckpt/sjht/20200804_model_v4.5/sjht_common/yolov3_model/20200802_sjht_common_33_57lassnum.ckpt")
        saver.restore(sess, model_path)

        data_dict = dict()
        path = image_path
        img_ori = cv2.imread(path)
        img = cv2.resize(img_ori, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        print(img[:, 0, :, :])

        logits_, center_feature_, route_1_, out_1_, downsample_1, concat1_, downsample_2_, concat2_, route_2_, route_3_ = sess.run(
            [logits, center_feature, route_1, out_1, downsample_1, concat1, downsample_2, concat2, route_2, route_3],
            feed_dict={
                input_data: img})  # (1, 145152) (1, 36288, 1) (1, 36288, 1)
        data_dict["logits_"] = logits_.tolist()
        data_dict["center_feature_"] = center_feature_.tolist()
        with open("./data/datadict_cls.json", "w", encoding="utf-8") as file:
            json.dump(data_dict, file)
        print("finish!!!")


def inference_one_img_det(model_path, image_path, num_class, anchors):
    tf.reset_default_graph()
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, 2016, 3008, 3],
                                    name='input_data')
        yolo_model = yolov3_trt(num_class, anchors, "darknet53")
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(
            pred_feature_maps)

        # pred_scores = pred_confs * pred_probs
        # boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class,
        #                                 max_boxes=150, score_thresh=0.1,
        #                                 nms_thresh=0.2)

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        data_dict = dict()
        path = image_path
        img_ori = cv2.imread(path)
        img = cv2.resize(img_ori, (3008, 2016))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        print(img[:, 0, :, :])
        while True:
            start = time.time()
            pred_boxes_, pred_confs_, pred_probs_ = sess.run(
                [pred_boxes, pred_confs, pred_probs],
                feed_dict={
                    input_data: img})  # (1, 145152) (1, 36288, 1) (1, 36288, 1)
            print(f"time loss:{time.time() - start}")

            # data_dict["pred_boxes"] = pred_boxes_.tolist()
            # data_dict["pred_confs"] = pred_confs_.tolist()
            # data_dict["pred_probs"] = pred_probs_.tolist()
            #
            # with open("./data/datadict_det_w.json", "w", encoding="utf-8") as file:
            #     json.dump(data_dict, file)
            # print("finish!!!")


if __name__ == '__main__':
    # # cls
    # model_path = '/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/ckpt/20201209_whht_v1.0_1000_5cassnum.ckpt'
    # model_path = '/mnt/data/binovo/data/model/sjht-lzbl-271/ckpt_model/cls/202107121918_classify_model_default_name'
    model_path = '/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/lzbl/sjht-lzbl-271/1.0/model_and_temp_file/classify_model/202107121918_classify_model_default_name'
    # image_path = '/mnt/data/bihua/LocalProject/yolo3Tensorflow/camera1_2020-12-23_07_53_02_441848.jpg'
    # image_path = '/mnt/data/opt/data/ai-product-injection-mold-inserts/0_1626425172.0611386_test.jpg'
    image_path = '/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/lzbl/sjht-lzbl-271/1.0/model_and_temp_file/combine_test_result/lzbl/wrong_full/0_1626435133.4139924_test.png'
    classNum = 6
    inference_one_img_cls(model_path, image_path, classNum=classNum)

    # det
    # model_path = '/mnt/data/binovo/data/model/sjht-lzbl-271/ckpt_model/det/202107121441_detection_model_default_name'
    # image_path = '/mnt/data/opt/data/ai-product-injection-mold-inserts/0_1626435133.4139924_test.png'
    # num_class = 1
    # anchors = np.reshape(np.asarray([15.00, 30.00, 19.00, 19.00, 30.00, 15.00, 25.00, 50.00, 36.00, 36.00, 50.00, 25.00, 43.00, 86.00, 60.00, 60.00, 86.00, 43.00], np.float32), [-1, 2]) * 2.5
    # inference_one_img_det(model_path, image_path, num_class, anchors)
