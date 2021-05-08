# coding: utf-8

from __future__ import division, print_function

import os
import shutil
import json
import cv2
import random
import numpy as np
import tensorflow as tf
from utils.data_augmentation import data_augmentation as my_data_augmentation
from utils.data_augmentation import data_augmentation_with_gray as my_data_augmentation_with_gray
import utils.globalvar as globalvar
from utils.eval_utils import calc_iou
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
jpeg = TurboJPEG()

#解析单行数据
def parse_line(line):

    if 'str' not in str(type(line)):
        line = line.decode()

    s = line.strip().split(' ')
    pic_path = s[0]
    s = s[1:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = s[i * 5], float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)

    boxes = np.asarray(boxes, np.float32)
    return pic_path, boxes, labels

#resize 图片尺寸，同时resize 框
def resize_image_and_correct_boxes(img, boxes, img_size):
    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:
        img = np.expand_dims(img, -1)
    ori_height, ori_width = img.shape[:2]
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))

    # convert to float
    img = np.asarray(img, np.float32)
    # print(boxes)
    # boxes
    # xmin, xmax
    if boxes.shape[0] > 0:
        boxes[:, 0] = boxes[:, 0] / ori_width * new_width
        boxes[:, 2] = boxes[:, 2] / ori_width * new_width
        # ymin, ymax
        boxes[:, 1] = boxes[:, 1] / ori_height * new_height
        boxes[:, 3] = boxes[:, 3] / ori_height * new_height

    return img, boxes

#数据扩充
def data_augmentation(img_path, img, boxes, label, img_size,  class_num_dic, train_with_gray, use_label_smothing = False):
    confs = None
    if train_with_gray:
        img, boxes, label, confs = my_data_augmentation_with_gray(img_path, img, boxes, label, img_size, class_num_dic, use_label_smothing)
    else:
        img, boxes, label, confs = my_data_augmentation(img_path, img, boxes, label, img_size, class_num_dic, use_label_smothing)

    return img, boxes, label, confs

#生成标注数据
def process_box(pic_path, img, boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    '''
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    if boxes.shape[0] > 0:
        box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
        box_sizes = boxes[:, 2:4] - boxes[:, 0:2]
    else:
        box_centers = None
        box_sizes = None

    # [13, 13, 3, 5+num_class]
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 5 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 5 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 5 + class_num), np.float32)

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    if boxes.shape[0] > 0:
        box_sizes = np.expand_dims(box_sizes, 1)
        # broadcast tricks
        # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
        mins = np.maximum(- box_sizes / 2, - anchors / 2)
        maxs = np.minimum(box_sizes / 2, anchors / 2)
        # [N, 9, 2]
        whs = maxs - mins

        # [N, 9]
        iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
        # [N]
        best_match_idx = np.argmax(iou, axis=1)

        ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
        for i, idx in enumerate(best_match_idx):
            # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 2
            feature_map_group = 2 - idx // 3
            # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
            ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
            x = int(np.floor(box_centers[i, 0] / ratio))
            y = int(np.floor(box_centers[i, 1] / ratio))
            k = anchors_mask[feature_map_group].index(idx)
            c = labels[i]
            # print feature_map_group, '|', y,x,k,c

            #y_true[feature_map_group][y, x, k, :2] = box_centers[i]
            #y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
            #y_true[feature_map_group][y, x, k, 4] = 1.
            # # # #y_true[feature_map_group][y, x, k, 4] = confs[i]
            #y_true[feature_map_group][y, x, k, 5+c] = 1.

            list_temp = [32., 16., 8.]
            for k in range(3):
                ratio = list_temp[k]
                x = int(np.floor(box_centers[i, 0] / ratio))
                y = int(np.floor(box_centers[i, 1] / ratio))
                # print(box_centers)
                # print(x, y)
                for m in range(3):
                    #
                    y_true[k][y, x, m, :2] = box_centers[i]
                    y_true[k][y, x, m, 2:4] = box_sizes[i]
                    y_true[k][y, x, m, 4] = 1.
                    y_true[k][y, x, m, 5 + c] = 1.

    return y_true_13, y_true_26, y_true_52

def adjust_data(img, boxes, labels, fill_zero_label_names):
    if len(boxes) > 0:
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2
        y_center = (boxes[:, 1] + boxes[:, 3]) / 2
        delete_ids = []
        for id in range(len(labels)):
            label = labels[id]
            # if label == "wxqy":
            if label in fill_zero_label_names:
                img[int(boxes[id][1]):int(boxes[id][3]), int(boxes[id][0]):int(boxes[id][2]), :] = 0
                for i in range(len(boxes)):
                    if boxes[id][0] <= x_center[i] <= boxes[id][2] and boxes[id][1] <= y_center[i] <= boxes[id][3]:
                        if i not in delete_ids:
                            delete_ids.append(i)
        boxes_temp = []
        labels_temp = []
        for id in range(len(labels)):
            if id not in delete_ids:
                boxes_temp.append(boxes[id])
                labels_temp.append(labels[id])

        boxes = np.asarray(boxes_temp, np.float32)
        labels = labels_temp
    else:
        pass

    return img, boxes, labels


def bgr2gray(bgr):
    gray = np.dot(bgr[..., :3], [0.114, 0.587, 0.299])
    gray = gray[:, :, np.newaxis]
    gray = np.tile(gray, [1, 1, 3])
    gray = np.asarray(gray, dtype=np.uint8)
    return gray


def read_img(pic_path, data_save_path_temp, nfs_mount_path):
    if 'str' not in str(type(data_save_path_temp)):
        data_save_path_temp = data_save_path_temp.decode()
    if 'str' not in str(type(nfs_mount_path)):
        nfs_mount_path = nfs_mount_path.decode()
    host_path = pic_path.replace(nfs_mount_path, data_save_path_temp)
    host_dir = os.path.dirname(host_path)

    if not os.path.exists(host_dir):
        try:
            os.makedirs(host_dir)
            globalvar.globalvar.logger.info_ai(meg="make dir:%s" % host_dir)
            # print("make dir:%s"%host_dir)
        except Exception as ex:
            globalvar.globalvar.logger.info_ai(meg="make dir:%s, find exception:%s" % (host_dir, ex))
            # print(ex)
            pass
    if not os.path.exists(host_path):
        try:
            shutil.copy(pic_path, host_path)
        except Exception as ex:
            globalvar.globalvar.logger.info_ai(meg="copy nfs img to local dir:%s, find exception:%s" % (pic_path, ex))
            # print(ex)
            pass
    try:
        in_file = open(host_path, 'rb')
        img = jpeg.decode(in_file.read())
        in_file.close()
    except Exception as ex:
        globalvar.globalvar.logger.info_ai(meg="TurboJPEG read img failed:%s, find exception:%s, so, use opencv to read" % (host_path, ex))
        # print(ex)
        # print(host_path)
        img = cv2.imread(host_path)

    return img


def parse_data(line, classes_list, class_num, img_size, anchors, mode, data_save_path_temp, nfs_mount_path, class_num_dic, train_with_gray, fill_zero_label_names):

    if 'str' not in str(type(class_num_dic)):
        class_num_dic = class_num_dic.decode()
    class_num_dic = json.loads(class_num_dic.replace("'", "\""))

    classes_list_temp = []
    for class_name in classes_list:
        classes_list_temp.append(class_name.decode())
    pic_path, boxes, labels = parse_line(line)

    img = read_img(pic_path, data_save_path_temp, nfs_mount_path)

    if img is None:
        globalvar.globalvar.logger.info_ai(meg="read img is None:%s"%pic_path)

    img, boxes, labels = adjust_data(img, boxes, labels, fill_zero_label_names)

    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    boxes_temp = []
    for box in boxes:
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(w, box[2])
        box[3] = min(h, box[3])
        if box[2] - box[0] < 5 or box[3] - box[1] < 5:
            pass
        else:
            boxes_temp.append(box)
    boxes = np.array(boxes_temp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # do data augmentation here
    if mode.decode() == 'train' and boxes.shape[0] > 0:
        img, boxes, labels, confs = data_augmentation(pic_path, img, boxes, labels, img_size, class_num_dic, train_with_gray, False)
    else:
        #img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
        img, boxes, labels, confs = data_augmentation(pic_path, img, boxes, labels, img_size, class_num_dic, train_with_gray, False)

    img = img.astype(np.float32)
    img = img / 255.
    y_true_13, y_true_26, y_true_52 = process_box(pic_path, img, boxes, labels, img_size, class_num, anchors)

    return img, y_true_13, y_true_26, y_true_52, pic_path


