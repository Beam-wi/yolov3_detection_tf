from __future__ import division, print_function
import csv
import random
import argparse
import time
import sys
import cv2
import os
import tensorflow as tf
from utils.config_parse import get_config
import numpy as np
from utils.eval_utils import calc_iou
from collections import Counter
from utils.get_data_from_web import get_data_from_web_and_save
from utils.data_utils import parse_line, adjust_data
from utils.logging_util import Logger
import utils.globalvar as globalvar
from utils.test_utils import qianjian_detection, show_box

project_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == '__main__':
    #################
    # ArgumentParser
    #################
    parser = argparse.ArgumentParser(description="YOLO-V3 training procedure.")
    parser.add_argument("--config_project", type=str, default="./data/config/config_project.yaml",
                        help="项目配置文件.")
    parser.add_argument("--config_common", type=str, default="./data/config/config_common.yaml",
                        help="默认配置文件.")
    args = parser.parse_args()

    config_common = get_config(args.config_common)
    config_project = get_config(args.config_project).as_dict()
    config_common.update(config_project)

    #创建部分存储路径
    if not os.path.exists(config_common.data_set["log_save_path"]):
        try:
            os.makedirs(config_common.data_set["log_save_path"])
            print("make log save dir:%s"%config_common.data_set["log_save_path"])
        except Exception as ex:
            print("make log save dir:%s, failed, cause of %s" % (config_common.data_set["log_save_path"], ex))
            sys.exit(1)
    #创建日志文件
    logger = Logger(config_common.data_set["log_save_path"], "model_test")
    logger.clean_log_dir()

    logger.info_ai(meg="project name is:%s"%config_common.project_name)
    logger.info_ai(meg="config info", get_ins={"config": config_common})

    if not os.path.exists(config_common.data_set["test_result_save_path"]):
        try:
            os.makedirs(config_common.data_set["test_result_save_path"])
            logger.info_ai(meg="make test result data save dir:%s"%config_common.data_set["test_result_save_path"])
        except Exception as ex:
            logger.info_ai(meg="make test result data save dir:%s, failed, cause of %s" % (config_common.data_set["test_result_save_path"], ex))
            sys.exit(1)

    if not os.path.exists(config_common.data_set["wrong_label_save_path"]):
        try:
            os.makedirs(config_common.data_set["wrong_label_save_path"])
            logger.info_ai(meg="make wrong label save dir:%s"%config_common.data_set["wrong_label_save_path"])
        except Exception as ex:
            logger.info_ai(meg="make wrong label save dir:%s, failed, cause of %s" % (config_common.data_set["wrong_label_save_path"], ex))
            sys.exit(1)

    #设置全局变量
    globalvar.set_logger(logger)
    globalvar.set_config(config_common)

    #判断必须文件是否存在
    assert os.path.exists(config_common.data_set["train_file_path"]), "train_file_path not exit:%s"%config_common.data_set["train_file_path"]
    assert os.path.exists(config_common.data_set["val_file_path"]), "val_file_path not exit:%s"%config_common.data_set["val_file_path"]
    assert os.path.exists(config_common.data_set["nfs_mount_path"]), "nfs_mount_path not exit:%s"%config_common.data_set["nfs_mount_path"]
    assert os.path.exists(config_common.data_set["anchor_path"]), "anchor_path not exit:%s"%config_common.data_set["anchor_path"]
    assert os.path.exists(config_common.data_set["detection_class_name_path"]), "detection_class_name_path not exit:%s"%config_common.data_set["detection_class_name_path"]
    assert os.path.exists(config_common.data_set["class_name_path"]), "class_name_path not exit:%s"%config_common.data_set["class_name_path"]
    assert os.path.exists(config_common.data_set["log_save_path"]), "log_save_path not exit:%s"%config_common.data_set["log_save_path"]
    assert os.path.exists(config_common.data_set["model_save_path"]), "model_save_path not exit:%s"%config_common.data_set["model_save_path"]
    assert os.path.exists(config_common.data_set["classify_model_save_path"]), "classify_model_save_path not exit:%s"%config_common.data_set["classify_model_save_path"]
    assert os.path.exists(config_common.data_set["test_result_save_path"]), "test_result_save_path not exit:%s"%config_common.data_set["test_result_save_path"]
    #设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = config_common.model_set["gpu_device"]
    gpus = []
    for id in config_common.model_set["gpu_device"].strip().split(","):
        gpus.append("/gpu:%s"%id)

    gpu_device = gpus

    try:
    # if True:
        #准备数据，如果提供了数据批次，则以数据批次数据为准，自动拉取数据集
        if config_common.data_set["train_batch_path"] != "None":
            logger.info_ai(meg="data batch is not None, get new data from web")
            train_num, val_num, class_num_dic = get_data_from_web_and_save(data_batch_path=config_common.data_set["train_batch_path"], data_train_save_path=config_common.data_set["train_file_path"],
                                                                           data_val_save_path=config_common.data_set["val_file_path"], nfs_mount_path=config_common.data_set["nfs_mount_path"],
                                                                           config=config_common, logger=logger, project_name="sjht")
        else:
            logger.info_ai(meg="data batch is None, use train file and val file")
            data_train = open(config_common.data_set["train_file_path"]).readlines()
            train_num = len(data_train)
            data_val = open(config_common.data_set["val_file_path"]).readlines()
            val_num = len(data_val)
            # 储存每个类别嵌件个数，以便后续裁剪图片时按概率裁剪
            class_num_dic = {}
            for line in data_val:
                pic_path, boxes, labels = parse_line(line)
                for label in labels:
                    if label not in config_common.data_set["fill_zero_label_names"] and label != "gj":
                        if label not in class_num_dic:
                            class_num_dic[label] = 1
                        else:
                            class_num_dic[label] += 1

            logger.info_ai(meg="prepare data over, get all class name from data file", get_ins={"class_num_dic": class_num_dic})

        detection_scores = list(np.arange(0.05, 0.99, 0.05))
        class_scores = list(np.arange(0.05, 0.99, 0.05))
        if config_common.test_other_info_set["show_wrong_data"]:
            detection_scores = config_common.test_other_info_set["detection_threshold_for_show"]
            class_scores = config_common.test_other_info_set["classify_threshold_for_show"]

        detection_class_score = {}
        for detection_score in detection_scores:
            for class_score in class_scores:
                key = "%f_%f"%(detection_score, class_score)
                value = [detection_score, class_score]
                detection_class_score[key] = value

        #如果设置多个gpu推断，选择第一个gpu进行测试
        with tf.device(gpu_device[0]):
            qianjian_detection = qianjian_detection()
            if len(qianjian_detection.class_name_classify) == 1:
                config_common.test_other_info_set["do_classify"] = False
            try:
                qianjian_detection.model_init_detection()
            except Exception as ex:
                logger.info_ai(meg="initialize detection model failed, cause of %s" % (ex))
                sys.exit(1)

            if config_common.test_other_info_set["do_classify"]:
                try:
                    qianjian_detection.model_init_classify()
                except Exception as ex:
                    logger.info_ai(meg="initialize classify model failed, cause of %s" % (ex))
                    sys.exit(1)


            test_data = open(config_common.data_set["val_file_path"]).readlines()
            logger.info_ai(meg="test data num:%d"%(len(test_data)))

            multi_category_dict = {}
            for key in list(detection_class_score.keys()):
                if not config_common.test_other_info_set["do_classify"]:
                    num_class = qianjian_detection.num_class_detection
                    true_labels_dict = {qianjian_detection.class_name_detection[i]: 0 for i in range(num_class)}  # {class: count}
                    pred_labels_dict = {qianjian_detection.class_name_detection[i]: 0 for i in range(num_class)}  # 计算每个类别识别个数
                    true_positive_dict = {qianjian_detection.class_name_detection[i]: 0 for i in range(num_class)}
                    wrong_classes_dict = {qianjian_detection.class_name_detection[i]: 0 for i in range(num_class)}
                    find_classes_dict = {qianjian_detection.class_name_detection[i]: 0 for i in range(num_class)}
                else:
                    num_class = qianjian_detection.num_class_classify
                    true_labels_dict = {qianjian_detection.class_name_classify[i]: 0 for i in range(num_class)}  # {class: count}
                    pred_labels_dict = {qianjian_detection.class_name_classify[i]: 0 for i in range(num_class)}  # 计算每个类别识别个数
                    true_positive_dict = {qianjian_detection.class_name_classify[i]: 0 for i in range(num_class)}
                    wrong_classes_dict = {qianjian_detection.class_name_classify[i]: 0 for i in range(num_class)}
                    find_classes_dict = {qianjian_detection.class_name_classify[i]: 0 for i in range(num_class)}

                value = {"true_labels_dict":true_labels_dict, "pred_labels_dict":pred_labels_dict, "true_positive_dict":true_positive_dict, "wrong_classes_dict":wrong_classes_dict, "find_classes_dict":find_classes_dict}
                multi_category_dict[key] = value
            test_id = 0
            wrong_num = 0
            for line in test_data:
                test_id = test_id + 1
                logger.info_ai(meg="current test id:%d, all test num:%d"%(test_id, len(test_data)))

                img_test_path, boxes, labels = parse_line(line)
                if not os.path.exists(img_test_path):
                    logger.info_ai(meg="img file is not exit:%s"%img_test_path)
                    wrong_num = wrong_num + 1
                    continue
                img_ori = cv2.imread(img_test_path)
                img_h, img_w, c = img_ori.shape
                img_adjust, boxes, labels = adjust_data(img_ori, boxes, labels, config_common.data_set["fill_zero_label_names"])

                if not set(labels).issubset(set(qianjian_detection.name2id_classify.keys())):
                    logger.info_ai(meg="these labels not in class names: %s, in the file:%s"%(str(set(labels) - set(qianjian_detection.name2id_classify.keys())), img_test_path))
                    wrong_num = wrong_num + 1
                    continue

                boxes_detection, scores_detection, labels_detection_name, scores_classify, labels_classify_name = qianjian_detection.forward(img_adjust, do_classify=config_common.test_other_info_set["do_classify"])


                if not config_common.test_other_info_set["do_classify"]:
                    labels = ["qj"] * len(labels)
                else:
                    labels_temp = []
                    for label in labels:
                        label_temp = qianjian_detection.class_name_classify[qianjian_detection.name2id_classify[label]]
                        labels_temp.append(label_temp)
                    labels = labels_temp
                boxes = np.array(boxes, dtype=np.float32)

                for key, value in detection_class_score.items():
                    select_id = []
                    for idx in range(len(boxes_detection)):
                        if scores_detection[idx] >= value[0] and scores_classify[idx] >= value[1]:
                            select_id.append(idx)
                        else:
                            pass

                    boxes_detection_select = boxes_detection[select_id]
                    scores_detection_select = scores_detection[select_id]
                    labels_detection_name_select = labels_detection_name[select_id]
                    scores_classify_select = scores_classify[select_id]
                    labels_classify_name_select = labels_classify_name[select_id]

                    pred_boxes = boxes_detection_select
                    true_boxes = boxes
                    true_labels_list = labels
                    pred_confs = scores_detection_select

                    if config_common.test_other_info_set["do_classify"]:
                        pred_labels_list = labels_classify_name_select.tolist()
                    else:
                        pred_labels_list = labels_detection_name_select.tolist()

                    #记录标注类别中各类别数量
                    if len(true_labels_list) != 0:
                        for cls, count in Counter(true_labels_list).items():
                            multi_category_dict[key]["true_labels_dict"][cls] += count

                    #如果全是背景，那么预测出来得都是误报，要记录下来
                    if len(true_boxes) <= 0:
                        for n in range(len(pred_labels_list)):
                            multi_category_dict[key]["pred_labels_dict"][pred_labels_list[n]] += 1
                        if len(pred_boxes) > 0 and config_common.test_other_info_set["show_wrong_data"]:
                            show_box(img_ori, true_boxes, true_labels_list, pred_boxes, pred_labels_list,
                                     scores_detection_select, scores_classify_select)
                        continue #不在进行下一步操作
                    if len(pred_boxes) <= 0:
                        if len(true_boxes) > 0 and config_common.test_other_info_set["show_wrong_data"]:
                            show_box(img_ori, true_boxes, true_labels_list, pred_boxes, pred_labels_list,
                                     scores_detection_select, scores_classify_select)
                        continue #如果没有预测出嵌件，也不用进行下一步操作

                    #计算iou
                    iou_matrix = calc_iou(pred_boxes, true_boxes)
                    max_iou_idx = np.argmax(iou_matrix, axis=-1)

                    correct_idx = []
                    correct_conf = []
                    find_idx = []
                    for k in range(max_iou_idx.shape[0]):
                        multi_category_dict[key]["pred_labels_dict"][pred_labels_list[k]] += 1
                        match_idx = max_iou_idx[k]  # V level
                        if iou_matrix[k, match_idx] >= config_common.test_other_info_set["test_iou_thresh"] and true_labels_list[match_idx] == pred_labels_list[k]:
                            if match_idx not in find_idx: #如果不在已经找到的列表里面，把id加入列表
                                find_idx.append(match_idx)
                            if match_idx not in correct_idx: #如果不在匹配正确的列表里面，把id加入匹配的列表，因为有可能会多个预测框匹配到同一个嵌件
                                correct_idx.append(match_idx)
                                correct_conf.append(pred_confs[k])
                            else:
                                same_idx = correct_idx.index(match_idx)
                                if pred_confs[k] > correct_conf[same_idx]: #如果多个预测框匹配到同一个嵌件，选着阈值最大的一个
                                    correct_idx.pop(same_idx)
                                    correct_conf.pop(same_idx)
                                    correct_idx.append(match_idx)
                                    correct_conf.append(pred_confs[k])
                        elif iou_matrix[k, match_idx] >= config_common.test_other_info_set["test_iou_thresh"] and true_labels_list[match_idx] != pred_labels_list[k]:
                            multi_category_dict[key]["wrong_classes_dict"][true_labels_list[match_idx]] += 1
                            if match_idx not in find_idx:
                                find_idx.append(match_idx)
                            if config_common.test_other_info_set["write_wrong_label"]:
                                assert os.path.exists(config_common.data_set["wrong_label_save_path"]), "wrong_label_save_path not exit:%s" % config_common.data_set["wrong_label_save_path"]
                                save_label_path = os.path.join(config_common.data_set["wrong_label_save_path"], "%f,%f"%(value[0], value[1]), "wrong_label", true_labels_list[match_idx])
                                if not os.path.exists(save_label_path):
                                    os.makedirs(save_label_path)
                                save_box_path = os.path.join(save_label_path, "%s_%s_%f.jpg"%(true_labels_list[match_idx], pred_labels_list[k], scores_classify_select[k]))
                                x0, y0, x1, y1 = pred_boxes[k]
                                img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
                                cv2.imwrite(save_box_path, img_cut)
                            else:
                                pass

                    for idx in range(len(true_labels_list)):
                        if idx not in find_idx:
                            save_not_find_label_path = os.path.join(config_common.data_set["wrong_label_save_path"],
                                                                    "%f,%f" % (value[0], value[1]), "not_find_label",
                                                                    true_labels_list[idx])
                            if not os.path.exists(save_not_find_label_path):
                                os.makedirs(save_not_find_label_path)
                            save_box_path = os.path.join(save_not_find_label_path, "%s.jpg" % (true_labels_list[idx]))
                            x0, y0, x1, y1 = true_boxes[idx]
                            img_cut = img_ori[max(int(y0-20), 0):min(int(y1+20), img_h-1), max(0, int(x0-20)):min(int(x1+20), img_w-1), :]
                            cv2.imwrite(save_box_path, img_cut)



                    if len(find_idx) != len(true_boxes) or len(pred_boxes) > len(true_boxes): #识别个数不对时显示或者有真实目标没有找到时
                        if config_common.test_other_info_set["show_wrong_data"]:
                            show_box(img_ori, true_boxes, true_labels_list, pred_boxes, pred_labels_list,
                                     scores_detection_select, scores_classify_select)
                        else:
                            pass

                    for t in correct_idx:
                        multi_category_dict[key]["true_positive_dict"][true_labels_list[t]] += 1
                    for t in find_idx:
                        multi_category_dict[key]["find_classes_dict"][true_labels_list[t]] += 1


            F1_score_dict = {}
            #用于记录每种嵌件筛选出来得阈值
            class_threshold_select = {}
            for key, num in class_num_dic.items():
                if num > 0:
                    class_threshold_select[key] = {"detection_threshold":None, "classify_threshold":None, "recall":None, "precision":None}

            for key, value in multi_category_dict.items():
                true_labels_dict = value["true_labels_dict"]
                pred_labels_dict = value["pred_labels_dict"]
                true_positive_dict = value["true_positive_dict"]
                wrong_classes_dict = value["wrong_classes_dict"]
                find_classes_dict = value["find_classes_dict"]

                recall = np.array(list(true_positive_dict.values()), np.float32) / np.array(list(true_labels_dict.values()), np.float32)
                class_wrong = np.array(list(wrong_classes_dict.values()), np.float32) / np.array(list(true_labels_dict.values()), np.float32)
                find = np.array(list(find_classes_dict.values()), np.float32) / np.array(list(true_labels_dict.values()), np.float32)
                not_find = 1 - find #find里已经包含了错误识别得框
                precision = np.array(list(true_positive_dict.values()), np.float32) / np.array(list(pred_labels_dict.values()), np.float32)

                recall_sum = sum(true_positive_dict.values()) / sum(true_labels_dict.values())
                class_wrong_sum = sum(wrong_classes_dict.values()) / sum(true_labels_dict.values())
                find_sum = sum(find_classes_dict.values()) / sum(true_labels_dict.values())
                not_find_sum = 1 - find_sum
                precision_sum = sum(true_positive_dict.values()) / sum(pred_labels_dict.values())

                if not os.path.exists(config_common.data_set["test_result_save_path"]):
                    os.makedirs(config_common.data_set["test_result_save_path"])
                save_path_temp = os.path.join(config_common.data_set["test_result_save_path"], "%s_result.csv"%(key))
                f = open(save_path_temp, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(f)
                csv_writer.writerow(["class_name", "accuracy", "error_rate", "miss_detect_rate", "precision"])
                for id in range(len(qianjian_detection.class_name_classify)):
                    csv_writer.writerow([qianjian_detection.class_name_classify[id], recall[id], class_wrong[id], not_find[id], precision[id]])
                    #如果满足阈值，记录下来
                    if recall[id] >= config_common.test_other_info_set["recall_threshold"] and precision[id] >= config_common.test_other_info_set["precision_threshold"]:
                        if qianjian_detection.class_name_classify[id] not in list(class_threshold_select.keys()):
                            logger.info_ai(meg="this class name not in class_threshold_select:%s"%qianjian_detection.class_name_classify[id])
                            continue
                        if class_threshold_select[qianjian_detection.class_name_classify[id]]["detection_threshold"] is None:
                            class_threshold_select[qianjian_detection.class_name_classify[id]] =\
                                {"detection_threshold": float(key.split("_")[0]), "classify_threshold": float(key.split("_")[1]), "recall": recall[id], "precision": precision[id]}
                        else:
                            if class_threshold_select[qianjian_detection.class_name_classify[id]]["detection_threshold"] < 0.2\
                                    or class_threshold_select[qianjian_detection.class_name_classify[id]]["classify_threshold"] < 0.8:
                                class_threshold_select[qianjian_detection.class_name_classify[id]] = \
                                    {"detection_threshold": float(key.split("_")[0]),
                                     "classify_threshold": float(key.split("_")[1]), "recall": recall[id],
                                     "precision": precision[id]}
                            else:
                                pass
                logger.info_ai(meg="key:%s, mean accuracy:%f, mean error_rate:%f, mean miss_detect_rate:%f, mean precision:%f"%(key, recall_sum, class_wrong_sum, not_find_sum, precision_sum))

                f1_score = 2 * (recall_sum * precision_sum) / (recall_sum + precision_sum)
                if f1_score not in list(F1_score_dict.keys()):
                    F1_score_dict[f1_score] = [key, recall_sum, precision_sum]
                else:
                    logger.info_ai(meg="the key is in F1_score_dict:%f"%f1_score)

                csv_writer.writerow(["average", recall_sum, class_wrong_sum, not_find_sum, precision_sum])
                f.close()

            f1_scores = list(F1_score_dict.keys())
            f1_score_max = max(f1_scores)

            info = "suitable_key, recall: %f, precesion:%f, f1_score: %f, detection score is: %s, classify score is: %s"%(F1_score_dict[f1_score_max][1], F1_score_dict[f1_score_max][2],
                                                                                                                         f1_score_max, F1_score_dict[f1_score_max][0].split("_")[0], F1_score_dict[f1_score_max][0].split("_")[1])
            logger.info_ai(meg=info)

            object_threshold_select_save_path = os.path.join(config_common.data_set["test_result_save_path"], config_common.test_other_info_set["object_threshold_select_save_name"])
            object_threshold_select_writer = open(object_threshold_select_save_path, "w")
            mat = "{:^20}\t{:^20}\t{:^20}\t{:^20}\t{:^20}\n"
            first_line = mat.format("label_name", "detection_threshold", "classify_threshold", "recall", "precision")
            object_threshold_select_writer.write(first_line)
            #判断测试是否合格,并存储阈值文件
            is_pass = True
            for key, value in class_threshold_select.items():
                if value["detection_threshold"] is not None:
                    line = mat.format(key, value["detection_threshold"], value["classify_threshold"], value["recall"], value["precision"])
                    object_threshold_select_writer.write(line)
                    logger.info_ai(meg=line)
                else:
                    line = mat.format(key, "None", "None", "None", "None")
                    object_threshold_select_writer.write(line)
                    logger.info_ai(meg=line)
                    is_pass = False
            object_threshold_select_writer.close()

            if is_pass:
                logger.info_ai(meg="combine test pass")
            else:
                logger.info_ai(meg="combine test not pass")
        logger.info(msg="test over, succeed")
        sys.exit(0)

    except Exception as ex:
        logger.info(msg="find exception:%s"%ex)
        sys.exit(1)