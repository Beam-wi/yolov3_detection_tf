# coding: utf-8

from __future__ import division, print_function

import os
import sys
import shutil
import argparse
import tensorflow as tf
import numpy as np
from utils.data_utils import parse_line
from utils.data_utils import parse_data
from utils.misc_utils import parse_anchors, read_class_names, shuffle_and_overwrite, config_learning_rate, config_optimizer, \
    average_gradients, get_background_lines
from utils.nms_utils import gpu_nms
from utils.get_data_from_web import get_data_from_web_and_save
from utils.config_parse import get_config
from utils.model import yolov3_trt
from utils.logging_util import Logger
import utils.globalvar as globalvar


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
    logger = Logger(config_common.data_set["log_save_path"], "detection_train")
    logger.clean_log_dir()

    logger.info_ai(meg="project name is:%s"%config_common.project_name)
    logger.info_ai(meg="config info", get_ins={"config": config_common})

    if not os.path.exists(config_common.data_set["model_save_path"]):
        try:
            os.makedirs(config_common.data_set["model_save_path"])
            logger.info_ai(meg="make model save dir:%s"%config_common.data_set["model_save_path"])
        except Exception as ex:
            logger.info_ai(meg="make model save dir:%s, failed, cause of %s" % (config_common.data_set["model_save_path"], ex))
            sys.exit(1)

    if not os.path.exists(config_common.data_set["data_save_path_temp"]):
        try:
            os.makedirs(config_common.data_set["data_save_path_temp"])
            logger.info_ai(meg="make data save dir:%s"%config_common.data_set["data_save_path_temp"])
        except Exception as ex:
            logger.info_ai(meg="make data save dir:%s, failed, cause of %s" % (config_common.data_set["data_save_path_temp"], ex))
            sys.exit(1)

    #获取背景图片，如果要贴背景到图片中进行训练的话
    if config_common.data_set["add_background_path"] != "None":
        background_lines = get_background_lines(config_common.data_set["add_background_path"])
        globalvar.set_background_add_lines(background_lines)

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
    assert os.path.exists(config_common.data_set["model_restore_path"]+".meta"), "model_restore_path not exit:%s"%config_common.data_set["model_restore_path"]
    assert os.path.exists(config_common.data_set["log_save_path"]), "log_save_path not exit:%s"%config_common.data_set["log_save_path"]
    assert os.path.exists(config_common.data_set["model_save_path"]), "model_save_path not exit:%s"%config_common.data_set["model_save_path"]
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
            for line in data_train:
                pic_path, boxes, labels = parse_line(line)
                for label in labels:
                    if label not in config_common.data_set["fill_zero_label_names"] and label != "gj":
                        if label not in class_num_dic:
                            class_num_dic[label] = 1
                        else:
                            class_num_dic[label] += 1

        logger.info_ai(meg="prepare data over, get all class name from data file", get_ins={"class_num_dic": class_num_dic})

        lr_decay_freq = int(config_common.model_set["lr_decay_freq_epoch"] * train_num / (len(gpu_device) * config_common.model_set["batch_size"]))


        # args params
        anchors = parse_anchors(config_common.data_set["anchor_path"]) * 2.5
        classes = read_class_names(config_common.data_set["class_name_path"])
        classes_list = list(classes.values())
        logger.info_ai(meg="train class name get from class name file", get_ins={"classes_list": classes_list})
        #把不在训练类别里面的类别从类别个数字典中删除
        for key in list(class_num_dic.keys()):
            if key not in classes_list:
                class_num_dic.pop(key)
                logger.info_ai(meg="************data calss name:%s not in classes list"%key)
        class_num_dic = {key: 1/(value+1) for key, value in class_num_dic.items()}

        class_num_dic = str(class_num_dic)
        class_num = 1 #进行检测训练时，所有目标合并为一个类别
        train_img_cnt = len(open(config_common.data_set["train_file_path"], 'r').readlines())
        val_img_cnt = len(open(config_common.data_set["val_file_path"], 'r').readlines())
        train_batch_num = int(np.ceil(float(train_img_cnt) / (config_common.model_set["batch_size"] * len(gpu_device)))) - 2
        val_batch_num = int(np.ceil(float(val_img_cnt) / (config_common.model_set["batch_size"] * len(gpu_device)))) - 2

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            # setting placeholders
            is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
            handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

            shuffle_and_overwrite(config_common.data_set["train_file_path"]) #每轮都随机打乱数据集顺序
            train_dataset = tf.data.TextLineDataset(config_common.data_set["train_file_path"])
            train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
                lambda x: tf.py_func(parse_data, [x, classes_list, class_num, config_common.model_set["image_size"], anchors, 'train', config_common.data_set["data_save_path_temp"], config_common.data_set["nfs_mount_path"],
                                                  class_num_dic, config_common.model_set["train_with_gray"], config_common.data_set["fill_zero_label_names"]], [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]),
                num_parallel_calls=config_common.model_set["num_threads"], batch_size=config_common.model_set["batch_size"]))
            train_dataset = train_dataset.prefetch(config_common.model_set["prefetech_buffer"])

            val_dataset = tf.data.TextLineDataset(config_common.data_set["val_file_path"])
            val_dataset = val_dataset.apply(tf.contrib.data.map_and_batch(
                lambda x: tf.py_func(parse_data, [x, classes_list, class_num, config_common.model_set["image_size"], anchors, 'val', config_common.data_set["data_save_path_temp"], config_common.data_set["nfs_mount_path"],
                                                  class_num_dic, config_common.model_set["train_with_gray"], config_common.data_set["fill_zero_label_names"]], [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]),
                num_parallel_calls=config_common.model_set["num_threads"], batch_size=config_common.model_set["batch_size"]))
            val_dataset.prefetch(config_common.model_set["prefetech_buffer"])

            # creating two dataset iterators
            train_iterator = train_dataset.make_initializable_iterator()
            val_iterator = val_dataset.make_initializable_iterator()

            # creating two dataset handles
            train_handle = train_iterator.string_handle()
            val_handle = val_iterator.string_handle()
            # select a specific iterator based on the passed handle
            dataset_iterator = tf.data.Iterator.from_string_handle(handle_flag, train_dataset.output_types,
                                                                   train_dataset.output_shapes)
            ################
            # register the gpu nms operation here for the following evaluation scheme
            pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
            pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
            gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, class_num)
            ################
            global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            if config_common.model_set["use_warm_up"]:
                learning_rate = tf.cond(tf.less(global_step, train_batch_num * config_common.model_set["warm_up_epoch"]),
                                        lambda: config_common.model_set["warm_up_lr"], lambda: config_learning_rate(config_common,
                                                                                              global_step - train_batch_num * config_common.model_set["warm_up_epoch"], lr_decay_freq))
            else:
                learning_rate = config_learning_rate(config_common, global_step, lr_decay_freq)

            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = config_optimizer(config_common.model_set["optimizer_name"], learning_rate)
            #add my clip
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 3.0)
            with tf.device(gpu_device[0]):
                tower_grads = []
                with tf.variable_scope(tf.get_variable_scope()):
                    for gpu_id in range(len(gpu_device)):
                        with tf.device(gpu_device[gpu_id]):
                            with tf.name_scope('%s_%d' % ('tower', gpu_id)) as scope:
                                # get an element from the choosed dataset iterator
                                image, y_true_13, y_true_26, y_true_52, img_path = dataset_iterator.get_next()
                                # 目标中心位于那个anchor,那个anchor就负责检测这个物体,他的x,y,w,h就是真是目标的数据,其余的anchor为0
                                y_true = [y_true_13, y_true_26, y_true_52]
                                # tf.data pipeline will lose the data shape, so we need to set it manually
                                image.set_shape([None, config_common.model_set["image_size"][1], config_common.model_set["image_size"][0], 3])
                                for y in y_true:
                                    y.set_shape([None, None, None, None, None])

                                ##################
                                # Model definition
                                ##################
                                yolo_model = yolov3_trt(class_num, anchors, config_common.model_set["backbone_name"], config_common.model_set["train_with_two_feature_map"])
                                with tf.variable_scope('yolov3'):
                                    pred_feature_maps = yolo_model.forward(image, is_training=is_training, train_with_gray=config_common.model_set["train_with_gray"])
                                loss = yolo_model.compute_loss(pred_feature_maps, y_true)

                                tf.get_variable_scope().reuse_variables()
                                y_pred = yolo_model.predict(pred_feature_maps)
                                tf.summary.scalar('%s_%d_train_batch_statistics/total_loss' % ('tower', gpu_id), loss[0])
                                tf.summary.scalar('%s_%d_train_batch_statistics/loss_xy' % ('tower', gpu_id), loss[1])
                                tf.summary.scalar('%s_%d_train_batch_statistics/loss_wh' % ('tower', gpu_id), loss[2])
                                tf.summary.scalar('%s_%d_train_batch_statistics/loss_conf' % ('tower', gpu_id), loss[3])
                                tf.summary.scalar('%s_%d_train_batch_statistics/loss_class' % ('tower', gpu_id), loss[4])

                                grads = optimizer.compute_gradients(loss[0])
                                tower_grads.append(grads)
                if config_common.model_set["restore_part"] == ['None']:
                    saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=[None]))
                else:
                    saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=config_common.model_set["restore_part"]))
                if config_common.model_set["update_part"] == ['None']:
                    update_vars = tf.contrib.framework.get_variables_to_restore(include=[None])
                else:
                    update_vars = tf.contrib.framework.get_variables_to_restore(include=config_common.model_set["update_part"])

                saver_to_save = tf.train.Saver(max_to_keep=1) #只保留最后一个模型
                # set dependencies for BN ops
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads = average_gradients(tower_grads)
                    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
                    train_op = tf.group(apply_gradient_op)

                # with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9), allow_soft_placement=True, log_device_placement=False)) as sess:
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), train_iterator.initializer])#local variables 不被存储的变量
                    train_handle_value, val_handle_value = sess.run([train_handle, val_handle])
                    saver_to_restore.restore(sess, config_common.data_set["model_restore_path"])
                    merged = tf.summary.merge_all()
                    writer = tf.summary.FileWriter(config_common.data_set["log_save_path"], sess.graph)

                    logger.info_ai(meg='\n----------- start to train -----------\n')
                    for epoch in range(config_common.model_set["total_epoches"]):
                        for i in range(train_batch_num):
                            _, summary, y_pred_, y_true_, loss_, global_step_, lr, img_path_ = sess.run([train_op, merged, y_pred, y_true, loss, global_step, learning_rate, img_path],
                                                                                             feed_dict={is_training: True, handle_flag: train_handle_value})
                            writer.add_summary(summary, global_step=global_step_)
                            info = "Epoch: {}, global_step: {}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}, lr:{:.9f}".format(
                                epoch, global_step_, loss_[0], loss_[1], loss_[2], loss_[3], loss_[4], lr)
                            logger.info_ai(meg=info)

                            # start to save
                            if global_step_ % config_common.model_set["save_freq"] == 0 and global_step_ > 0:
                                if not os.path.exists(config_common.data_set["model_save_path"]):
                                    os.makedirs(config_common.data_set["model_save_path"])

                                shutil.copy(config_common.data_set["anchor_path"], config_common.data_set["model_save_path"])
                                saver_to_save.save(sess, config_common.data_set["model_save_path"] + '/%s'%config_common.model_set["model_save_name"])
                                logger.info_ai(meg="save model:%s"%(config_common.data_set["model_save_path"] + '/%s'%config_common.model_set["model_save_name"]))

                        shuffle_and_overwrite(config_common.data_set["train_file_path"])
                        sess.run(train_iterator.initializer)
        logger.info(msg="train over, succeed")
        sys.exit(0)

    except Exception as ex:
        logger.info(msg="find exception:%s"%ex)
        sys.exit(1)
