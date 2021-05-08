import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from utils.model import darknet_plus
from utils.data_utils import parse_line
from utils.cut_box_data_mutil import cut_data_multi_process
from utils.misc_utils import read_class_names
from utils.get_data_from_web import get_data_from_web_and_save
from datetime import datetime 
from utils.classify_utils import builddata_shuffle_and_overwrite, parse_data, center_distance_loss
from utils.config_parse import get_config
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
    logger = Logger(config_common.data_set["log_save_path"], "classify_train")
    logger.clean_log_dir()

    logger.info_ai(meg="project name is:%s"%config_common.project_name)
    logger.info_ai(meg="config info", get_ins={"config": config_common})

    if not os.path.exists(config_common.data_set["classify_model_save_path"]):
        try:
            os.makedirs(config_common.data_set["classify_model_save_path"])
            logger.info_ai(meg="make classify model save dir:%s"%config_common.data_set["classify_model_save_path"])
        except Exception as ex:
            logger.info_ai(meg="make classify model save dir:%s, failed, cause of %s" % (config_common.data_set["classify_model_save_path"], ex))
            sys.exit(1)

    if not os.path.exists(config_common.data_set["classify_train_data_save_path_temp"]):
        try:
            os.makedirs(config_common.data_set["classify_train_data_save_path_temp"])
            logger.info_ai(meg="make classify train data save dir:%s"%config_common.data_set["classify_train_data_save_path_temp"])
        except Exception as ex:
            logger.info_ai(meg="make classify train data save dir:%s, failed, cause of %s" % (config_common.data_set["classify_train_data_save_path_temp"], ex))
            sys.exit(1)

    if not os.path.exists(config_common.data_set["classify_val_data_save_path_temp"]):
        try:
            os.makedirs(config_common.data_set["classify_val_data_save_path_temp"])
            logger.info_ai(meg="make classify val data save dir:%s"%config_common.data_set["classify_val_data_save_path_temp"])
        except Exception as ex:
            logger.info_ai(meg="make classify val data save dir:%s, failed, cause of %s" % (config_common.data_set["classify_val_data_save_path_temp"], ex))
            sys.exit(1)

    #设置全局变量
    globalvar.set_logger(logger)
    globalvar.set_config(config_common)

    #判断必须文件是否存在
    assert os.path.exists(config_common.data_set["train_file_path"]), "train_file_path not exit:%s"%config_common.data_set["train_file_path"]
    assert os.path.exists(config_common.data_set["val_file_path"]), "val_file_path not exit:%s"%config_common.data_set["val_file_path"]
    assert os.path.exists(config_common.data_set["nfs_mount_path"]), "nfs_mount_path not exit:%s"%config_common.data_set["nfs_mount_path"]
    assert os.path.exists(config_common.data_set["anchor_path"]), "anchor_path not exit:%s"%config_common.data_set["anchor_path"]
    assert os.path.exists(config_common.data_set["class_name_path"]), "class_name_path not exit:%s"%config_common.data_set["class_name_path"]
    assert os.path.exists(config_common.data_set["classify_model_restore_model"]+".meta"), "model_classify_restore_path not exit:%s"%config_common.data_set["model_classify_restore_path"]
    assert os.path.exists(config_common.data_set["log_save_path"]), "log_save_path not exit:%s"%config_common.data_set["log_save_path"]
    assert os.path.exists(config_common.data_set["classify_model_save_path"]), "classify_model_save_path not exit:%s"%config_common.data_set["classify_model_save_path"]
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

        classes = read_class_names(config_common.data_set["class_name_path"])
        classes_list = list(classes.values())
        logger.info_ai(meg="train class name get from class name file", get_ins={"classes_list": classes_list})

        name2id = {}
        id = 0
        for name in classes_list:
            name2id[name] = id
            id = id + 1
        logger.info_ai(meg="get class name id", get_ins={"name2id": name2id})
        if config_common.data_set["need_cut_object"]:
            logger.info_ai(meg="cut train data box start")
            cut_data_multi_process(config_common.data_set["train_file_path"], config_common.data_set["classify_train_data_save_path_temp"], do_extension_boxes=True, extension_ratio=config_common.test_other_info_set["extension_ratio"])
            logger.info_ai(meg="cut train data box over")
            logger.info_ai(meg="cut val data box start")
            cut_data_multi_process(config_common.data_set["val_file_path"], config_common.data_set["classify_val_data_save_path_temp"], do_extension_boxes=True, extension_ratio=config_common.test_other_info_set["extension_ratio"])
            logger.info_ai(meg="cut val data box over")
        else:
            logger.info_ai(meg="set not cut any object")


        with tf.Graph().as_default(), tf.device('/cpu:0'):
            is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
            handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
            builddata_shuffle_and_overwrite(config_common.data_set["classify_train_data_save_path_temp"], config_common.data_set["train_file_classify"], classes_list, name2id, mode="train", max_num=10000.)
            builddata_shuffle_and_overwrite(config_common.data_set["classify_val_data_save_path_temp"], config_common.data_set["val_file_classify"], classes_list, name2id, mode="val", max_num=100.) #如果是val，max_num 参数无效
            train_img_cnt = len(open(config_common.data_set["train_file_classify"], 'r').readlines())
            val_img_cnt = len(open(config_common.data_set["val_file_classify"], 'r').readlines())
            train_batch_num = int(np.ceil(float(train_img_cnt) / config_common.model_set["classify_batch_size"]))
            val_batch_num = int(np.ceil(float(val_img_cnt) / config_common.model_set["classify_batch_size"]))

            assert train_img_cnt != 0, "train data num is 0"
            assert val_img_cnt != 0, "val data num is 0"

            train_dataset = tf.data.TextLineDataset(config_common.data_set["train_file_classify"])
            train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
                lambda x: tf.py_func(parse_data, [x, len(classes_list), config_common.model_set["classify_size"], 'train'], [tf.float32, tf.float32, tf.int32]),
                num_parallel_calls=config_common.model_set["num_threads"], batch_size=config_common.model_set["classify_batch_size"]))
            train_dataset.prefetch(config_common.model_set["prefetech_buffer"])

            val_dataset = tf.data.TextLineDataset(config_common.data_set["val_file_classify"])
            val_dataset = val_dataset.apply(tf.contrib.data.map_and_batch(
                lambda x: tf.py_func(parse_data, [x, len(classes_list), config_common.model_set["classify_size"], 'val'], [tf.float32, tf.float32, tf.int32]),
                num_parallel_calls=config_common.model_set["num_threads"], batch_size=config_common.model_set["classify_batch_size"]))
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
            image, y_true, y = dataset_iterator.get_next()
            image.set_shape([None, config_common.model_set["classify_size"][1], config_common.model_set["classify_size"][0], 3])
            y_true.set_shape([None, len(classes_list)])
            y.set_shape([None, ])

            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            with tf.device(gpu_device[0]):
                model = darknet_plus(len(classes_list))
                with tf.variable_scope('yolov3_classfication'):
                    logits, center_feature = model.forward(image, is_training=is_training)

                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
                saver_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3_classfication"], exclude=["yolov3_classfication/yolov3_head"])) #yolov3_classfication/yolov3_head/dense_1
                with tf.variable_scope('center_loss'):
                    center_loss, _, distance = center_distance_loss(center_feature, y, 0.4, len(classes_list))

                center_loss = center_loss * 5e-1
                distance = distance * 1e-2
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config_common.model_set["momentum"], use_nesterov=True)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                loss_all = cost + l2_loss * config_common.model_set["weight_decay"]
                with tf.control_dependencies(update_ops):
                    train = optimizer.minimize(loss_all)
                    #train = optimizer.minimize(loss_all, var_list=update_vars)

                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth=True

                with tf.Session(config=config) as sess:
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), train_iterator.initializer])
                    train_handle_value, val_handle_value = sess.run([train_handle, val_handle])
                    saver_restore.restore(sess, config_common.data_set["classify_model_restore_model"])
                    logger.info_ai(meg="load model from %s"%config_common.data_set["classify_model_restore_model"])

                    epoch_learning_rate = config_common.model_set["classify_init_learning_rate"]
                    max_acc = 0
                    for epoch in range(1, config_common.model_set["classify_total_epochs"] + 1):
                        if epoch % 4 == 0 :
                            epoch_learning_rate = max(epoch_learning_rate / 10, 1e-6)
                        train_acc = 0.0
                        train_loss = 0.0

                        for step in range(1, train_batch_num + 1):
                            train_feed_dict = {
                                is_training: True,
                                handle_flag: train_handle_value,
                                learning_rate: epoch_learning_rate
                            }

                            batch_loss, _, batch_acc, batch_center_loss, distance_, loss_all_ = sess.run([cost, train, accuracy, center_loss, distance, loss_all], feed_dict=train_feed_dict)
                            train_loss += batch_loss
                            train_acc += batch_acc
                            if step%10 == 0:
                                info = "step is %d, train batch loss is %f, train center loss is %f, distance is %f, loss_all is %f, batch_acc is %f, val_max_acc:%.4f, lr:%f"%(step, batch_loss, batch_center_loss, distance_, loss_all_,  batch_acc, max_acc, epoch_learning_rate)
                                logger.info_ai(info)

                        train_loss /= train_batch_num # average loss
                        train_acc /= train_batch_num # average accuracy
                        sess.run(val_iterator.initializer)
                        test_acc = 0.0
                        test_loss = 0.0
                        for step in range(1, val_batch_num + 1):
                            val_feed_dict = {
                                is_training: False,
                                handle_flag: val_handle_value,
                                learning_rate: epoch_learning_rate
                            }
                            loss_, acc_ = sess.run([cost, accuracy], feed_dict=val_feed_dict)

                            test_loss += loss_
                            test_acc += acc_

                        test_loss /= val_batch_num  # average loss
                        test_acc /= val_batch_num  # average accuracy

                        info = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f, val_max_acc:%.4f \n" % (
                            epoch, config_common.model_set["classify_total_epochs"], train_loss, train_acc, test_loss, test_acc, max_acc)
                        logger.info_ai(meg=info)
                        if test_acc > max_acc:
                            max_acc = test_acc
                        if abs(test_acc - max_acc) < 0.05:
                            now = datetime.now()
                            time_buid = datetime.strftime(now, '%Y-%m-%d')
                            saver.save(sess=sess, save_path='%s/%s' % (config_common.data_set["classify_model_save_path"], config_common.model_set["classify_model_save_name"]))
                            logger.info_ai(meg='%s/%s' % (config_common.data_set["classify_model_save_path"], config_common.model_set["classify_model_save_name"]))
                        builddata_shuffle_and_overwrite(config_common.data_set["classify_train_data_save_path_temp"],
                                                        config_common.data_set["train_file_classify"], classes_list,
                                                        name2id, mode="train", max_num=10000.)
                        builddata_shuffle_and_overwrite(config_common.data_set["classify_val_data_save_path_temp"],
                                                        config_common.data_set["val_file_classify"], classes_list, name2id,
                                                        mode="val", max_num=100.)  # 如果是val，max_num 参数无效
                        sess.run(train_iterator.initializer)

        logger.info(msg="classify train over, succeed")
        sys.exit(0)

    except Exception as ex:
        logger.info(msg="find exception:%s"%ex)
        sys.exit(1)