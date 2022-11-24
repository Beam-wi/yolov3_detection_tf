# coding: utf-8

import os
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf
import utils.globalvar as globalvar

from utils.model import darknet_plus
from utils.cut_box_data_mutil import cut_data_multi_process
from utils.classify_utils import builddata_shuffle_and_overwrite, parse_data, center_distance_loss
from utils.config_parse import get_config as getConfig
from utils.logging_util import Logger
from utils.utils import makeDir, dataConfirm, makeValFile


projectPath = os.path.abspath(os.path.dirname(__file__))


def getDataSet(graph, clsFileTrain, clsFileVal, classNameDict, cfg):
    with graph.as_default(), tf.device('/cpu:0'):
        isTraining = tf.placeholder(dtype=tf.bool, name="phase_train")
        handleFlag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

        trainDataset = tf.data.TextLineDataset(clsFileTrain)
        trainDataset = trainDataset.apply(tf.contrib.data.map_and_batch(
            lambda x: tf.py_func(parse_data,
                                 [x, len(classNameDict.keys()), cfg.model_set["classify_size"], 'train'],
                                 [tf.float32, tf.float32, tf.int32]),
            num_parallel_calls=cfg.model_set["num_threads"],
            batch_size=cfg.model_set["classify_batch_size"]))
        trainDataset.prefetch(cfg.model_set["prefetech_buffer"])

        valDataset = tf.data.TextLineDataset(clsFileVal)
        valDataset = valDataset.apply(tf.contrib.data.map_and_batch(
            lambda x: tf.py_func(parse_data,
                                 [x, len(classNameDict.keys()), cfg.model_set["classify_size"], 'val'],
                                 [tf.float32, tf.float32, tf.int32]),
            num_parallel_calls=cfg.model_set["num_threads"],
            batch_size=cfg.model_set["classify_batch_size"]))
        valDataset.prefetch(cfg.model_set["prefetech_buffer"])

        # creating two dataset iterators
        trainIterator = trainDataset.make_initializable_iterator()
        valIterator = valDataset.make_initializable_iterator()

        # creating two dataset handles
        trainHandle = trainIterator.string_handle()
        valHandle = valIterator.string_handle()
        # select a specific iterator based on the passed handle
        datasetIterator = tf.data.Iterator.from_string_handle(
            handleFlag, trainDataset.output_types, trainDataset.output_shapes)
        image, yTrue, y = datasetIterator.get_next()
        image.set_shape(
            [None, cfg.model_set["classify_size"][1], cfg.model_set["classify_size"][0], 3])
        yTrue.set_shape([None, len(classNameDict.keys())])
        y.set_shape([None, ])

        learningRate = tf.placeholder(tf.float32, name='learning_rate')

        return image, yTrue, y, trainIterator, trainHandle, valHandle, \
               classNameDict, learningRate, isTraining, valIterator, handleFlag


def getModel(graph, gpus, image, yTrue, y, trainIterator, trainHandle,
             valHandle, classNameDict, learningRate, isTraining, cfg, logger):
    with graph.as_default(), tf.device(gpus[0]):
        model = darknet_plus(len(classNameDict.keys()))
        with tf.variable_scope('yolov3_classfication'):
            logits, centerFeature = model.forward(image, is_training=isTraining)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yTrue,logits=logits))
        saverRestore = tf.train.Saver(
            var_list=tf.contrib.framework.get_variables_to_restore(
                include=["yolov3_classfication"],
                exclude=["yolov3_classfication/yolov3_head"]))  # yolov3_classfication/yolov3_head/dense_1
        with tf.variable_scope('center_loss'):
            centerLoss, _, distance = center_distance_loss(
                centerFeature, y, 0.4, len(classNameDict.keys()))

        centerLoss = centerLoss * 5e-1
        distance = distance * 1e-2
        l2Loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learningRate,
            momentum=cfg.model_set["momentum"],use_nesterov=True)

        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        lossAll = cost + l2Loss * cfg.model_set["weight_decay"]
        with tf.control_dependencies(updateOps):
            trainOp = optimizer.minimize(lossAll)

        correctPrediction = tf.equal(tf.argmax(logits, 1), tf.argmax(yTrue, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # build sess
        sess = tf.Session(config=config)
        # variables initialize
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer(),
                  trainIterator.initializer])
        # get handle
        trainHandleValue, valHandleValue = sess.run([trainHandle, valHandle])
        saverRestore.restore(sess,
                             os.path.join(
                                 projectPath,
                                 cfg.data_set["classify_model_restore_path"]))
        logger.info(f"load model from: "
                    f"{os.path.join(projectPath, cfg.data_set['classify_model_restore_path'])}" )

        return sess, cost, trainOp, accuracy, centerLoss, distance, lossAll, \
               trainHandleValue, valHandleValue, isTraining, saver


def train(sess, cfg, cost, trainOp, accuracy, centerLoss, distance, lossAll,
          trainIterator, valIterator, trainBatchNum, valBatchNum, dataTrainDir,
          dataValDir, clsFileTrain, clsFileVal, classNameDict, valNum,
          handleFlag, trainHandleValue, valHandleValue, learningRate,
          isTraining, logger, saver):
    epochLearningRate = cfg.model_set["classify_init_learning_rate"]
    maxAcc = 0
    for epoch in range(1, cfg.model_set["classify_total_epochs"] + 1):
        if epoch % 4 == 0:
            epochLearningRate = max(epochLearningRate / 10, 1e-6)
        trainAcc = 0.0
        trainLoss = 0.0
        for step in range(1, trainBatchNum + 1):
            trainFeedDict = {
                isTraining: True,
                handleFlag: trainHandleValue,
                learningRate: epochLearningRate}
            batchLoss, _, batchAcc, batchCenterLoss, distance_, lossAll_ = \
                sess.run(
                    [cost, trainOp, accuracy, centerLoss, distance, lossAll],
                    feed_dict=trainFeedDict)
            trainLoss += batchLoss
            trainAcc += batchAcc
            if step % 10 == 0:
                logger.info(f"Epoch: {epoch}/{cfg.model_set['classify_total_epochs']}, "
                            f"Step: {step}, batchLoss: {'%.4f'%batchLoss}, "
                            f"centerLoss: {'%.4f'%batchCenterLoss}, distance: {'%.4f'%distance_}, "
                            f"loss: {'%.4f'%lossAll_}, batchAcc: {'%.4f'%batchAcc}, "
                            f"valAaxAcc: {'%.4f'%maxAcc}, lr: {epochLearningRate}")

        trainLoss /= trainBatchNum  # average loss
        trainAcc /= trainBatchNum  # average accuracy

        if valNum != 0:
            sess.run(valIterator.initializer)
            testAcc = 0.0
            testLoss = 0.0
            for step in range(1, valBatchNum + 1):
                valFeedDict = {
                    isTraining: False,
                    handleFlag: valHandleValue,
                    learningRate: epochLearningRate}
                loss_, acc_ = sess.run(
                    [cost, accuracy],
                    feed_dict=valFeedDict)
                testLoss += loss_
                testAcc += acc_
            testLoss /= valBatchNum  # average loss
            testAcc /= valBatchNum  # average accuracy
            logger.info(f"Epoch: {epoch}/{cfg.model_set['classify_total_epochs']}, "
                           f"loss: {'%.4f'%trainLoss}, trainAcc: {'%.4f'%trainAcc}, "
                           f"testLoss: {'%.4f'%testLoss}, testAcc: {'%.4f'%testAcc}, "
                           f"valMaxAcc:{'%.4f'%maxAcc} \n")
            if testAcc > maxAcc:
                maxAcc = testAcc
            if abs(testAcc - maxAcc) < 0.05:
                modelSavePath = os.path.join(
                    cfg.data_set["model_and_temp_file_save_path"],
                    cfg.data_set["classify_model_save_dir_name"])
                if not os.path.exists(modelSavePath):
                    os.makedirs(modelSavePath)
                saver.save(
                    sess=sess,
                    save_path=f'{modelSavePath}/{cfg.model_set["classify_model_save_name"]}')
                logger.info(f'{modelSavePath}/{cfg.model_set["classify_model_save_name"]}')
            builddata_shuffle_and_overwrite(
                dataValDir, clsFileVal, list(classNameDict.keys()),
                classNameDict, mode="val", max_num=100.)  # 如果是val，max_num 参数无效
        else:
            modelSavePath = os.path.join(
                cfg.data_set["model_and_temp_file_save_path"],
                cfg.data_set["classify_model_save_dir_name"])
            if not os.path.exists(modelSavePath):
                os.makedirs(modelSavePath)
            saver.save(
                sess=sess,
                save_path=f'{modelSavePath}/{cfg.model_set["classify_model_save_name"]}')
            logger.info(f'{modelSavePath}/{cfg.model_set["classify_model_save_name"]}')
        builddata_shuffle_and_overwrite(
            dataTrainDir, clsFileTrain, list(classNameDict.keys()),
            classNameDict, mode="train", max_num=10000.)
        sess.run(trainIterator.initializer)


def main():
    parser = argparse.ArgumentParser(description="YOLO-V3 training procedure.")
    parser.add_argument("--config_project", type=str,
                        default=".data/config/project.yaml",
                        help="项目配置文件.")
    parser.add_argument("--config_common", type=str,
                        default="data/config/common.yaml",
                        help="默认配置文件.")
    args = parser.parse_args()

    args.config_common = os.path.join(projectPath, args.config_common)
    cfg = getConfig(args.config_common)
    cfg.update(getConfig(args.config_project).as_dict())

    # 创建日志文件
    logSavePath = os.path.join(cfg.data_set["model_and_temp_file_save_path"],
                               cfg.data_set["log_save_dir_name"])
    makeDir(logSavePath)
    logger = Logger(logSavePath, "train_classification")
    logger.clean_log_dir()
    logger.info(f"project name is: {cfg.project_name}")
    logger.info(f"config: {cfg}")

    # 创建部分存储路径
    dataTempDir = cfg.data_set["data_save_path_temp"]
    dataTrainDir = os.path.join(
        cfg.data_set["model_and_temp_file_save_path"],
        "data_classify",
        cfg.data_set["classify_train_data_save_dir_name"])
    dataValDir = os.path.join(
        cfg.data_set["model_and_temp_file_save_path"],
        "data_classify",
        cfg.data_set["classify_val_data_save_dir_name"])
    modelSaveDir = os.path.join(
        cfg.data_set["model_and_temp_file_save_path"],
        cfg.data_set["classify_model_save_dir_name"])

    makeDir(dataTempDir, logger=logger)
    makeDir(dataTrainDir, logger=logger)
    makeDir(dataValDir, logger=logger)
    makeDir(modelSaveDir, logger=logger)

    # 设置全局变量
    globalvar.set_logger(logger)
    globalvar.set_config(cfg)

    # 判断必须文件是否存在
    nfsMountDir = cfg.data_set["nfs_mount_path"]
    detTrainPath = cfg.data_set["detection_train_file_path"]
    detValPath = detTrainPath.replace('_train.txt', '_val.txt')
    clsClassPath = cfg.data_set["class_name_path"]
    pretrainPath = os.path.join(projectPath,
                                cfg.data_set["classify_model_restore_path"])
    configSavePath = os.path.join(
        modelSaveDir, f"{cfg.model_set['classify_model_save_name']}.yaml")
    assert os.path.exists(nfsMountDir), \
        "nfs mount path not exit:%s" % nfsMountDir
    assert os.path.exists(detTrainPath), \
        "detection train file path not exit:%s" % detTrainPath
    assert os.path.exists(clsClassPath), \
        "class name path not exit:%s" % clsClassPath
    assert os.path.exists(pretrainPath + ".meta"), \
        "detection model restore path not exit:%s" % pretrainPath
    if not os.path.exists(detValPath):
        makeValFile(detValPath, cfg.data_set["detection_val_file_path"])
    cfg.save(configSavePath)

    # 参数
    # detGpuDevice = cfg.model_set["detection_train_gpu_device"]
    detGpuDevice = cfg.model_set["other_gpu_device"]
    lrDecayFreqEpoch = cfg.model_set["lr_decay_freq_epoch"]
    batchSize = cfg.model_set["classify_batch_size"]
    imageSize = cfg.model_set["classify_size"]
    clsModelName = cfg.model_set["classify_model_save_name"]
    epoches = cfg.model_set["classify_total_epochs"]
    extRatio = cfg.model_set["extension_ratio"]
    clsFileTrain = os.path.join(
        cfg.data_set["model_and_temp_file_save_path"],
        "data_classify",
        cfg.data_set["classify_train_file"])
    clsFileVal = os.path.join(
        cfg.data_set["model_and_temp_file_save_path"],
        "data_classify",
        cfg.data_set["classify_val_file"])

    # 设置显卡,os.env是限制访问GPU序号，
    # 程序中实际调用GPU设置tf.device时应从0开始一一对应选则的物理GPU（由小到大）
    os.environ['CUDA_VISIBLE_DEVICES'] = detGpuDevice
    gpus = [f"/gpu:{d}" for d in range(len(detGpuDevice.strip().split(",")))]
    classNameDict = {x.split()[0]: i for i, x in enumerate(open(clsClassPath).readlines())}
    logger.info(f"get class name: {classNameDict} from file: {clsClassPath}")

    # 统计确认训练数据原图
    classNumDict, trainDetNum = \
        dataConfirm(
            detTrainPath, detValPath, classNameDict.keys(),
            nfsMountDir, dataTempDir, cfg, logger, batchPath=None)
    logger.info(f"get train class num: {classNumDict}")

    # 准备切图用于分类
    try:
        if cfg.data_set["need_cut_object"]:
            logger.info("Cutting train data box >>> >>> >>>")
            cut_data_multi_process(
                detTrainPath, dataTrainDir, nfsMountDir, dataTempDir, do_extension_boxes=True, extension_ratio=extRatio)
            logger.info("Cutting train data box over <<< <<< <<<")
            logger.info("Cutting val data box >>> >>> >>>")
            cut_data_multi_process(
                detValPath, dataValDir, nfsMountDir, dataTempDir, do_extension_boxes=True, extension_ratio=extRatio)
            logger.info("Cutting val data box over <<< <<< <<<")
        else:
            logger.info("set not cut any object")
            print(dataTrainDir)
            assert len(list(os.listdir(dataTrainDir))) > 0, \
                "No img confirm if saved cut image or need_cut_object is True."
    except Exception as e:
        logger.error(f'Exception occur when make cut images: {e}')
        raise Exception(f'Exception occur when make cut images: {e}')

    try:
        # 将分类数据数量合理化并生成txt
        builddata_shuffle_and_overwrite(
            dataTrainDir, clsFileTrain, list(classNameDict.keys()),
            classNameDict, mode="train", max_num=10000.)
        builddata_shuffle_and_overwrite(
            dataValDir, clsFileVal, list(classNameDict.keys()),
            classNameDict, mode="val", max_num=100.)  # 如果是val，max_num 参数无效

        trainNum = len(open(clsFileTrain, 'r').readlines())
        valNum = len(open(clsFileVal, 'r').readlines())
        trainBatchNum = int(np.ceil(float(trainNum) / batchSize))
        valBatchNum = int(np.ceil(float(valNum) / batchSize))
        assert trainNum != 0, "train data num is 0"
        # assert valNum != 0, "val data num is 0"

        # make graph
        graph = tf.Graph()

        # dataset and placeholders
        image, yTrue, y, trainIterator, trainHandle, valHandle, classNameDict, \
        learningRate, isTraining, valIterator, handleFlag, =\
            getDataSet(graph, clsFileTrain, clsFileVal, classNameDict, cfg)

        # model
        sess, cost, trainOp, accuracy, centerLoss, distance, lossAll, \
        trainHandleValue, valHandleValue, isTraining, saver= \
            getModel(graph, gpus, image, yTrue, y, trainIterator, trainHandle,
                     valHandle, classNameDict, learningRate, isTraining, cfg, logger)

        # train
        train(sess, cfg, cost, trainOp, accuracy, centerLoss, distance, lossAll,
              trainIterator, valIterator, trainBatchNum, valBatchNum, dataTrainDir,
              dataValDir, clsFileTrain, clsFileVal, classNameDict, valNum,
              handleFlag, trainHandleValue, valHandleValue, learningRate,
              isTraining, logger, saver)
    except Exception as e:
        logger.error(f'Exception occur: {e}')
        raise Exception(f'Exception occur: {e}')




if __name__ == '__main__':
    main()


"""
添加环境变量
export PATH=/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda

--config_project
./data/config/project.yaml 
--config_common
./data/config/common.yaml
"""