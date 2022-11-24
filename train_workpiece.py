# coding: utf-8

from __future__ import division, print_function

import os
import argparse
import tensorflow as tf
import numpy as np
import utils.globalvar as globalvar

from utils.data_utils import parse_data
from utils.misc_utils import shuffle_and_overwrite, config_learning_rate, config_optimizer, \
    average_gradients, get_background_lines
from utils.nms_utils import gpu_nms
from utils.config_parse import get_config as getConfig
from utils.model import yolov3_trt
from utils.logging_util import Logger
from utils.utils import makeDir, dataConfirm, makeValFile


projectPath = os.path.abspath(os.path.dirname(__file__))


def getDataSet(graph, detTrainPath, detValPath, nfsMountDir, dataTempDir,
             trainBatchNum, classNumDict, classNames, classNum, imageSize,
             anchors, batchSize, lrDecayFreq, withGray, cfg):
    with graph.as_default(), tf.device('/cpu:0'):
        # setting placeholders
        isTraining = tf.placeholder(dtype=tf.bool, name="phase_train")
        handleFlag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

        # build Dataset
        shuffle_and_overwrite(detTrainPath)  # 每轮都随机打乱数据集顺序
        trainDataset = tf.data.TextLineDataset(detTrainPath)
        trainDataset = trainDataset.apply(tf.contrib.data.map_and_batch(
            lambda x: tf.py_func(parse_data,
                                 [x, classNames, classNum, imageSize, anchors, 'train', dataTempDir, nfsMountDir, classNumDict, withGray],
                                 [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]),
            num_parallel_calls=cfg.model_set["num_threads"], batch_size=batchSize))
        trainDataset = trainDataset.prefetch(cfg.model_set["prefetech_buffer"])

        valDataset = tf.data.TextLineDataset(detValPath)
        valDataset = valDataset.apply(tf.contrib.data.map_and_batch(
            lambda x: tf.py_func(parse_data,
                                 [x, classNames, classNum, imageSize, anchors, 'val', dataTempDir, nfsMountDir, classNumDict, withGray],
                                 [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]),
            num_parallel_calls=cfg.model_set["num_threads"],
            batch_size=batchSize))
        valDataset.prefetch(cfg.model_set["prefetech_buffer"])

        # add to iterator
        # creating two dataset iterators
        trainIterator = trainDataset.make_initializable_iterator()
        valIterator = valDataset.make_initializable_iterator()

        # creating two dataset handles
        trainHandle = trainIterator.string_handle()
        valHandle = valIterator.string_handle()
        # select a specific iterator based on the passed handle
        datasetIterator = tf.data.Iterator.from_string_handle(
            handleFlag, trainDataset.output_types, trainDataset.output_shapes)

        # register the gpu nms operation for the following evaluation scheme
        predBoxesFlag = tf.placeholder(tf.float32, [1, None, None])
        predScoresFlag = tf.placeholder(tf.float32, [1, None, None])
        predLabelsFlag = tf.placeholder(tf.float32, [1, None, None])
        gpuNmsOp = gpu_nms(predBoxesFlag, predScoresFlag, predLabelsFlag, classNum)

        # iterator config
        globalStep = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        if cfg.model_set["use_warm_up"]:
            learningRate = tf.cond(tf.less(globalStep, trainBatchNum * cfg.model_set["warm_up_epoch"]),
                                   lambda: cfg.model_set["warm_up_lr"],
                                   lambda: config_learning_rate(cfg, globalStep - trainBatchNum * cfg.model_set["warm_up_epoch"], lrDecayFreq))
        else:
            learningRate = config_learning_rate(cfg, globalStep, lrDecayFreq)

        tf.summary.scalar('learningRate', learningRate)

        # build optimizer
        optimizer = config_optimizer(cfg.model_set["optimizer_name"], learningRate)
        # add my clip
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 3.0)

        return datasetIterator, trainIterator, trainHandle, valHandle, \
               optimizer, learningRate, isTraining, globalStep, handleFlag


def getModel(graph, datasetIterator, isTraining, optimizer, globalStep, trainIterator,
             trainHandle, valHandle, pretrainPath, logSavePath, imageSize,
             classNum, anchors, withGray, gpus, cfg, logger):
    with graph.as_default(), tf.device('/cpu:0'):
        # with tf.device(gpus[0]):
        with tf.device(gpus[0]):
            # add to gpu
            towerGrads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for gpuId in range(len(gpus)):
                    with tf.device(gpus[gpuId]):
                        with tf.name_scope('%s_%d' % ('tower', gpuId)) as scope:
                            # get an element from the choosed dataset iterator
                            image, yTrue13, yTrue26, yTrue52, imgPath = datasetIterator.get_next()
                            # 目标中心位于那个anchor,那个anchor就负责检测这个物体,
                            # 他的x,y,w,h就是真是目标的数据,其余的anchor为0
                            yTrue = [yTrue13, yTrue26, yTrue52]
                            # tf.data pipeline will lose the data shape,
                            # so we need to set it manually
                            image.set_shape([None, imageSize[1], imageSize[0], 3])
                            for y in yTrue:
                                y.set_shape([None, None, None, None, None])

                            # Model definition
                            yoloModel = yolov3_trt(
                                classNum, anchors, cfg.model_set["backbone_name"],
                                cfg.model_set["train_with_two_feature_map"])
                            with tf.variable_scope('yolov3'):
                                # model forward get feature maps.
                                predFeatureMaps = yoloModel.forward(
                                    image, is_training=isTraining, train_with_gray=withGray)

                            # compute loss
                            loss = yoloModel.compute_loss(predFeatureMaps, yTrue)

                            tf.get_variable_scope().reuse_variables()
                            # predict
                            yPred = yoloModel.predict(predFeatureMaps)
                            tf.summary.scalar(f"tower_{gpuId}_train_/total_loss", loss[0])
                            tf.summary.scalar(f"tower_{gpuId}_train/loss_xy", loss[1])
                            tf.summary.scalar(f'tower_{gpuId}_train/loss_wh', loss[2])
                            tf.summary.scalar(f'tower_{gpuId}_train/loss_conf', loss[3])
                            tf.summary.scalar(f'tower_{gpuId}_train/loss_class', loss[4])
                            grads = optimizer.compute_gradients(loss[0])
                            towerGrads.append(grads)
            # set model tools
            # restore
            if cfg.model_set["restore_part"] == ['None']:
                saverToRestore = tf.train.Saver(
                    var_list=tf.contrib.framework.get_variables_to_restore(
                        include=[None]))
            else:
                saverToRestore = tf.train.Saver(
                    var_list=tf.contrib.framework.get_variables_to_restore(
                        include=cfg.model_set["restore_part"]))
            # update
            if cfg.model_set["update_part"] == ['None']:
                updateVars = tf.contrib.framework.get_variables_to_restore(
                    include=[None])
            else:
                updateVars = tf.contrib.framework.get_variables_to_restore(
                    include=cfg.model_set["update_part"])
            # save
            saverToSave = tf.train.Saver(max_to_keep=1)  # 只保留最后一个模型

            # set dependencies for BN ops
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                grads = average_gradients(towerGrads)
                applyGradientOp = optimizer.apply_gradients(grads, global_step=globalStep)
                trainOp = tf.group(applyGradientOp)

            # build sess
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False))
            # variables initialize
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer(),
                      trainIterator.initializer])  # local variables 不被存储的变量
            # get handle
            trainHandleValue, valHandleValue = sess.run([trainHandle, valHandle])
            saverToRestore.restore(sess, pretrainPath)
            logger.info(f"load model from: {pretrainPath}")
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logSavePath, sess.graph)

            return sess, yPred, yTrue, loss, writer, trainOp, merged, \
                   trainHandleValue, imgPath, saverToSave


def train(sess, trainOp, merged, yPred, yTrue, loss, globalStep, learningRate,
          imgPath, trainBatchNum, trainIterator, isTraining, handleFlag,
          trainHandleValue, saverToSave, detTrainPath, modelSavePath,
          epoches, detSaveFreq, writer, logger):
    logger.info(f'\n>>>>>>>>>>> start to train >>>>>>>>>>>\n')
    for epoch in range(epoches):
        for i in range(trainBatchNum):
            _, summary, yPred_, yTrue_, loss_, globalStep_, lr, imgPath_ = \
                sess.run([trainOp, merged, yPred, yTrue, loss, globalStep, learningRate, imgPath],
                         feed_dict={isTraining: True, handleFlag: trainHandleValue})
            writer.add_summary(summary, global_step=globalStep_)
            info = f"Epoch: {epoch}, step: {globalStep_}, " \
                   f"loss: {'%.3f'%loss_[0]}, xyLoss: {'%.3f'%loss_[1]}, " \
                   f"whLoss: {'%.3f'%loss_[2]}, confLoss: {'%.3f'%loss_[3]}, " \
                   f"classLoss: {'%.3f'%loss_[4]}, lr:{'%.9f'%lr}"
            logger.info(f"{info}")

            # start to save
            if globalStep_ % int(detSaveFreq*trainBatchNum) == 0 and globalStep_ > 0:
                saverToSave.save(sess, modelSavePath)
                logger.info_ai(f"save model: {detSaveFreq}")

        shuffle_and_overwrite(detTrainPath)
        sess.run(trainIterator.initializer)
    sess.close()
    logger.info('Detection model train finished.')


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
    logger = Logger(logSavePath, "train_detection")
    logger.clean_log_dir()
    logger.info(f"project name is: {cfg.project_name}")
    logger.info(f"config: {cfg}")

    # 创建部分存储路径
    dataTempDir = cfg.data_set["data_save_path_temp"]
    modelSaveDir = os.path.join(
        cfg.data_set["model_and_temp_file_save_path"],
        cfg.data_set["detection_model_save_dir_name"])

    makeDir(dataTempDir, logger=logger)
    makeDir(modelSaveDir, logger=logger)

    # 获取背景图片，如果要贴背景到图片中进行训练的话
    if cfg.data_set["add_background_path"] != "None":
        background_lines = get_background_lines(cfg.data_set["add_background_path"])
        globalvar.set_background_add_lines(background_lines)

    # 设置全局变量
    globalvar.set_logger(logger)
    globalvar.set_config(cfg)

    # 判断必须文件是否存在
    nfsMountDir = cfg.data_set["nfs_mount_path"]
    detTrainPathOr = cfg.data_set["detection_train_file_path"]
    detValPathOr = detTrainPathOr.replace('_train.txt', '_val.txt') \
        if '_train.txt' in detTrainPathOr else detTrainPathOr.replace('.txt', '_val.txt')
    clsClassPath = cfg.data_set["class_name_path"]
    # detClassPath = os.path.join(projectPath, "data_detection",
    #                             cfg.data_set["detection_class_name_path"])
    pretrainPath = os.path.join(projectPath,
                                cfg.data_set["detection_model_restore_path"])
    detDataPath = os.path.join(cfg.data_set["model_and_temp_file_save_path"],
                               "data_detection")
    configSavePath = os.path.join(
        modelSaveDir, f"{cfg.model_set['detection_model_save_name']}.yaml")
    assert os.path.exists(nfsMountDir), \
        "nfs mount path not exit:%s" % nfsMountDir
    assert os.path.exists(detTrainPathOr), \
        "detection train file path not exit:%s" % detTrainPathOr
    # assert os.path.exists(detClassPath), \
    #     "detection class name path not exit:%s" % detClassPath
    assert os.path.exists(clsClassPath), \
        "class name path not exit:%s" % clsClassPath
    assert os.path.exists(pretrainPath + ".meta"), \
        "detection model restore path not exit:%s" % pretrainPath
    if not os.path.exists(detValPathOr):
        makeValFile(detValPathOr, cfg.data_set["detection_val_file_path"])
    if not os.path.exists(detDataPath):
        os.makedirs(detDataPath)
    cfg.save(configSavePath)

    # 参数
    detGpuDevice = cfg.model_set["detection_train_gpu_device"]
    # detGpuDevice = cfg.model_set["other_gpu_device"]
    lrDecayFreqEpoch = cfg.model_set["lr_decay_freq_epoch"]
    batchSize = cfg.model_set["detection_batch_size"]
    imageSize = cfg.model_set["image_size"]
    withGray = cfg.model_set["train_with_gray"]
    detSaveFreq = cfg.model_set["detection_save_freq"]
    detModelName = cfg.model_set["detection_model_save_name"]
    epoches = cfg.model_set["detection_total_epoches"]
    anchorsPath = os.path.join(detDataPath, "anchors.txt")
    detTrainPath = os.path.join(detDataPath, os.path.basename(detTrainPathOr))
    detValPath = os.path.join(detDataPath, os.path.basename(detValPathOr))

    os.system(f'cp {detTrainPathOr} {detTrainPath}')
    os.system(f'cp {detValPathOr} {detValPath}')
    with open(anchorsPath, "w", encoding='utf-8') as af:
        for i, d in enumerate(cfg.data_set["anchors"]):
            anc = f", {'%.2f'%d}" if i else f"{'%.2f'%d}"
            af.write(anc)

    # 设置显卡,os.env是限制访问GPU序号，
    # 程序中实际调用GPU设置tf.device时应从0开始一一对应选则的物理GPU（由小到大）
    os.environ['CUDA_VISIBLE_DEVICES'] = detGpuDevice
    gpus = [f"/gpu:{d}" for d in range(len(detGpuDevice.strip().split(",")))]

    anchors = np.reshape(np.asarray(cfg.data_set["anchors"], np.float32),[-1, 2]) * 2.5
    classNames = [x.split()[0] for x in open(clsClassPath).readlines()]
    logger.info(f"get class name: {classNames} from file: {clsClassPath}")

    # 统计确认训练数据数目，是否保存本地
    classNumDict, trainNum = \
        dataConfirm(
            detTrainPath, detValPath, classNames, nfsMountDir, dataTempDir,
            cfg, logger, batchPath=None)
    logger.info(f"get train class num: {classNumDict}")

    lrDecayFreq = int(lrDecayFreqEpoch * trainNum / (len(gpus) * batchSize))

    # 把不在训练类别里面的类别从类别个数字典中删除
    classNumDict = str({c: 1/(classNumDict[c]+1) for c in classNames})

    # 进行检测训练时，所有目标合并为一个类别
    classNum = 1 if cfg.data_set['mergeLabel'] else len(classNames)
    trainBatchNum = int(np.ceil(float(trainNum) / (batchSize * len(gpus)))) - 2

    # make graph
    graph = tf.Graph()

    # dataset and placeholders
    datasetIterator, trainIterator, trainHandle, valHandle, \
    optimizer, learningRate, isTraining, globalStep, handleFlag = \
        getDataSet(graph, detTrainPath, detValPath, nfsMountDir, dataTempDir,
                 trainBatchNum, classNumDict, classNames, classNum,
                 imageSize, anchors, batchSize, lrDecayFreq, withGray, cfg)

    # model
    sess, yPred, yTrue, loss, writer, trainOp, merged, trainHandleValue, imgPath, saverToSave =\
        getModel(graph, datasetIterator, isTraining, optimizer, globalStep, trainIterator,
                trainHandle, valHandle, pretrainPath, logSavePath, imageSize,
                classNum, anchors, withGray, gpus, cfg, logger)

    # train
    modelSavePath = f"{modelSaveDir}/{detModelName}"
    train(sess, trainOp, merged, yPred, yTrue, loss, globalStep, learningRate,
          imgPath, trainBatchNum, trainIterator, isTraining, handleFlag,
          trainHandleValue, saverToSave, detTrainPath, modelSavePath,
          epoches, detSaveFreq, writer, logger)



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
