from __future__ import division, print_function

import os
import cv2
import csv
import copy
import argparse
import numpy as np
import tensorflow as tf

from collections import Counter
from terminaltables import AsciiTable

from utils.config_parse import get_config as getConfig
from utils.eval_utils import bboxIou
from utils.data_utils import parse_line as parseLine
from utils.logging_util import Logger
import utils.globalvar as globalvar
from utils.test_utils import qianjian_detection as QianjianDetection
from utils.test_utils import draw_box as drawBox
from utils.utils import makeDir, singDataConfig
from utils.data_augmentation import randomRotation


# # 多块使用逗号隔开
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
projectPath = os.path.abspath(os.path.dirname(__file__))


def dataConfirm(valFile, classNames, logger):
    dataVal = open(valFile).readlines()
    valNum = len(dataVal)

    # 储存每个类别嵌件个数，以便后续裁剪图片时按概率裁剪
    classNumDic = {}
    for line in dataVal:
        picPath, boxes, labels = parseLine(line)
        for label in labels:
            if label in classNames:
                if label not in classNumDic:
                    classNumDic[label] = 1
                else:
                    classNumDic[label] += 1

    logger.info(f"get {os.path.basename(valFile).split('_')[0]} class: "
                f"{classNumDic}")
    return classNumDic, valNum


def dataFilter(img, boxes, labels, filterArea, classNumDict):
    """
    过滤掉不检的label，将干扰区填充0
    """
    if len(boxes) == 0:
        return img, boxes, labels

    xCenter = (boxes[:, 0] + boxes[:, 2]) / 2
    yCenter = (boxes[:, 1] + boxes[:, 3]) / 2
    deleteIds = list()
    for id, label in enumerate(labels):
        if label in filterArea:
            img[int(boxes[id][1]):int(boxes[id][3]), int(boxes[id][0]):int(boxes[id][2]), :] = 0
            for i in range(len(boxes)):
                if boxes[id][0] <= xCenter[i] <= boxes[id][2] and boxes[id][1] <= yCenter[i] <= boxes[id][3]:
                    if i not in deleteIds:
                        deleteIds.append(i)
        elif label not in classNumDict.keys():
            if id not in deleteIds:
                deleteIds.append(id)

    boxesTemp = [boxes[id] for id in range(len(labels)) if id not in deleteIds]
    labelsTemp = [labels[id] for id in range(len(labels)) if id not in deleteIds]

    boxes = np.asarray(boxesTemp, np.float32)
    labels = labelsTemp

    return img, boxes, labels


def extensionBoxes(boxes, ratio=1.2):
    centerX = (boxes[:, 2] + boxes[:, 0]) / 2
    centerY = (boxes[:, 3] + boxes[:, 1]) / 2
    boxesW = boxes[:, 2] - boxes[:, 0]
    boxesH = boxes[:, 3] - boxes[:, 1]

    x0 = centerX - boxesW * ratio / 2
    x1 = centerX + boxesW * ratio / 2
    y0 = centerY - boxesH * ratio / 2
    y1 = centerY + boxesH * ratio / 2
    boxesExt = np.concatenate([x0[:, np.newaxis], y0[:, np.newaxis], x1[:, np.newaxis], y1[:, np.newaxis]],
                                     axis=1)
    boxesExt = np.asarray(boxesExt, dtype=np.int32)

    return boxesExt


def clampBboxs(boxes, imgSize, toRemove=1):
    boxes[:, 0] = boxes[:, 0].clip(min=0, max=imgSize[0] - toRemove)
    boxes[:, 1] = boxes[:, 1].clip(min=0, max=imgSize[1] - toRemove)
    boxes[:, 2] = boxes[:, 2].clip(min=0, max=imgSize[0] - toRemove)
    boxes[:, 3] = boxes[:, 3].clip(min=0, max=imgSize[1] - toRemove)

    return boxes


def modelLoad(gpus, cfg, logger):
    with tf.device(gpus[0]):
        try:
            model = QianjianDetection(projectPath)
        except Exception as ex:
            logger.error(f"Model generate failed, error occur {ex}")
            raise Exception(f"Model generate failed, error occur {ex}")
        if len(model.class_name_classify) == 1:
            # cfg.test_other_info_set["do_classify"] = False
            pass
        try:
            model.model_init_detection()
        except Exception as ex:
            logger.error(f"Detection model init failed, error occur {ex}")
            raise Exception(f"Detection model init failed, error occur {ex}")

        if cfg.test_other_info_set["do_classify"]:
            try:
                model.model_init_classify()
            except Exception as ex:
                logger.info(f"Classify model init failed, error occur {ex}")
                raise Exception(f"Classify model init failed, error occur {ex}")

        return model


def statistic(
        cfg, model, boxes, labels, detClsScore, boxesDet, labelsDet,
        scoresDet, labelsCls,  scoresCls, imgOri, imgPath, workDir,
        statisticDict, exten=0):
    H, W, _ = imgOri.shape
    # if not cfg.test_other_info_set["do_classify"]:
    #     labels = ["qj"] * len(labels)
    # else:
    #     labels = [model.class_name_classify[model.name2id_classify[l]] for l in labels]
    labels = [model.class_name_classify[model.name2id_classify[l]] for l in labels]
    boxes = np.array(boxes, dtype=np.float32)

    wriSta = True
    for key, value in detClsScore.items():
        selectId = [i for i, s in enumerate(zip(scoresDet, scoresCls)) if s[0] >= value[0] and s[1] >= value[1]]

        boxesDetectionSelect = boxesDet[selectId]
        scoresDetectionSelect = scoresDet[selectId]
        labelsDetectionNameSelect = labelsDet[selectId]
        scoresClassifySelect = scoresCls[selectId]
        labelsClassifyNameSelect = labelsCls[selectId]

        predBoxes = boxesDetectionSelect
        # predBoxes = extension_boxes(boxesDetectionSelect, extension_ratio=1.5)
        trueBoxes = copy.deepcopy(boxes).reshape((-1, 4))
        trueLabelsList = labels

        if cfg.test_other_info_set["do_classify"]:
            predLabelsList = labelsClassifyNameSelect.tolist()
        else:
            # predLabelsList = labelsDetectionNameSelect.tolist()
            predLabelsList = labelsDetectionNameSelect

        # 计算pred、true混淆矩阵
        iouThr = cfg.test_other_info_set["test_iou_thresh"]
        iouMatrix, tmwDict = bboxIou(predBoxes,
                                     trueBoxes,
                                     label_a=predLabelsList,
                                     label_b=trueLabelsList,
                                     scoresDet=scoresDetectionSelect,
                                     scoresCls=scoresClassifySelect,
                                     iou_thr=iouThr)

        # 统计预测情况
        true = Counter(trueLabelsList)
        pred = Counter(predLabelsList)
        trueLabel = Counter(tmwDict['T']['l'])
        wrongLabel = Counter(tmwDict['W']['l'])
        miss = Counter(tmwDict['M']['l'])
        missLabel = {c: 0 for c in statisticDict[key]['gtDict'].keys()}
        for c in missLabel.keys():
            if c in true:
                if c in miss:
                    missLabel[c] = true[c] - miss[c]
                else:
                    missLabel[c] = true[c]

        for cls in statisticDict[key]['gtDict'].keys():
            statisticDict[key]["gtDict"][cls] += true[cls] if cls in true else 0
            statisticDict[key]["prDict"][cls] += pred[cls] if cls in pred else 0
            statisticDict[key]["posDict"][cls] += trueLabel[cls] if cls in trueLabel else 0
            statisticDict[key]["negDict"][cls] += wrongLabel[cls] if cls in wrongLabel else 0
            statisticDict[key]["findDict"][cls] += missLabel[cls]

        # 绘制预测有误全图
        if cfg.test_other_info_set['writeWFull']:
            if wriSta:
                if tmwDict['M']['b'].shape[0] > 0 or tmwDict['W']['b'].shape[0] > 0:
                    drawBox(copy.deepcopy(imgOri),
                            trueBoxes,
                            trueLabelsList, predBoxes, predLabelsList,
                            scoresDetectionSelect,
                            scoresClassifySelect, tmwDict,
                            img_test_path=imgPath,
                            img_path_dir=f"{workDir}/wrong_full")
                wriSta = False
        else:
            if wriSta:
                drawBox(copy.deepcopy(imgOri),
                        trueBoxes,
                        trueLabelsList, predBoxes, predLabelsList,
                        scoresDetectionSelect,
                        scoresClassifySelect, tmwDict,
                        img_test_path=imgPath,
                        img_path_dir=f"{workDir}/wrong_full")

                wriSta = False

        if cfg.test_other_info_set["writeWBox"]:
            # 存储漏报box
            missPath = os.path.join(workDir, 'wrong_box', key, 'miss')
            missBoxes = clampBboxs(
                extensionBoxes(tmwDict['M']['b'], ratio=exten), (W, H)) if exten else tmwDict['M']['b']
            for i, c in enumerate(tmwDict['M']['l']):
                if not os.path.exists(os.path.join(missPath, c)):
                    os.makedirs(os.path.join(missPath, c))
                imgName = f"{i}_{os.path.basename(imgPath)}"
                xMin, yMin, xMax, yMax = missBoxes[i]
                imgCut = imgOri[int(yMin):int(yMax), int(xMin):int(xMax), :]
                cv2.imwrite(os.path.join(missPath, c, imgName), imgCut)

            # 存储误报box
            wrongPath = os.path.join(workDir, 'wrong_box', key, 'wrong')
            wrongBoxes = clampBboxs(
                extensionBoxes(tmwDict['W']['b'], ratio=exten), (W, H)) if exten else tmwDict['W']['b']
            for i, c in enumerate(tmwDict['W']['l']):
                if not os.path.exists(os.path.join(wrongPath, c)):
                    os.makedirs(os.path.join(wrongPath, c))
                tl, sd, sc = tmwDict['W']['tl'][i], tmwDict['W']['sd'][i], tmwDict['W']['sc'][i]
                imgName = f"tpdc_{tl}_{c}_{'%.5f'%sd}_{'%.5f'%sc}_{i}_{os.path.basename(imgPath)}"
                xMin, yMin, xMax, yMax = wrongBoxes[i]
                if yMax-yMin and xMax-xMin:
                    imgCut = imgOri[int(yMin):int(yMax), int(xMin):int(xMax), :]
                    cv2.imwrite(os.path.join(wrongPath, c, imgName), imgCut)


def makeCsv(cfg, logger, classNumDict, statisticDict, workDir):
    # 记录各类别最优结果
    optimum = {c: {'det_thr': None, 'cls_thr': None, 'r': 0, 'p': 0} for c in classNumDict.keys()}
    f = open(f"{workDir}/{cfg.data_set['csv_save_dir_name']}.csv", 'w', newline='', encoding='utf-8')
    csvWriter = csv.writer(f)
    for key, value in statisticDict.items():
        gtDict = np.array(list(value["gtDict"].values()), np.float32)
        prDict = np.array(list(value["prDict"].values()), np.float32)
        posDict = np.array(list(value["posDict"].values()), np.float32)
        negDict = np.array(list(value["negDict"].values()), np.float32)
        findDict = np.array(list(value["findDict"].values()), np.float32)

        recall = posDict / gtDict
        wrong = negDict / gtDict
        find = findDict / gtDict
        miss = 1 - find  # find里已经包含了错误识别得框
        precision = posDict / prDict

        recallAv = recall.mean()
        wrongAv = wrong.mean()
        findAv = find.mean()
        missAv = miss.mean()
        precisionAv = precision.mean()

        csvWriter.writerow(["det_cls", f"{key}"])
        csvWriter.writerow(["className", "errorRate", "missRate", "Recall", "Precision"])
        for i, c in enumerate(classNumDict.keys()):
            csvWriter.writerow([c, wrong[i], miss[i], recall[i], precision[i]])
            if recall[i]+precision[i] >= optimum[c]['r']+optimum[c]['p']:
                optimum[c]['r'], optimum[c]['p'] = recall[i], precision[i]
                optimum[c]['det_thr'], optimum[c]['cls_thr'] = key.split("_")
        csvWriter.writerow(['average', wrongAv, missAv, recallAv, precisionAv])
        csvWriter.writerow('\n\n')
    f.close()

    recallLimit = cfg.test_other_info_set["recall_threshold"]
    precisionLimit = cfg.test_other_info_set["precision_threshold"]
    passData = [['className', 'detThr', 'clsThr', 'Recall', 'Precision']]
    faultData = [['className', 'detThr', 'clsThr', 'Recall', 'Precision']]
    passPre, passRec = np.empty([0], np.float32), np.empty((0,), np.float32)
    faultPre, faultRec = np.empty([0], np.float32), np.empty((0,), np.float32)
    # for c, v in optimum.items():
    for c in sorted(optimum):
        v = optimum[c]
        if v['r'] >= recallLimit and v['p'] >= precisionLimit:
            passData.append([c]+list(v.values()))
            passPre = np.concatenate((passPre, np.array([v['p']], np.float32)), axis=-1)
            passRec = np.concatenate((passRec, np.array([v['r']], np.float32)), axis=-1)
        else:
            faultPre = np.concatenate((faultPre, np.array([v['p']], np.float32)), axis=-1)
            faultRec = np.concatenate((faultRec, np.array([v['r']], np.float32)), axis=-1)
            faultData.append([c]+list(v.values()))
    passData.append(['average', '-', '-', passRec.mean(), passPre.mean()])
    faultData.append(['average', '-', '-', faultRec.mean(), faultPre.mean()])
    tablePass, tableFault = AsciiTable(passData), AsciiTable(faultData)
    tablePass.inner_footing_row_border = True
    tableFault.inner_footing_row_border = True
    logger.info('\nPass:\n' + tablePass.table + '\n')
    logger.info('\nFault:\n' + tableFault.table)

    with open(f"{workDir}/default_gj.txt", 'w', encoding='utf-8') as ft:
        ft.write('Pass:\n'+tablePass.table+'\n\n\n\n')
        ft.write('Fault:\n'+tableFault.table)


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
    logger = Logger(logSavePath, "model_test")
    logger.clean_log_dir()
    logger.info(f"project name is: {cfg.project_name}")
    logger.info(f"config: {cfg}")

    # 创建部分存储路径
    dataTempDir = cfg.data_set["data_save_path_temp"]
    savePath = os.path.join(cfg.data_set["model_and_temp_file_save_path"],
                             cfg.data_set["combine_test_result_save_dir_name"])
    # wrong_label_path = os.path.join(save_path,
    #                                 cfg.data_set["wrong_label_save_dir_name"])
    # wrong_img_path = os.path.join(save_path, 'wrong_full_img')
    # csv_path = os.path.join(save_path, cfg.data_set["csv_save_dir_name"])

    makeDir(dataTempDir, logger=logger)
    makeDir(savePath, logger=logger)
    # makeDir(wrong_label_path, logger=logger)
    # makeDir(wrong_img_path, logger=logger)
    # makeDir(csv_path, logger=logger)

    # 设置全局变量
    globalvar.set_logger(logger)
    globalvar.set_config(cfg)

    # 判断必须文件是否存在
    nfsMountDir = cfg.data_set["nfs_mount_path"]
    detClassPath = os.path.join(projectPath, cfg.data_set["detection_class_name_path"])

    # assert os.path.exists(cfg.data_set["detection_val_file_path"]), \
    #     "detection_val_file_path not exit:%s" % cfg.data_set["detection_val_file_path"]
    assert os.path.exists(nfsMountDir), \
        "nfs mount path not exit:%s" % nfsMountDir
    assert os.path.exists(detClassPath), \
        "detection_class_name_path not exit:%s" % detClassPath
    assert os.path.exists(cfg.data_set["class_name_path"]), \
        "class_name_path not exit:%s" % cfg.data_set["class_name_path"]
    classNames = [x.split()[0] for x in open(cfg.data_set["class_name_path"]).readlines()]
    assert not len(classNames) == 0, 'class name file can not be empty.'

    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.model_set["other_gpu_device"]
    gpus = [f"/gpu:{d}" for d in cfg.model_set["other_gpu_device"].strip().split(",")]

    # load model 默认使用第一个GPU
    try:
        model = modelLoad(gpus, cfg, logger)
    except Exception as e:
        logger.error(f"Exception occur when loadModel: {e}")
        raise Exception(f"Exception occur when loadModel: {e}")

    # 按每个工件检测
    for valFile in cfg.data_set['detection_val_file_path']:
        workpieceName = os.path.basename(valFile).split("_")[0]
        workDir = f"{savePath}/{workpieceName}"
        if not os.path.exists(workDir):
            os.makedirs(workDir)
        cfg.save(f"{workDir}/config.yaml")
        # 确认测试数据
        # classNumDict, _ = dataConfirm(valFile, classNames, logger)
        try:
            classNumDict, num = singDataConfig(
                valFile, classNames, nfsMountDir, dataTempDir, logger)
        except Exception as e:
            logger.error(f"exception occur when config data {valFile}, {e}")
            raise Exception(f"exception occur when config data {valFile}, {e}")
        # assert not classNumDict == {}, 'No match label in the val_file.'
        testData = open(valFile).readlines()
        logger.info(f"test data num: {num}")

        detScores = np.arange(0.05, 1, 0.05).tolist()
        clsScores = np.arange(0.05, 1, 0.05).tolist()
        # detScores = np.arange(0.55, 1, 0.05).tolist()
        # clsScores = np.arange(0.55, 1, 0.05).tolist()
        if cfg.test_other_info_set["show_wrong_data"]:
            detScores = cfg.test_other_info_set["detection_threshold_for_show"]
            clsScores = cfg.test_other_info_set["classify_threshold_for_show"]
        detClsScore = {
            f"{'%.2f' % ds}_{'%.2f' % cs}": [float('%.2f' % ds), float('%.2f' % cs)] for
            ds in detScores for cs in clsScores}

        # 制作统计各类别预测数量字典
        keys = ["gtDict", "prDict", "posDict", "negDict", "findDict"]
        statisticDict = \
            {s: {k: {c: 0 for c in classNumDict.keys()} for k in keys} for s in detClsScore.keys()}

        wrongNum = 0
        for testId, line in enumerate(testData):
            logger.info(f"Current id: {testId}, all num: {num}")
            imgPath, boxes, labels = parseLine(line, prefix=dataTempDir)
            if not os.path.exists(imgPath):
                logger.info(f"Image file: {imgPath} is not exit")
                wrongNum += 1
                continue

            imgOri = cv2.imread(imgPath)
            if imgOri is None:
                continue
            # imgOri, boxes, labels = randomRotation(imgOri, boxes, labels, angle_value=8)  # 随机旋转

            img, boxes, labels = dataFilter(
                imgOri, boxes, labels, cfg.data_set["fill_zero_label_names"], classNumDict)

            if not set(labels).issubset(set(model.name2id_classify.keys())):
                logger.info(f"{str(set(labels) - set(model.name2id_classify.keys()))} "
                            f"not in class names: %s, in the file: {imgPath}")
                wrongNum += 1
                continue

            boxesDet, scoresDet, labelsDet, scoresCls, labelsCls = \
                model.forward(img, do_classify=cfg.test_other_info_set["do_classify"])

            # 统计预测情况
            statistic(cfg, model, boxes, labels, detClsScore, boxesDet,
                      labelsDet, scoresDet, labelsCls, scoresCls, imgOri,
                      imgPath, workDir, statisticDict, exten=2)

        # 保存cvs并获取最优条件下的评分标准
        makeCsv(cfg, logger, classNumDict, statisticDict, workDir)


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