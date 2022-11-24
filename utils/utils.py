# coding: utf-8

import os
import shutil

from utils.data_utils import parse_line
from utils.get_data_from_web import get_data_from_web_and_save


# 创建文件夹
def makeDir(path, logger=None):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            if logger:
                logger.info(f"Make dir: {path}")
        except Exception as ex:
            if logger:
                logger.error(f"Make dir: {path}, occur: {ex}")
            raise Exception(f"Make dir: {path}, occur: {ex}")


# 进度条
def processBar(percent, startStr='', endStr='', totalLength=15):
    """
    demo:
        for i in range(101):
            time.sleep(0.1)
            processBar(i/100, startStr='', endStr='101', totalLength=15)
    """
    bar = ''.join(
        ["\033[31;41m%s\033[0m" % '   '] * int(percent * totalLength)) + ''
    bar = '\r' + startStr + bar.ljust(totalLength) + ' {:0>4.1f}%|'.format(
        percent * 100) + endStr
    print(bar, end='', flush=True)


# 合并多个文件
def makeValFile(detValPath, valPathList):
    with open(detValPath, 'w', encoding='utf-8') as fd:
        for file in valPathList:
            for line in open(file, "r", encoding='utf-8').readlines():
                fd.write(line)


# 确认训练测试数据是否存到本地，可由batch或file确认
def dataConfirm(trainFile, valFile, classNames, nfsMountDir, dataTempDir,
                cfg, logger, batchPath=None):
    if batchPath:
        try:
            trainNum, valNum, classNumDict = get_data_from_web_and_save(
                batchPath, trainFile, valFile, nfsMountDir, cfg, logger,
                project_name="sjht")
            logger.info(f"get train {trainNum} new data from web")
        except Exception as e:
            logger.error(f'Exception occur when get data from web: {e}')
            raise Exception(f'Exception occur when get data from web: {e}')
    else:
        try:
            dataDict = {"train": trainFile, "val": valFile}
            resultDict = {}
            for mode, file in dataDict.items():
                classNumDict, num = singDataConfig(
                    file, classNames, nfsMountDir, dataTempDir, logger)
                resultDict[mode] = {"c": classNumDict, "n": num}
                logger.info(f"get {mode} {num} new data from {file}\n")
            classNumDict, trainNum = resultDict['train']['c'], resultDict['train']['n']
        except Exception as e:
            logger.error(f'Exception occur when parse local data: {e}')
            raise Exception(f'Exception occur when parse local data: {e}')

    return classNumDict, trainNum


# 单个文件确认
def singDataConfig(file, classNames, nfsMountDir, dataTempDir, logger,
                   timeLimit=50):
    data = open(file).readlines()
    num = len(data)
    classNumDict = {}
    warnTimes = 0
    for i, line in enumerate(data):
        if warnTimes > timeLimit:
            raise Exception(f"Over {warnTimes-1} times error occur "
                            f"when confirm data, check your original images.")
        picPath, boxes, labels = parse_line(line)
        imgOrPath = os.path.join(nfsMountDir, picPath)
        imgTmPath = os.path.join(dataTempDir, picPath)
        if not os.path.exists(imgTmPath):
            # os.system(f"cp {imgOrPath} {imgTmPath}")
            if not os.path.exists(os.path.dirname(imgTmPath)):
                os.makedirs(os.path.dirname(imgTmPath))
            try:
                shutil.copy(imgOrPath, imgTmPath)
            except Exception as e:
                warnTimes += 1
                logger.warning(f'error occur: {e}')
        for label in labels:
            if label in classNames:
                if label not in classNumDict:
                    classNumDict[label] = 1
                else:
                    classNumDict[label] += 1
        processBar((i+1) / num, startStr='', endStr=str(num))

    return classNumDict, num