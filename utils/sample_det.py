import os
import cv2
import json
import shutil

import numpy as np
import utils.globalvar as glo
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
jpeg = TurboJPEG()


def parsePoly(line, clsList, prefix):
    """
    :param
        sing: id$path.jpg%l0 x00 y00 x01 y01 ... %l1 x10 y10 x11 y11 ...
    :return:
        picPath: imPth
        boxes: (ndArray(n, 2), [0, n1, n2, ..., nn]).     (((x, y)), [ind, ...])
        labels: [..., n]
    """
    splits = line.strip().split('$')[0].split('%')
    picPath = splits[0] if not prefix else f"{prefix}/{splits[0]}"
    boxes = np.empty((0, 2), dtype=np.float32)
    boxInds = [0]
    labels = list()
    ind = 0
    for i, mes_ in splits[1:]:
        mes = mes_.split(' ')
        num = (len(mes)-1)/2
        if clsList:
            if mes[0] in clsList + glo.globalvar.config.data_set["fill_zero_label_names"]:
                boxes = np.concatenate(
                    boxes,
                    np.array(list(zip(*(iter(mes[1:]),)*num)), dtype=np.float32), axis=0)
                ind += num
                boxInds.append(ind)
                labels.append(mes[0])
            else:
                pass
        else:
            boxes = np.concatenate(
                boxes,
                np.array(list(zip(*(iter(mes[1:]),) * num)), dtype=np.float32),
                axis=0)
            ind += num
            boxInds.append(ind)
            labels.append(mes[0])

    return picPath, (boxes, boxInds), labels


def parseRect(line, clsList, prefix):
    """
    :param
        sing: id$path.jpg l0 x00 y00 x01 y01 ... ln x10 y10 x11 y11.
    :return:
        picPath: imPth
        boxes: (ndArray(n, 2), [0, n1, n2, ..., nn]).     (((x, y)), [ind, ...])
        labels: [..., n]
    """
    splits = line.strip().split('$')[0].split(' ')
    picPath = splits[0] if not prefix else f"{prefix}/{splits[0]}"
    boxes = np.empty((0, 2), dtype=np.float32)
    boxInds = [0]
    labels = list()
    for i, mes in (zip(*(iter(splits[1:]),)*5)):
        if clsList:
            if mes[0] in clsList + glo.globalvar.config.data_set["fill_zero_label_names"]:
                boxes = np.concatenate(
                    boxes,
                    np.array(
                        [[eval(mes[1]), eval(mes[2])],
                         [eval(mes[3]), eval(mes[2])],
                         [eval(mes[3]), eval(mes[4])],
                         [eval(mes[1]), eval(mes[4])]], dtype=np.float32), axis=0)
                boxInds.append(4 * (i + 1))
                labels.append(mes[0])
            else:
                pass
        else:
            boxes = np.concatenate(
                boxes,
                np.array(
                    [[eval(mes[1]), eval(mes[2])],
                     [eval(mes[3]), eval(mes[2])],
                     [eval(mes[3]), eval(mes[4])],
                     [eval(mes[1]), eval(mes[4])]], dtype=np.float32), axis=0)
            boxInds.append(4*(i+1))
            labels.append(mes[0])

    return picPath, (boxes, boxInds), labels


# 解析单行数据
def parseLine(line, clsList=None, prefix=None):
    line = line.decode() if not isinstance(line, str) else line
    picPath, boxes, labels = parsePoly(line, clsList, prefix) if '%' in line \
        else parseRect(line, clsList, prefix)

    return picPath, boxes, labels


def readImg(picPath, dataTempDir, nfsMountDir):
    if not isinstance(dataTempDir, str):
        dataTempDir = dataTempDir.decode()
    if not isinstance(nfsMountDir, str):
        nfsMountDir = nfsMountDir.decode()
    picPath = picPath.replace(nfsMountDir, '').replace(dataTempDir, '')
    nfsPath = os.path.join(nfsMountDir, picPath)
    localPath = os.path.join(dataTempDir, picPath)
    localDir = os.path.dirname(localPath)

    if not os.path.exists(localDir):
        try:
            os.makedirs(localDir)
            glo.globalvar.logger.info_ai(meg="make dir:%s" % localDir)
        except Exception as ex:
            glo.globalvar.logger.info(
                f"make dir {localDir}, find exception: {ex}")
            pass
    if not os.path.exists(localPath):
        try:
            shutil.copy(nfsPath, localPath)
        except Exception as ex:
            glo.globalvar.logger.info(
                f"copy {nfsPath} to {localPath}, find exception: {ex}")
            pass
    try:
        inFile = open(localPath, 'rb')
        img = jpeg.decode(inFile.read())
        inFile.close()
    except Exception as ex:
        glo.globalvar.logger.warning(
            f"TurboJPEG read {localPath} failed, "
            f"find exception: {ex}, replace cv2.")
        img = cv2.imread(localPath)

    return img


def letterBoxImageCV(image, size, fillValue=0):
    """
    resize image with unchanged aspect ratio using padding
    """
    iw, ih = image.shape[1], image.shape[0]
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    new_image = np.zeros((h, w, 3), np.uint8) + fillValue
    new_image[(h - nh) // 2:(h - nh) // 2 + nh,
    (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    shift = [(w - nw) // 2, (h - nh) // 2]
    return new_image, scale, shift


def detResize(img, boxes, size):
    """
    :param
    size: [w, h]
    """
    H, W, C = img.shape
    offsetH, offsetW = int((size[1]-H)/2), int((size[0]-W)/2)
    if boxes[0].shape[0]:
        boxes[0][:, 0] += offsetW
        boxes[0][:, 1] += offsetH
    img, _, _ = letterBoxImageCV(img, size)

    return img, boxes


def adjustData(img, boxes, labels, fillZeroLabelNames):
    boxesDist = np.empty((0, 2), dtype=np.float32)
    boxesIndDist = [0]
    labelsDist = list()
    if boxes[0].shape[0]:

        ind = 0
        for i, label in enumerate(labels):
            poly = boxes[0][boxes[1][i]:boxes[1][i + 1]]
            if label in fillZeroLabelNames:
                cv2.fillPoly(img, poly, (0, 0, 0))
            else:
                ind += (boxes[1][i+1] - boxes[1][i])
                boxesDist = np.concatenate(boxesDist, poly)
                boxesIndDist.append(ind)
                labelsDist.append(label)
    else:
        pass

    return img, (boxesDist, boxesIndDist), labelsDist


def dataAug():
    pass


def processBox():
    pass


def parse_data(line, clsList, clsNum, imSize, anchors, mode, dataSavePathTemp, nfsMountPath, clsNumDict, trainWithGray):
    if not isinstance(clsNumDict, str):
        clsNumDict = clsNumDict.decode()
    clsNumDict = json.loads(clsNumDict.replace("'", "\""))

    clsListTemp = list(name.decode() for name in clsList)
    picPath, boxes, labels = parseLine(line, clsListTemp, prefix=None)

    img = readImg(picPath, dataSavePathTemp, nfsMountPath)
    OW, OH, OC = glo.globalvar.config.data_set["oriSize"]
    if img is None:
        glo.globalvar.logger.warning(f"read img is None: {picPath} fill with 0")
        img = np.zeros((OH, OW, OC), dtype=np.float32)
        boxes = (np.empty((0, 2), dtype=np.float32), [])
        labels = []

    # 专为小图填充到大图用
    setResize = glo.globalvar.config.data_set["detResize"]
    img, boxes = detResize(img, boxes, setResize) if isinstance(setResize, (list, tuple)) else (img, boxes)
    fillZeroLabelNames = glo.globalvar.config.data_set["fill_zero_label_names"]
    img, boxes, labels = adjustData(img, boxes, labels, fillZeroLabelNames)



    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    boxesTemp = []
    labelsTemp = []
    for i, box in enumerate(boxes):
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(w, box[2])
        box[3] = min(h, box[3])
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:  # 过滤小于2的像素
            pass
        else:
            boxesTemp.append(box)
            labelsTemp.append(labels[i])
    boxes = np.array(boxesTemp)
    labels = labelsTemp
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # do data augmentation here
    if mode.decode() == 'train' and boxes.shape[0] > 0:
        img, boxes, labels, confs = dataAug(picPath, img, boxes, labels, imSize, clsNumDict, trainWithGray, False)

    img = img.astype(np.float32)
    img = img / 255.
    yTrue13, yTrue26, yTrue52 = processBox(picPath, img, boxes, labels, imSize, clsNum, anchors)

    return img, yTrue13, yTrue26, yTrue52, picPath


