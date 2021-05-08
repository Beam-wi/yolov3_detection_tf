import urllib.request
import os

import requests
import urllib.request
import time
import hashlib
from utils.data_utils import parse_line

def requestWithSign(path, params, host="http://ai-openapi.gmm01.com/"): #host="http://openapi.gmmsj.com/"
    fixed_params = {
        "merchant_name": "AI_ADMIN",
        "timestamp": str(int(time.time())),
        "signature_method": "MD5"
    }

    params.update(fixed_params)

    url = host + path
    params["signature"] = sign(params)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
        'Connection': 'close'  # 禁用连续请求，防止IP被禁用
    }
    print("{}?{}".format(url, params))
    response = requests.get(url=url, params=params, headers=headers)
    response.raise_for_status()
    result = response.json()

    if 'data' in result:
        result = result['data']
    else:
        result = None
    return result


def sign(params):
    sigKey = "8HM2NiElGzSIq9nNPtTW0ZH8Vk7YLWRB"
    sigValue = ""
    paraSign = ""
    sortData = {}

    sortData = sorted(params.items(), key=lambda asd: asd[0], reverse=False)

    for item in sortData:
        paraSign = paraSign + item[0] + "=" + str(item[1])

    paraSign = paraSign + sigKey
    paraSign = paraSign.encode()
    print(paraSign)
    sigValue = hashlib.md5(paraSign).hexdigest()

    return sigValue


def getHttp(url):
    page = urllib.request.urlopen(url)
    str = page.read()
    return str


def get_data(logger, batch_list, project_name="sjht", delete_labels=[], nfs_mount_path=""):
    not_data_batch = []
    batch_data = []
    for batch in batch_list:
        batch = batch.split("\n")[0]
        data = requestWithSign("aiadminapi/GetImageRecord/listByWhere", {"batchNo": batch, "pageSize": 10000, "belongBusiness": project_name, "handlerStatus": 2,
                                "status": 1})
        if data is None:
            not_data_batch.append(batch)
            continue

        for children in data:
            img_path = os.path.join(nfs_mount_path, children["path"])
            boxes_list = children["imageLabelRecordList"]
            boxes = []
            labels = []
            for box in boxes_list:
                x0, y0, x1, y1, x2, y2, x3, y3, rotate_angle, start_point = \
                    box["leftTopX"], box["leftTopY"], box["rightTopX"], box["rightTopY"], box["leftBottomX"], box[
                        "leftBottomY"], box["rightBottomX"], box["rightBottomY"], box["rotateAngle"], box["startPoint"]
                labels.append(box["labelType"])
                if box["rotateAngle"] == 0. and box["labelShape"] == 0 and box["rightTopX"] == 0:
                    boxes.append([x0, y0, x3, y3])
                else:
                    boxes.append([min(x0, x1, x2, x3), min(y0, y1, y2, y3), max(x0, x1, x2, x3), max(y0, y1, y2, y3)])

            line = img_path
            for i in range(len(boxes)):
                if labels[i] not in delete_labels:
                    line = line + " %s %d %d %d %d" % (labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                else:
                    pass

            if len(boxes_list) > 0 and (len(line.split(" ")) > 2):
                batch_data.append(line)
            else:
                pass
    logger.info_ai(meg="this batch not have data in download train data", get_ins={"not_data_batch": not_data_batch})

    return batch_data



def get_data_val(logger, batch_list, project_name="sjht", delete_labels=[], nfs_mount_path=""):
    not_data_batch = []
    batch_data = []
    for batch in batch_list:
        batch = batch.split("\n")[0]
        data = requestWithSign("aiadminapi/GetImageRecord/listByWhere", {"batchNo": batch, "pageSize": 10000, "belongBusiness": project_name})
        if data is None:
            not_data_batch.append(batch)
            continue

        for children in data:
            img_path = os.path.join(nfs_mount_path, children["path"])
            # img_path = img_path.replace("images/sjht", "sjht")
            boxes_list = children["imageLabelRecordList"]
            boxes = []
            labels = []

            for box in boxes_list:
                x0, y0, x1, y1, x2, y2, x3, y3, rotate_angle, start_point = \
                    box["leftTopX"], box["leftTopY"], box["rightTopX"], box["rightTopY"], box["leftBottomX"], box[
                        "leftBottomY"], box["rightBottomX"], box["rightBottomY"], box["rotateAngle"], box["startPoint"]
                labels.append(box["labelType"])
                if box["rotateAngle"] == 0. and box["labelShape"] == 0 and box["rightTopX"] == 0:
                    boxes.append([x0, y0, x3, y3])
                else:
                    boxes.append([min(x0, x1, x2, x3), min(y0, y1, y2, y3), max(x0, x1, x2, x3), max(y0, y1, y2, y3)])

            line = img_path
            for i in range(len(boxes)):
                if labels[i] not in delete_labels:
                    line = line + " %s %d %d %d %d" % (labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                else:
                    pass
            if len(boxes_list) > 0 and (len(line.split(" ")) > 2):
                batch_data.append(line)
            else:
                pass
    logger.info_ai(meg="this batch not have data in download val data", get_ins={"not_data_batch": not_data_batch})

    return batch_data


def get_data_background(logger, batch_list, project_name="sjht", delete_labels=[], nfs_mount_path=""):
    not_data_batch = []
    batch_data = []
    for batch in batch_list:
        batch = batch.split("\n")[0]
        data = requestWithSign("aiadminapi/GetImageRecord/listByWhere", {"batchNo": batch, "pageSize": 10000, "belongBusiness": project_name})
        if data is None:
            not_data_batch.append(batch)
            continue

        for children in data:
            img_path = os.path.join(nfs_mount_path, children["path"])
            # img_path = img_path.replace("images/sjht", "sjht")
            boxes_list = children["imageLabelRecordList"]
            boxes = []
            labels = []
            for box in boxes_list:
                x0, y0, x1, y1, x2, y2, x3, y3, rotate_angle, start_point = \
                    box["leftTopX"], box["leftTopY"], box["rightTopX"], box["rightTopY"], box["leftBottomX"], box[
                        "leftBottomY"], box["rightBottomX"], box["rightBottomY"], box["rotateAngle"], box["startPoint"]
                labels.append(box["labelType"])
                if box["rotateAngle"] == 0. and box["labelShape"] == 0 and box["rightTopX"] == 0:
                    boxes.append([x0, y0, x3, y3])
                else:
                    boxes.append([min(x0, x1, x2, x3), min(y0, y1, y2, y3), max(x0, x1, x2, x3), max(y0, y1, y2, y3)])

            line = img_path
            for i in range(len(boxes)):
                if labels[i] not in delete_labels:
                    line = line + " %s %d %d %d %d" % (labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                else:
                    pass
            if len(boxes_list) > 0 and (len(line.split(" ")) > 2):
                batch_data.append(line)
            else:
                batch_data.append(line)
    logger.info_ai(meg="this batch not have data in download background data", get_ins={"not_data_batch": not_data_batch})

    return batch_data


def get_data_from_web_and_save(data_batch_path, data_train_save_path, data_val_save_path, nfs_mount_path, config, logger, project_name="sjht"):
    data_batch_path = data_batch_path.strip()
    assert os.path.exists(data_batch_path), "data batch path is not exit:%s"%data_batch_path
    # assert os.path.exists(data_train_save_path), "data train save path is not exit"
    # assert os.path.exists(data_val_save_path), "data val save path is not exit"
    data_train_save = open(data_train_save_path, "w")
    data_val_save = open(data_val_save_path, "w")

    delete_labels = config.data_set["delete_labels"]
    delete_labels_background = config.data_set["delete_labels_background"]
    batch_lines = [line.strip() for line in open(data_batch_path).readlines()]

    data_train_batch_id = batch_lines.index("#train_data_batch")
    data_train_background_batch_id = batch_lines.index("#train_data_background_batch")
    val_data_batch_id = batch_lines.index("#val_data_batch")
    data_train_batch = batch_lines[data_train_batch_id: data_train_background_batch_id]
    data_train_background_batch = batch_lines[data_train_background_batch_id: val_data_batch_id]
    data_val_batch = batch_lines[val_data_batch_id:]

    data_train = get_data(logger, data_train_batch, project_name=project_name, delete_labels=delete_labels, nfs_mount_path=nfs_mount_path)
    data_train_background = get_data_background(logger, data_train_background_batch, project_name=project_name, delete_labels=delete_labels_background, nfs_mount_path=nfs_mount_path)
    data_val = get_data_val(logger, data_val_batch, project_name=project_name, delete_labels=delete_labels, nfs_mount_path=nfs_mount_path)

    logger.info_ai(meg="download data, train data num:%d, background data num:%d, val data num:%d"%(len(data_train), len(data_train_background), len(data_val)))
    #储存每个类别嵌件个数，以便后续裁剪图片时按概率裁剪
    class_num_dic = {}
    for line in data_train:
        pic_path, boxes, labels = parse_line(line)
        for label in labels:
            if label not in config.data_set["fill_zero_label_names"] and label != "gj":
                if label not in class_num_dic:
                    class_num_dic[label] = 1
                else:
                    class_num_dic[label] += 1

    data_train = data_train + data_train_background

    for line in data_train:
        data_train_save.write(line)
        data_train_save.write("\n")
    data_train_save.close()
    logger.info_ai(meg="train data save in:%s" % data_train_save_path)

    if len(data_val) > 0:
        for line in data_val:
            data_val_save.write(line)
            data_val_save.write("\n")
    data_val_save.close()
    logger.info_ai(meg="val data save in:%s" % data_val_save_path)

    return len(data_train), len(data_val), class_num_dic
