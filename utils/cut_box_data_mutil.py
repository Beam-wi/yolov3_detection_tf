import numpy as np
import cv2
import os
import sys
import time
import datetime
from datetime import datetime
from multiprocessing import Process
from multiprocessing import Manager
import utils.globalvar as globalvar

from utils.data_utils import parse_line, readImg
from utils.utils import processBar


process_over = Manager().list([0 for i in range(32)])


def extension_boxes(image, boxes, extension_ratio=1.2):
    def clamp_bboxs(boxs, img_size, to_remove=1):
        boxs[:, 0] = boxs[:, 0].clip(min=0, max=img_size[0] - to_remove)
        boxs[:, 1] = boxs[:, 1].clip(min=0, max=img_size[1] - to_remove)
        boxs[:, 2] = boxs[:, 2].clip(min=0, max=img_size[0] - to_remove)
        boxs[:, 3] = boxs[:, 3].clip(min=0, max=img_size[1] - to_remove)
        return boxs
    ori_w, ori_h = image.shape[1], image.shape[0]
    x_center = (boxes[:, 2] + boxes[:, 0]) / 2
    y_center = (boxes[:, 3] + boxes[:, 1]) / 2
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]

    x0 = x_center - boxes_w * extension_ratio / 2
    x1 = x_center + boxes_w * extension_ratio / 2
    y0 = y_center - boxes_h * extension_ratio / 2
    y1 = y_center + boxes_h * extension_ratio / 2
    boxes_extension = np.concatenate([x0[:, np.newaxis], y0[:, np.newaxis], x1[:, np.newaxis], y1[:, np.newaxis]], axis=1)
    boxes_extension = clamp_bboxs(boxes_extension, (ori_w, ori_h))
    boxes_extension = np.asarray(boxes_extension, dtype=np.int32)

    return boxes_extension


def cut_box_signal_process(data_lines, process_id, save_path, nfsMountDir, dataTempDir, do_extension_boxes, extension_ratio=1.3,  process_over=None):
    globalvar.globalvar.logger.info(f"{process_id} process start.")
    time.sleep(0.8)
    # save_id = 0
    num = len(data_lines)
    for n, line in enumerate(data_lines):
        try:
            # save_id = save_id + 1
            pic_path_ori, boxes, labels = parse_line(line)
            if len(boxes) == 0:
                continue
            # img = cv2.imread(pic_path_ori)
            img = readImg(pic_path_ori, dataTempDir, nfsMountDir)
            if img is None:
                globalvar.globalvar.logger.info(f"Image is not exits: {pic_path_ori}")
            if do_extension_boxes:
                boxes = extension_boxes(img, boxes, extension_ratio=extension_ratio)
            for i in range(len(boxes)):
                # save_id = save_id + 1
                x0, y0, x1, y1 = boxes[i]
                img_save = img[y0:y1, x0:x1, :]
                if not os.path.exists(f"{save_path}/{labels[i]}"):
                    os.makedirs(f"{save_path}/{labels[i]}")
                cut_img_save_path = f"{save_path}/{labels[i]}/{pic_path_ori.split('/')[-1].replace('.jpg', '').replace('.png', '')}_{process_id}_{i}.jpg"
                cv2.imwrite(cut_img_save_path, img_save)
        except Exception as ex:
            globalvar.globalvar.logger.info(f"Cut box img find Exception: {ex}")
        processBar((n+1) / num, startStr='', endStr=str(num)+'  ')
    process_over[process_id] = 1
    globalvar.globalvar.logger.info(f"{process_id} process id over.")


# 清除所有临时训练图片
def del_dir_all(path):
    if os.path.exists(path):
        names = os.listdir(path)  # 获取当前路径下所有文件
        for name in names:
            delete_path = os.path.join(path, name)
            if os.path.isdir(delete_path):
                os.rmdir(delete_path)
            else:
                os.remove(delete_path)


def cut_data_multi_process(data_path, save_path, nfsMountDir, dataTempDir, do_extension_boxes=False, extension_ratio=1.3):
    global process_over
    process = []
    save_path = save_path.strip()
    num_process = 32
    data_lines = open(data_path).readlines()
    # num_lines = int(len(data_lines) / num_process) + 1
    numLine = int(len(data_lines) / (num_process-1))
    now = datetime.now()
    time_buid = datetime.strftime(now, '%Y%m%d')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path)) != 0:
        globalvar.globalvar.logger.warning(
            f"The data path is not empty: {save_path}")
        dataLists = os.listdir(save_path)
        dataLists.sort()
        lastData = dataLists[-1]
        # 建议不要轻易使用rm指令，以免删除重要文件
        # os.system(f"cd {save_path}")
        # os.system(f"rm -r ./*")
        # globalvar.globalvar.logger.info(
        #     f"The data path has been deleted: {save_path}")
        save_path = os.path.join(save_path, lastData)
        globalvar.globalvar.logger.warning(
            f"The data path will be use to train: {save_path}")
        return
    else:
        save_path = os.path.join(save_path, time_buid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_lines = open(data_path).readlines()
    for i in range(num_process):
        process_lines = data_lines[i * numLine:(i + 1) * numLine] if i < (num_process-1) else data_lines[i * numLine:]
        p = Process(target=cut_box_signal_process, args=(process_lines, i, save_path, nfsMountDir, dataTempDir, do_extension_boxes, extension_ratio, process_over))
        p.start()
        process.append(p)

    for i in range(num_process):
        process[i].join()

    while sum(process_over) != 32:
        time.sleep(5)

    # globalvar.globalvar.logger.info_ai("cut box over", get_ins={"process_over": process_over})
