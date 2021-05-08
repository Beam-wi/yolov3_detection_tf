import cv2
import random
import copy
import os
import math
import numpy as np
import albumentations as A
from utils.eval_utils import calc_iou
from scipy.interpolate import UnivariateSpline
import utils.globalvar as globalvar


def resize_image_and_correct_boxes(img, boxes, img_size):
    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:
        img = np.expand_dims(img, -1)
    ori_height, ori_width = img.shape[:2]
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))
    # convert to float
    # xmin, xmax
    for id in range(len(boxes)):
        boxes[id, 0] = boxes[id, 0] / ori_width * new_width
        boxes[id, 2] = boxes[id, 2] / ori_width * new_width
        boxes[id, 1] = boxes[id, 1] / ori_height * new_height
        boxes[id, 3] = boxes[id, 3] / ori_height * new_height
    return img, boxes

def img_rotation(imgrgb, angle):
    '''
    Rotate a image
    :param imgrgb: image which wait for rotating
    :param angle: rotate angle
    :return: image rotated
    '''
    rows, cols, channel = imgrgb.shape
    rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotation = cv2.warpAffine(imgrgb, rotation, (cols, rows), borderValue=125)
    return img_rotation


def img_blur(imgrgb):
    '''
    Randomly different grade fuzzy process
    :param imgrgb: image wait for fuzziing process
    :return: fuzzied image
    '''
    choice_list = [3, 3]
    my_choice = random.sample(choice_list, 1)
    img_blur = cv2.blur(imgrgb, (my_choice[0], my_choice[0]))
    return img_blur


def img_addweight(imgrgb):
    '''
    Randomly mixed weighting
    :param imgrgb: image wait for mixing
    :return: mixed image
    '''
    choice_list = [i * 10 for i in range(1, 18)]
    my_choice = random.sample(choice_list, 1)
    blur = cv2.GaussianBlur(imgrgb, (0, 0), my_choice[0])
    img_addweight = cv2.addWeighted(imgrgb, 1.5, blur, -0.5, 0)
    return img_addweight


def img_addcontrast_brightness(imgrgb):
    '''
    Randomly add bright
    :param imgrgb: image wait for adding bright
    :return: image added bright
    '''
    a = random.sample([i / 10 for i in range(4, 13)], 1)[0]
    g = random.sample([i for i in range(0, 3)], 1)[0]
    h, w, ch = imgrgb.shape
    src2 = np.zeros([h, w, ch], imgrgb.dtype)
    img_bright = cv2.addWeighted(imgrgb, a, src2, 1 - a, g)
    return img_bright

def flip_rl(img, boxes, labels):
    h, w = img.shape[0], img.shape[1]
    # chage boxes
    if len(boxes) > 0:
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = w - oldx2
        boxes[:, 2] = w - oldx1
    else:
        pass

    # change img
    img = img[:, ::-1, :]

    return img, boxes, labels

def flip_ud(img, boxes, labels):
    h, w = img.shape[0], img.shape[1]
    # chage boxes
    if len(boxes) > 0:
        oldy1 = boxes[:, 1].copy()
        oldy2 = boxes[:, 3].copy()
        boxes[:, 1] = h - oldy2
        boxes[:, 3] = h - oldy1
    else:
        pass
    # change img
    img = img[::-1, :, :]

    return img, boxes, labels

def cut_random(img_path, img, boxes, labels, img_size, cut_size_boundary=10, cut_rate = 0.2):
    h, w = img.shape[0], img.shape[1]
    if len(boxes) > 0:
        h_up = np.min(boxes[:,1])
        h_down = np.max(boxes[:,3])
        w_left = np.min(boxes[:,0])
        w_right = np.max(boxes[:,2])
        boxes[:, 0] = boxes[:, 0] + 1
        boxes[:, 2] = boxes[:, 2] - 1
        boxes[:, 1] = boxes[:, 1] + 1
        boxes[:, 3] = boxes[:, 3] - 1
    else:
        h_up = 0
        h_down = h
        w_left = 0
        w_right = w


    h_up_cut = 0
    h_down_cut = 0
    w_left_cut = 0
    w_right_cut = 0
    find = False
    while(not find):
        h_up_cut = random.choice(range(min(max(int(h_up - cut_size_boundary), 1), int(h * cut_rate))))
        h_down_cut = random.choice(range(max(min(int(h_down + cut_size_boundary), h-1), int(h * (1-cut_rate))), h))
        w_left_cut = random.choice(range(min(max(int(w_left - cut_size_boundary), 1), int(w * cut_rate))))
        w_right_cut = random.choice(range(max(min(int(w_right + cut_size_boundary), w-1), int(w * (1-cut_rate))), w))
        if 0.8<(w-(w_right_cut-w_left_cut))/(h-(h_down_cut-h_up_cut)) <1.2:
            find = True

    img = img[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
    # h_new, w_new = img.shape[0], img.shape[1]
    if len(boxes) > 0:
        boxes[:, 0] = boxes[:, 0] - w_left_cut
        boxes[:, 2] = boxes[:, 2] - w_left_cut
        boxes[:, 1] = boxes[:, 1] - h_up_cut
        boxes[:, 3] = boxes[:, 3] - h_up_cut
    else:
        pass

    return img, boxes, labels

def cut_random_sjht(img, boxes, labels, img_size, cut_size_boundary=10, cut_rate = 0.2):
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    h_up = np.min(boxes[:,1])
    h_down = np.max(boxes[:,3])
    w_left = np.min(boxes[:,0])
    w_right = np.max(boxes[:,2])

    boxes[:, 0] = boxes[:, 0] + 1
    boxes[:, 2] = boxes[:, 2] - 1
    boxes[:, 1] = boxes[:, 1] + 1
    boxes[:, 3] = boxes[:, 3] - 1

    h_up_cut = random.choice(range(min(max(int(h_up - cut_size_boundary), 1), int(h * cut_rate))))
    h_down_cut = random.choice(range(max(min(int(h_down + cut_size_boundary), h-1), int(h * (1-cut_rate))), h))
    w_left_cut = random.choice(range(min(max(int(w_left - cut_size_boundary), 1), int(w * cut_rate))))
    w_right_cut = random.choice(range(max(min(int(w_right + cut_size_boundary), w-1), int(w * (1-cut_rate))), w))

    img = img[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
    h_new, w_new = img.shape[0], img.shape[1]

    dst = np.full(shape=(h, w, c), fill_value=0, dtype=img.dtype)
    off_h = random.randint(0, h - h_new)
    off_w = random.randint(0, w - w_new)
    dst[off_h:off_h+h_new, off_w:off_w+w_new, :] = img
    img = dst

    boxes[:, 0] = boxes[:, 0] - w_left_cut + off_w
    boxes[:, 2] = boxes[:, 2] - w_left_cut + off_w
    boxes[:, 1] = boxes[:, 1] - h_up_cut + off_h
    boxes[:, 3] = boxes[:, 3] - h_up_cut + off_h

    return img, boxes, labels

def rotation_random(img, boxes, labels):
    angle = random.choice([90, 180])
    h, w = img.shape[0], img.shape[1]
    if angle == 90:
        img = img_rotation(img, -90)
        if len(boxes)>0:
            boxe_tem = boxes.copy()
            boxes[:, 0] = h - boxe_tem[:, 3]
            boxes[:, 1] = boxe_tem[:, 0]
            boxes[:, 2] = h - boxe_tem[:, 1]
            boxes[:, 3] = boxe_tem[:, 2]
        else:
            pass
    else:
        img = img_rotation(img, -180)
        if len(boxes) > 0:
            boxe_tem = boxes.copy()
            boxes[:, 0] = w - boxe_tem[:, 2]
            boxes[:, 1] = h - boxe_tem[:, 3]
            boxes[:, 2] = w - boxe_tem[:, 0]
            boxes[:, 3] = h - boxe_tem[:, 1]
        else:
            pass

    return img, boxes, labels

def seamless(img, box, obj, boxes = None):
    success = True
    try:
        obj_resize = cv2.resize(obj, (box[2] - box[0], box[3] - box[1]))
        img[box[1]:box[3], box[0]:box[2], :] = obj_resize
    except:
        success = False
    return img, success

def color_random(img):
    b, g, r = cv2.split(img)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    b_ratio = 0
    g_ratio = 0
    r_ratio = 0
    option_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)
    find = False
    while (not find):
        b_ratio = random.sample(option_list, 1)[0]
        g_ratio = random.sample(option_list, 1)[0]
        r_ratio = random.sample(option_list, 1)[0]
        if (b_ratio + g_ratio + r_ratio) > 1.5 and b_ratio > 0.3 and g_ratio > 0.3 and r_ratio > 0.3:
            find = True
    b = b * b_ratio
    g = g * g_ratio
    r = r * r_ratio
    b = b.astype(np.uint8)
    g = g.astype(np.uint8)
    r = r.astype(np.uint8)
    img = cv2.merge([b,g,r])
    return img

def random_color_distort(img, brightness_delta=32, hue_vari=25, sat_vari=0.5, val_vari=0.5):
# def random_color_distort(img, brightness_delta=2, hue_vari=2, sat_vari=0.2, val_vari=0.2):
    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img

def occlusion_random(img, boxes, labels, cut_size_boundary=20):
    h, w = img.shape[0], img.shape[1]
    choice_list = [0, 1]
    for i in range(labels.shape[0]):
        choice = random.sample(choice_list, 1)[0]
        if choice == 0:
            box = boxes[i]
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            w_temp = x1-x0
            h_temp = y1-y0
            if w_temp > h_temp and w_temp >=4:
                add_w = max(random.choice(range(int(w_temp*1/4), int(w_temp*2/3))), 2)
                # print(add_w)
                find = False
                box_temp = np.array([[0, 0, 0, 0]])
                while(not find):
                    x_temp = random.choice(range(cut_size_boundary, w - add_w - cut_size_boundary))
                    y_temp = random.choice(range(cut_size_boundary, h - h_temp - cut_size_boundary))
                    box_temp = np.array([[x_temp, y_temp, x_temp + add_w, y_temp + h_temp]])
                    iou_temp = calc_iou(box_temp, boxes)
                    if np.sum(iou_temp) == 0:
                        find = True
                # print(box_temp[0])
                choice = random.sample(choice_list, 1)[0]
                if choice == 0:
                    # img[y0:y1, x0:x0+add_w, :] = img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :]
                    img, success = seamless(img, [x0, y0, x0+add_w, y1], img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :], boxes)
                    if success:
                        boxes[i][0] = boxes[i][0] + add_w
                    else:
                        pass
                else:
                    # img[y0:y1, x1-add_w:x1, :] = img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :]
                    img, success = seamless(img, [x1-add_w, y0, x1, y1],
                                   img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :], boxes)
                    if success:
                        boxes[i][2] = boxes[i][2] - add_w
                    else:
                        pass

            elif w_temp <= h_temp and h_temp >=4:
                add_h = max(random.choice(range(int(h_temp*1/4), int(h_temp*2/3))), 2)

                find = False
                box_temp = np.array([[0, 0, 0, 0]])
                while (not find):
                    x_temp = random.choice(range(cut_size_boundary, w - cut_size_boundary - w_temp))
                    y_temp = random.choice(range(cut_size_boundary, h - cut_size_boundary - add_h))
                    box_temp = np.array([[x_temp, y_temp, x_temp + w_temp, y_temp + add_h]])
                    iou_temp = calc_iou(box_temp, boxes)
                    if np.sum(iou_temp) == 0:
                        find = True
                choice = random.sample(choice_list, 1)[0]
                # print(box_temp[0])
                if choice == 0:
                    # img[y0:y0+add_h, x0:x1, :] = img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :]
                    img, success = seamless(img, [x0, y0, x1, y0+add_h],
                                   img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :], boxes)
                    if success:
                        boxes[i][1] = boxes[i][1] + add_h
                    else:
                        pass
                else:
                    # img[y1-add_h:y1, x0:x1, :] = img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :]
                    img, success = seamless(img, [x0, y1-add_h, x1, y1],
                                   img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :], boxes)
                    if success:
                        boxes[i][3] = boxes[i][3] - add_h
                    else:
                        pass
            else:
                pass
    # print("+++")
    return img, boxes, labels

def judge_box(obj, img_path, img, boxes):
    h, w = img.shape[0], img.shape[1]
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        if x0 < 0 or y0< 0 or x1 > w or y1 > h:
            cv2.imwrite("wrong.jpg", img)

def find_id(boxes, box):
    for id in range(boxes.shape[0]):
        if np.sum(boxes[id] - box) == 0:
            return id

def merge_box(boxes, labels, confs, threshold = 0.4):
    boxes_temp = []
    boxes_rest = []
    id_temp = []
    for id in range(boxes.shape[0]):
        if id not in id_temp:
            for i in range(boxes[id+1:].shape[0]):
                if id+i+1 not in id_temp:
                    boxes_rest.append(boxes[id+i+1])
            if len(boxes_rest) > 0:
                box_temp = np.array([[boxes[id][0], boxes[id][1], boxes[id][2], boxes[id][3]]])
                iou_temp = calc_iou(box_temp, boxes_rest)[0].tolist()
                iou_max = max(iou_temp)
                if iou_max > threshold:
                    id_temp.append(find_id(boxes, boxes_rest[iou_temp.index(iou_max)]))
                    box_delete = boxes_rest[iou_temp.index(iou_max)]
                    x1 = min(boxes[id][0], box_delete[0])
                    y1 = min(boxes[id][1], box_delete[1])
                    x2 = max(boxes[id][2], box_delete[2])
                    y2 = max(boxes[id][3], box_delete[3])
                    boxes_temp.append([x1, y1, x2, y2])
                else:
                    boxes_temp.append(boxes[id])
            else:
                boxes_temp.append(boxes[id])
        boxes_rest = []
    boxes_temp = np.array(boxes_temp)
    labels_temp = np.array([0]*len(boxes_temp))
    confs_temp = np.array([1]*len(boxes_temp))

    return boxes_temp, labels_temp, confs_temp


def random_change_contrast(img):
    rate = np.random.uniform(0.8, 1.5)
    add = np.random.uniform(-50, 50)
    img = np.uint8(np.clip((img * rate + add), 0, 255))
    return img

def build_random_background(img, boxes):
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    pic = np.random.rand(h, w, 3) * 0.0
    pic = pic.astype(np.uint8)
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        pic[y0:y1, x0:x1, :] = img[y0:y1, x0:x1, :]
    return pic

def cut_random_and_rebuild(img, boxes, labels, img_size, cut_size = [2048, 2048], cut_box_rate = 0.95):
    rate = random.sample([0.9, 1, 1.1, 1.2], 1)[0]
    cut_size = [int(rate * cut_size[0]), int(rate * cut_size[1])]
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    boxes = boxes.astype(np.int32)
    def box_iou(box, w_left_cut, h_up_cut, w_right_cut, h_down_cut):
        left_line = max(box[0], w_left_cut) + 1
        right_line = min(box[2], w_right_cut) - 1
        top_line = max(box[1], h_up_cut) + 1
        bottom_line = min(box[3], h_down_cut) -1

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0, []
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / ((box[2] - box[0])*(box[3] - box[1])) * 1.0, [left_line - w_left_cut, top_line - h_up_cut, right_line - w_left_cut, bottom_line - h_up_cut]


    def cut_box(boxes, labels, w_left_cut, h_up_cut, w_right_cut, h_down_cut, find):
        boxes_cut = []
        labels_cut = []
        find = True
        for id in range(len(boxes)):
            box = boxes[id]
            iou, box_cut = box_iou(box, w_left_cut, h_up_cut, w_right_cut, h_down_cut)
            if 0< iou < cut_box_rate:
                find = False
            if len(box_cut) > 0:
                boxes_cut.append(box_cut)
                labels_cut.append(labels[id])
        return find, boxes_cut, labels_cut

    if len(boxes) > 0:
        h_up = np.min(boxes[:, 1])
        h_down = np.max(boxes[:, 3])
        w_left = np.min(boxes[:, 0])
        w_right = np.max(boxes[:, 2])
        h_cut = 0
        w_cut = 0
        find = False
        times = 0
        while(not find):
            if h_down - h_up <= cut_size[0] and w_right - w_left <= cut_size[1]:
                h_cut_scope_min = max(0, h_down - cut_size[0])
                h_cut_scope_max = min(h_up, h - cut_size[0])
                w_cut_scope_min = max(0, w_right - cut_size[1])
                w_cut_scope_max = min(w_left, w - cut_size[1])
                if h_cut_scope_min + 1 >= h_cut_scope_max:
                    h_cut = h_cut_scope_min
                else:
                    h_cut = random.randint(h_cut_scope_min, h_cut_scope_max)
                if w_cut_scope_min + 1 >= w_cut_scope_max:
                    w_cut = w_cut_scope_min
                else:
                    w_cut = random.randint(w_cut_scope_min, w_cut_scope_max)
            elif h_down - h_up > cut_size[0] and w_right - w_left <= cut_size[1]:
                w_cut_scope_min = max(0, w_right - cut_size[1])
                w_cut_scope_max = min(w_left, w - cut_size[1])
                if w_cut_scope_min + 1 >= w_cut_scope_max:
                    w_cut = w_cut_scope_min
                else:
                    w_cut = random.randint(w_cut_scope_min, w_cut_scope_max)
                h_cut = random.sample([h_down - cut_size[0], h_up], 1)[0]
            elif h_down - h_up <= cut_size[0] and w_right - w_left > cut_size[1]:
                h_cut_scope_min = max(0, h_down - cut_size[0])
                h_cut_scope_max = min(h_up, h - cut_size[0])
                if h_cut_scope_min + 1 >= h_cut_scope_max:
                    h_cut = h_cut_scope_min
                else:
                    h_cut = random.randint(h_cut_scope_min, h_cut_scope_max)
                w_cut = random.sample([w_right - cut_size[1], w_left], 1)[0]
            elif h_down - h_up > cut_size[0] and w_right - w_left > cut_size[1]:
                h_cut = random.sample([h_down - cut_size[0], h_up], 1)[0]
                w_cut = random.sample([w_right - cut_size[1], w_left], 1)[0]
            else:
                print("something wrong in api_cut_random_and_rebuild")
            boxes_temp = copy.deepcopy(boxes)
            boxes_temp[:, 0] = boxes[:, 0] - w_cut
            boxes_temp[:, 2] = boxes[:, 2] - w_cut
            boxes_temp[:, 1] = boxes[:, 1] - h_cut
            boxes_temp[:, 3] = boxes[:, 3] - h_cut
            find = True
            for i in range(len(boxes_temp)):
                box = boxes_temp[i]
                if ((box[2] > cut_size[1]) and (box[0] < cut_size[1])) or ((box[2] > 0) and (box[0] < 0)) or \
                    ((box[3] > cut_size[0]) and (box[1] < cut_size[0])) or ((box[3] > 0) and (box[1] < 0)):
                    find = False
                else:
                    pass
            times = times + 1
            if times > 2000:
                print("not find in father")
                h_up_cut = 0
                h_down_cut = 0
                w_left_cut = 0
                w_right_cut = 0
                boxes_temp = []
                labels_temp = []
                h_cut_bound = cut_size[0]
                w_cut_bound = cut_size[1]
                num_box = len(boxes)
                box_select_id = random.choice(range(num_box))
                # h_min = int(max(0, boxes[box_select_id][3] - h_cut_bound))
                # h_max = int(min(boxes[box_select_id][1] + h_cut_bound, h - h_cut_bound))
                # w_min = int(max(0, boxes[box_select_id][2] - w_cut_bound))
                # w_max = int(min(boxes[box_select_id][0] + w_cut_bound, w - w_cut_bound))
                h_min = int(max(0, boxes[box_select_id][3] - h_cut_bound))
                h_max = int(min(boxes[box_select_id][1], h - h_cut_bound))
                w_min = int(max(0, boxes[box_select_id][2] - w_cut_bound))
                w_max = int(min(boxes[box_select_id][0], w - w_cut_bound))
                find_child = False
                times = 0
                while (not find_child):
                    times = times + 1
                    if h_min >= h_max:
                        h_up_cut = h_min
                    else:
                        h_up_cut = random.randint(h_min, h_max)
                    h_down_cut = h_up_cut + h_cut_bound
                    if w_min >= w_max:
                        w_left_cut = w_min
                    else:
                        w_left_cut = random.randint(w_min, w_max)
                    w_right_cut = w_left_cut + w_cut_bound
                    find_child, boxes_temp, labels_temp = cut_box(boxes, labels, w_left_cut, h_up_cut, w_right_cut,
                                                                  h_down_cut, find_child)
                    if len(boxes_temp) <= 0:
                        find_child = False
                    if times > 500 and times > 0 and times % 500 == 0:
                        box_select_id = random.choice(range(num_box))
                        h_min = int(max(0, boxes[box_select_id][3] - h_cut_bound))
                        h_max = int(min(boxes[box_select_id][1] + h_cut_bound, h - h_cut_bound))
                        w_min = int(max(0, boxes[box_select_id][2] - w_cut_bound))
                        w_max = int(min(boxes[box_select_id][0] + w_cut_bound, w - w_cut_bound))
                    if times > 3000:
                        cut_box_rate = max(0.8 * cut_box_rate, 0.5)
                        # find_child = True

                img_temp = img[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
                # h_new, w_new = img.shape[0], img.shape[1]
                boxes_temp = np.array(boxes_temp)
                labels_temp = np.array(labels_temp)
                boxes_temp_select = []
                labels_temp_select = []
                for i in range(len(boxes_temp)):
                    box = boxes_temp[i]
                    box[0] = max(0, box[0])
                    box[1] = max(0, box[1])
                    box[2] = min(cut_size[1], box[2])
                    box[3] = min(cut_size[0], box[3])
                    if box[2] < 0 or box[0] > w_cut_bound or box[1] < 0 or box[3] > h_cut_bound or (box[2] - box[0]) <= 5 or (box[3] - box[1]) <= 5:
                        print("+++++++++++++++++++++++++")
                    else:
                        boxes_temp_select.append(box)
                        labels_temp_select.append(labels_temp[i])
                boxes_temp_select = np.array(boxes_temp_select)
                labels_temp_select = np.array(labels_temp_select)
                return img_temp, boxes_temp_select, labels_temp_select
            else:
                pass

        h_cut = int(h_cut)
        w_cut = int(w_cut)
        img = img[h_cut:h_cut + cut_size[0], w_cut:w_cut + cut_size[1], :]
        boxes[:, 0] = boxes[:, 0] - w_cut
        boxes[:, 2] = boxes[:, 2] - w_cut
        boxes[:, 1] = boxes[:, 1] - h_cut
        boxes[:, 3] = boxes[:, 3] - h_cut
        boxes_temp = []
        labels_temp = []
        for i in range(len(boxes)):
            box = boxes[i]
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(cut_size[1], box[2])
            box[3] = min(cut_size[0], box[3])
            if box[2] < 0 or box[0] > cut_size[1] or box[3] > cut_size[0] or box[1] < 0 or (box[2]-box[0])<=5 or (box[3]-box[1])<=5:
                pass
            else:
                boxes_temp.append(box)
                labels_temp.append(labels[i])
        boxes_temp = np.array(boxes_temp)
        labels_temp = np.array(labels_temp)
        return img, boxes_temp, labels_temp
    else:
        h_cut = random.randint(0, h - cut_size[0])
        w_cut = random.randint(0, w - cut_size[1])
        h_cut = int(h_cut)
        w_cut = int(w_cut)
        img = img[h_cut:h_cut + cut_size[0], w_cut:w_cut + cut_size[1], :]
        return img, boxes, labels

def cut_random_with_size(image, boxes, labels, class_num_dic, cut_size=(640, 640), cut_factor=0.5):
    def box_iou(box, w_left_cut, h_up_cut, w_right_cut, h_down_cut):
        left_line = max(box[0], w_left_cut)
        right_line = min(box[2], w_right_cut)
        top_line = max(box[1], h_up_cut)
        bottom_line = min(box[3], h_down_cut)
        if left_line >= right_line or top_line >= bottom_line:
            return 0, []
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / ((box[2] - box[0]) * (box[3] - box[1])) * 1.0, [left_line - w_left_cut, top_line - h_up_cut,
                                                                               right_line - w_left_cut, bottom_line - h_up_cut]

    def cut_boxes(boxes, w_left_cut, h_up_cut, w_right_cut, h_down_cut):
        boxes_cut = []
        keepout_boxes = []
        for id in range(len(boxes)):
            box = boxes[id]
            iou, box_cut = box_iou(box, w_left_cut, h_up_cut, w_right_cut, h_down_cut)
            if len(box_cut) > 0:
                if iou > cut_factor:
                    boxes_cut.append(box_cut)
                elif 0 < iou <= cut_factor:
                    keepout_boxes.append(np.array(box_cut, np.int32))
                else:
                    pass
        return boxes_cut, keepout_boxes

    H, W, C = image.shape
    h_cut_bound, w_cut_bound = cut_size
    # labels = labels.tolist()
    if len(boxes) == 0:
        h_up_cut = random.randint(0, H - h_cut_bound)
        h_down_cut = h_up_cut + h_cut_bound
        w_left_cut = random.randint(0, W - w_cut_bound)
        w_right_cut = w_left_cut + w_cut_bound
        image = image[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        return image, boxes

    elif len(boxes) == 1 and labels[0] == "gj":
        x_center = int(random.randint(int(min(max(boxes[0][0], w_cut_bound / 2 + 1), (W - w_cut_bound / 2) - 1)),
                                      int(max(min(boxes[0][2], (W - w_cut_bound / 2) + 1), w_cut_bound / 2 + 1))))
        y_center = int(random.randint(int(min(max(boxes[0][1], h_cut_bound / 2 + 1), (H - h_cut_bound / 2) - 1)),
                                      int(max(min(boxes[0][3], (H - h_cut_bound / 2) + 1), h_cut_bound / 2 + 1))))
        h_up_cut = int(y_center - h_cut_bound/2 + 1)
        h_down_cut = h_up_cut + h_cut_bound
        w_left_cut = int(x_center - w_cut_bound/2 + 1)
        w_right_cut = w_left_cut + w_cut_bound
        image = image[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        boxes = np.array([])
        labels = np.array([])
        return image, boxes
    elif len(boxes) > 1 and "gj" in labels:
        gj_box = boxes[labels.index("gj")]
        boxes_temp = []
        for id in range(len(labels)):
            if labels[id] != "gj":
                boxes_temp.append(boxes[id])
        boxes = np.array(boxes_temp)

        x_center = int(random.randint(int(min(max(gj_box[0], w_cut_bound / 2 + 1), (W - w_cut_bound / 2) - 1)),
                                      int(max(min(gj_box[2], (W - w_cut_bound / 2) + 1), w_cut_bound / 2 + 1))))
        y_center = int(random.randint(int(min(max(gj_box[1], h_cut_bound / 2 + 1), (H - h_cut_bound / 2) - 1)),
                                      int(max(min(gj_box[3], (H - h_cut_bound / 2) + 1), h_cut_bound / 2 + 1))))
        h_up_cut = int(y_center - h_cut_bound / 2 + 1)
        h_down_cut = h_up_cut + h_cut_bound
        w_left_cut = int(x_center - w_cut_bound / 2 + 1)
        w_right_cut = w_left_cut + w_cut_bound

        boxes_temp, keepout_boxes = cut_boxes(boxes, w_left_cut, h_up_cut, w_right_cut, h_down_cut)
        image = image[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        for keepout_box_id in range(len(keepout_boxes)):
            image[keepout_boxes[keepout_box_id][1]:keepout_boxes[keepout_box_id][3],
            keepout_boxes[keepout_box_id][0]:keepout_boxes[keepout_box_id][2], :] = 0
        boxes_temp = np.array(boxes_temp)
        return image, boxes_temp

    else:

        for key in list(class_num_dic.keys()):
            if key not in labels:
                class_num_dic.pop(key)

        class_num_dic = {key:value/sum(class_num_dic.values()) for key, value in class_num_dic.items()}
        class_rate_dic = {}

        sum_rate = 0
        keys = list(class_num_dic.keys())
        for key in keys:
            if keys.index(key) == len(keys) - 1:
                class_rate_dic[key] = [sum_rate, 1.]
            else:
                class_rate_dic[key] = [sum_rate, sum_rate + class_num_dic[key]]

            sum_rate = sum_rate + class_num_dic[key]

        random_rate = random.random()
        if len(keys) > 1:
            select_name = random.choice(keys) #防止后面异常选不到数据
        else:
            select_name = keys[0]
        for key in keys:
            if class_rate_dic[key][0] <= random_rate <= class_rate_dic[key][1]:
                select_name = key
                break
            else:
                pass

        img_label_ids = []
        id = 0
        for name in labels:
            if name == select_name:
                img_label_ids.append(id)
            id = id + 1

        box_select_id = random.choice(img_label_ids)


        h_min = int(max(0, boxes[box_select_id][3] - h_cut_bound))
        h_max = int(min(boxes[box_select_id][1], H - h_cut_bound))
        w_min = int(max(0, boxes[box_select_id][2] - w_cut_bound))
        w_max = int(min(boxes[box_select_id][0], W - w_cut_bound))

        if h_min >= h_max:
            h_up_cut = h_min
        else:
            h_up_cut = random.randint(h_min, h_max)
        h_down_cut = h_up_cut + h_cut_bound
        if w_min >= w_max:
            w_left_cut = w_min
        else:
            w_left_cut = random.randint(w_min, w_max)
        w_right_cut = w_left_cut + w_cut_bound
        boxes_temp, keepout_boxes = cut_boxes(boxes, w_left_cut, h_up_cut, w_right_cut, h_down_cut)
        image = image[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        for keepout_box_id in range(len(keepout_boxes)):
            image[keepout_boxes[keepout_box_id][1]:keepout_boxes[keepout_box_id][3], keepout_boxes[keepout_box_id][0]:keepout_boxes[keepout_box_id][2], :] = 0
        boxes_temp = np.array(boxes_temp)
        return image, boxes_temp


def cut_random_with_size_back(image, boxes, cut_size=(640, 640), cut_factor=0.5):
    def box_iou(box, w_left_cut, h_up_cut, w_right_cut, h_down_cut):
        left_line = max(box[0], w_left_cut)
        right_line = min(box[2], w_right_cut)
        top_line = max(box[1], h_up_cut)
        bottom_line = min(box[3], h_down_cut)
        if left_line >= right_line or top_line >= bottom_line:
            return 0, []
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / ((box[2] - box[0]) * (box[3] - box[1])) * 1.0, [left_line - w_left_cut, top_line - h_up_cut,
                                                                               right_line - w_left_cut, bottom_line - h_up_cut]

    def cut_boxes(boxes, w_left_cut, h_up_cut, w_right_cut, h_down_cut):
        boxes_cut = []
        keepout_boxes = []
        for id in range(len(boxes)):
            box = boxes[id]
            iou, box_cut = box_iou(box, w_left_cut, h_up_cut, w_right_cut, h_down_cut)
            if len(box_cut) > 0:
                if iou > cut_factor:
                    boxes_cut.append(box_cut)
                elif 0 < iou <= cut_factor:
                    keepout_boxes.append(np.array(box_cut, np.int32))
                else:
                    pass
        return boxes_cut, keepout_boxes

    H, W, C = image.shape
    h_cut_bound, w_cut_bound = cut_size
    if len(boxes) == 0:
        h_up_cut = random.randint(0, H - h_cut_bound)
        h_down_cut = h_up_cut + h_cut_bound
        w_left_cut = random.randint(0, W - w_cut_bound)
        w_right_cut = w_left_cut + w_cut_bound
        image = image[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        return image, boxes

    else:
        num_box = len(boxes)
        box_select_id = random.choice(range(num_box))
        h_min = int(max(0, boxes[box_select_id][3] - h_cut_bound))
        h_max = int(min(boxes[box_select_id][1], H - h_cut_bound))
        w_min = int(max(0, boxes[box_select_id][2] - w_cut_bound))
        w_max = int(min(boxes[box_select_id][0], W - w_cut_bound))

        if h_min >= h_max:
            h_up_cut = h_min
        else:
            h_up_cut = random.randint(h_min, h_max)
        h_down_cut = h_up_cut + h_cut_bound
        if w_min >= w_max:
            w_left_cut = w_min
        else:
            w_left_cut = random.randint(w_min, w_max)
        w_right_cut = w_left_cut + w_cut_bound
        boxes_temp, keepout_boxes = cut_boxes(boxes, w_left_cut, h_up_cut, w_right_cut, h_down_cut)
        image = image[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        for keepout_box_id in range(len(keepout_boxes)):
            image[keepout_boxes[keepout_box_id][1]:keepout_boxes[keepout_box_id][3], keepout_boxes[keepout_box_id][0]:keepout_boxes[keepout_box_id][2], :] = 0
        boxes_temp = np.array(boxes_temp)
        return image, boxes_temp

def cut_random_new(img_path, img, boxes, labels, img_size, cut_size_boundary=1, cut_rate = 0.2):
    h, w = img.shape[0], img.shape[1]
    h_up = np.min(boxes[:,1])
    h_down = np.max(boxes[:,3])
    w_left = np.min(boxes[:,0])
    w_right = np.max(boxes[:,2])

    boxes[:, 0] = boxes[:, 0] + 1
    boxes[:, 2] = boxes[:, 2] - 1
    boxes[:, 1] = boxes[:, 1] + 1
    boxes[:, 3] = boxes[:, 3] - 1

    h_up_cut = random.choice(range(min(max(int(h_up - cut_size_boundary), 1), int(h * cut_rate))))
    h_down_cut = random.choice(range(max(min(int(h_down + cut_size_boundary), h-1), int(h * (1-cut_rate))), h))
    w_left_cut = random.choice(range(min(max(int(w_left - cut_size_boundary), 1), int(w * cut_rate))))
    w_right_cut = random.choice(range(max(min(int(w_right + cut_size_boundary), w-1), int(w * (1-cut_rate))), w))

    img = img[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
    h_new, w_new = img.shape[0], img.shape[1]
    boxes[:, 0] = boxes[:, 0] - w_left_cut
    boxes[:, 2] = boxes[:, 2] - w_left_cut
    boxes[:, 1] = boxes[:, 1] - h_up_cut
    boxes[:, 3] = boxes[:, 3] - h_up_cut
    return img, boxes, labels

def add_background_box(img, boxes, background_data, add_num):
    def calc_iou(pred_boxes, true_boxes):
        '''
        Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
        shape_info: pred_boxes: [N, 4]
                    true_boxes: [V, 4]
        '''
        # [N, 1, 4]
        pred_boxes = np.expand_dims(pred_boxes, -2)
        # [1, V, 4]
        true_boxes = np.expand_dims(true_boxes, 0)
        # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
        intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
        intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        # shape: [N, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 1, 2]
        pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
        # shape: [N, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # [1, V, 2]
        true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
        # [1, V]
        true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
        # shape: [N, V]
        iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)
        return iou
    add_num = random.choice(range(4, add_num))
    boxes = boxes.tolist()
    img_h, img_w, _ = img.shape
    background_data_id = [id for id in range(len(background_data))]
    select_data = random.sample(background_data_id, add_num)
    for id in range(add_num):
        try:
            add_data = cv2.imread(background_data[select_data[id]])
            add_h, add_w, _ = add_data.shape
            find = False
            id = 0
            while not find:
                x0 = random.choice(range(0, img_w - add_w))
                y0 = random.choice(range(0, img_h - add_h))
                box_add = [[x0, y0, x0+add_w, y0+add_h]]
                iou = calc_iou(box_add, boxes)[0]
                if max(iou) <= 0.:
                    find = True
                    img[y0:y0+add_h, x0:x0+add_w] = add_data
                    boxes.append([x0, y0, x0+add_w, y0+add_h])
                else:
                    pass
                id = id + 1
                if id > 500:
                    find = True
        except Exception as ex:
            globalvar.globalvar.logger.info_ai(meg="add background fail:%s"%ex)

    return img


def replace_random(img, boxes, labels, max_num_replace=1, replace_box_scope=(30, 500)):
    img_ori = copy.deepcopy(img)
    ori_height, ori_width = img.shape[:2]
    num_box = len(boxes)
    num_box_ids = [id for id in range(num_box)]
    select_data = random.sample(num_box_ids, min(max_num_replace, num_box))
    for id in select_data:
        x0, y0, x1, y1 = boxes[id].astype(np.int32)
        replace_box_w = random.choice(range(replace_box_scope[0], replace_box_scope[1]))
        replace_box_h = random.choice(range(replace_box_scope[0], replace_box_scope[1]))
        replace_box_center_x = random.choice(range(x0, x1))
        replace_box_center_y = random.choice(range(y0, y1))
        replace_box_center_x0 = int(max(0, replace_box_center_x - replace_box_w / 2))
        replace_box_center_x1 = int(min(replace_box_center_x + replace_box_w / 2, ori_width))
        replace_box_center_y0 = int(max(0, replace_box_center_y - replace_box_h / 2))
        replace_box_center_y1 = int(min(replace_box_center_y + replace_box_h / 2, ori_height))

        replace_box_w = replace_box_center_x1 - replace_box_center_x0
        replace_box_h = replace_box_center_y1 - replace_box_center_y0

        box_temp = np.array([[0, 0, 0, 0]])
        find = False
        while (not find):
            x_temp = random.choice(range(0, ori_width - replace_box_w))
            y_temp = random.choice(range(0, ori_height - replace_box_h))
            box_temp = np.array([[x_temp, y_temp, x_temp + replace_box_w, y_temp + replace_box_h]])
            iou_temp = calc_iou(box_temp, boxes)
            if np.sum(iou_temp) == 0:
                find = True

        img[replace_box_center_y0:replace_box_center_y1, replace_box_center_x0:replace_box_center_x1, :] = img[box_temp[0][1]:box_temp[0][3], box_temp[0][0]:box_temp[0][2], :]
        img[y0:y1, x0:x1, :] = img_ori[y0:y1, x0:x1, :]

    return img, boxes, labels


def random_color_temperature(img_rgb_in):
    def create_LUT_8UC1(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    ori_ch_value = [0, 64, 128, 192, 256]
    ratio = random.randint(5., 30.)/100.0
    half_ratio = ratio * 0.5
    incr_ch_value = [0, int(64 * (1 + half_ratio)), int(128 * (1 + ratio)), int(192 * (1 + half_ratio)), 256]
    decr_ch_value = [0, int(64 * (1 - half_ratio)), int(128 * (1 - ratio)), int(192 * (1 - half_ratio)), 192]

    incr_ch_lut = create_LUT_8UC1(ori_ch_value, incr_ch_value)
    decr_ch_lut = create_LUT_8UC1(ori_ch_value, decr_ch_value)
    img_bgr_in = img_rgb_in[..., ::-1]
    c_b, c_g, c_r = cv2.split(img_bgr_in)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.merge((c_b, c_g, c_r))
    img_rgb_warm = img_bgr_warm[..., ::-1]
    if random.sample([0, 1], 1)[0] == 0:
        return img_rgb_warm

    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb_warm,
                                           cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
    img_rgb_warm = img_bgr_warm[..., ::-1]

    return img_rgb_warm


def change_temperature(img_rgb, temperature_target):
    change_ratio = np.array(temperature_target, dtype=np.float32)/255.
    img_rgb = img_rgb * change_ratio
    img_rgb = np.asarray(img_rgb, dtype=np.uint8)

    return img_rgb

def random_rotation(img_path, img, boxes, labels, img_size, angle_value=10, angle_set=None):
    h, w = img.shape[0], img.shape[1]
    num_boxes = boxes.shape[0]
    def rotate_point(x, y, angle):
        x = x - int(w / 2)
        y = -1 * (y - int(h / 2))
        x_temp = copy.deepcopy(x)
        y_temp = copy.deepcopy(y)
        x = x_temp * math.cos(angle / 180. * math.pi) - y_temp * math.sin(angle / 180. * math.pi)
        y = x_temp * math.sin(angle / 180. * math.pi) + y_temp * math.cos(angle / 180. * math.pi)
        x = x + int(w / 2)
        y = -1 * y + int(h / 2)
        return x, y
    img_temp = copy.deepcopy(img)
    boxes_temp = copy.deepcopy(boxes)
    find = False
    try_id = 0
    while (not find):
        if angle_set is not None:
            angle = random.sample([-90, 90, 180], 1)[0]
        else:
            angle = random.randint(-1*angle_value, angle_value)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_change = cv2.warpAffine(img_temp, M, (w, h),  borderValue=(0, 0, 0))#borderMode=cv2.INTER_LINEAR, borderValue=cv2.BORDER_REPLICATE
        # boxes_change = copy.deepcopy(boxes)
        boxes_change = []
        labels_change = []
        find = True
        if num_boxes > 0:
            for i in range(num_boxes):
                x0, y0, x3, y3 = boxes_temp[i]
                x1, y1, x2, y2 = x3, y0, x0, y3
                x0, y0 = rotate_point(x0, y0, angle)
                x1, y1 = rotate_point(x1, y1, angle)
                x2, y2 = rotate_point(x2, y2, angle)
                x3, y3 = rotate_point(x3, y3, angle)
                x_min = min(x0, x1, x2, x3)
                x_max = max(x0, x1, x2, x3)
                y_min = min(y0, y1, y2, y3)
                y_max = max(y0, y1, y2, y3)
                # if x_min <= 0 or y_min <= 0 or x_max >= w or y_max >= h:
                #     find = False
                #     break
                if x_min <= 0 or y_min <= 0 or x_max >= w or y_max >= h:
                    img_change[int(min(max(0, y_min), w-1)):int(min(max(0, y_max), h-1)), int(min(max(0, x_min), w-1)):int(min(max(0, x_max), h-1)), :] = 0
                else:
                    boxes_change.append([x_min, y_min, x_max, y_max])
                    labels_change.append(labels[i])

                # boxes_change[i][0] = x_min
                # boxes_change[i][1] = y_min
                # boxes_change[i][2] = x_max
                # boxes_change[i][3] = y_max

            if find:
                img = img_change
                boxes = np.array(boxes_change)
                labels = np.array(labels_change)
        else:
            pass
        # try_id = try_id + 1
        # if try_id > 100:
        #     break

    return img, boxes, labels


def padding_random(img_path, img, boxes, labels, img_size, padding_rate=0.2):
    h, w, c = img.shape
    w_padding_size = int(random.uniform(0, int(w * padding_rate)))
    h_padding_size = int(random.uniform(0, int(h * padding_rate)))

    img_padding = np.zeros((h + h_padding_size, w + w_padding_size, 3), dtype=np.uint8)
    img_padding[:h, :w, :] = img

    return img, boxes, labels


def data_augmentation_with_gray(img_path, img, boxes, labels, img_size, class_num_dic, use_label_smothing = False):
    confs = None
    choice_list = [0, 1]
    h, w, c = img.shape
    choice = 0

    if choice == 0:
        cutsize = random.randint(1600, 2400)
        img, boxes = cut_random_with_size(img, boxes, labels, class_num_dic, cut_size=(cutsize, cutsize), cut_factor=0.8)
        labels = [0 for i in range(len(boxes))]

        choice = random.sample([0, 1, 2, 3], 1)[0]
        # choice = 0
        if choice != 0 and len(boxes) > 0 and globalvar.globalvar.config.model_set["add_background"] and globalvar.globalvar.background_add_lines is not None:
            img = add_background_box(img, boxes, globalvar.globalvar.background_add_lines, 6)

        choice = random.sample(choice_list, 1)[0]
        choice = 1
        if choice == 0 and len(boxes) > 2:
            img, boxes, labels = replace_random(img, boxes, labels, max_num_replace=2, replace_box_scope=(100, 500))

        img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
        choice = random.sample(choice_list, 1)[0]
        # choice = 0
        if choice == 0:
            img, boxes, labels = random_rotation(img_path, img, boxes, labels, img_size, angle_value=180, angle_set=None)
        choice = random.sample(choice_list, 1)[0]
        # choice = 0
        if choice == 0:
            img, boxes, labels = flip_rl(img, boxes, labels)
        choice = random.sample(choice_list, 1)[0]
        # choice = 0
        if choice == 0:
            img, boxes, labels = flip_ud(img, boxes, labels)

        transform = A.Compose(
            [
             A.augmentations.transforms.RandomGamma(gamma_limit=(50, 150), eps=None, always_apply=False, p=0.5),
             A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=[-0.3, 0.2], contrast_limit=0.1, brightness_by_max=False, always_apply=False, p=0.5),
            ])
        data_transformed = transform(image=img)
        img = data_transformed["image"]

    boxes = boxes.astype(np.float32)

    return img, boxes, labels, confs

def data_augmentation(img_path, img, boxes, labels, img_size, class_num_dic, use_label_smothing = False):
    confs = None
    choice_list = [0, 1]
    h, w, c = img.shape
    choice = 0

    if choice == 0:
        cutsize = random.randint(1600, 2400)
        img, boxes = cut_random_with_size(img, boxes, labels, class_num_dic, cut_size=(cutsize, cutsize), cut_factor=0.8)
        labels = [0 for i in range(len(boxes))]

        choice = random.sample([0, 1, 2, 3], 1)[0]
        # choice = 0
        if choice != 0 and len(boxes) > 0 and globalvar.globalvar.config.model_set["add_background"] and globalvar.globalvar.background_add_lines is not None:
            img = add_background_box(img, boxes, globalvar.globalvar.background_add_lines, 6)

        choice = random.sample(choice_list, 1)[0]
        choice = 1
        if choice == 0 and len(boxes) > 2:
            img, boxes, labels = replace_random(img, boxes, labels, max_num_replace=2, replace_box_scope=(100, 500))

        img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
        choice = random.sample(choice_list, 1)[0]
        # choice = 0
        if choice == 0:
            img, boxes, labels = random_rotation(img_path, img, boxes, labels, img_size, angle_value=180, angle_set=None)
        choice = random.sample(choice_list, 1)[0]
        # choice = 0
        if choice == 0:
            img, boxes, labels = flip_rl(img, boxes, labels)
        choice = random.sample(choice_list, 1)[0]
        # choice = 0
        if choice == 0:
            img, boxes, labels = flip_ud(img, boxes, labels)

        transform = A.Compose(
            [
             A.augmentations.transforms.RandomGamma(gamma_limit=(50, 150), eps=None, always_apply=False, p=0.5),
             A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=[-0.3, 0.2], contrast_limit=0.1, brightness_by_max=False, always_apply=False, p=0.5),
            ])
        data_transformed = transform(image=img)
        img = data_transformed["image"]

    boxes = boxes.astype(np.float32)

    return img, boxes, labels, confs
#
# def data_augmentation(img_path, img, boxes, labels, img_size, class_num_dic, use_label_smothing = False):
#     #confs = np.ones(labels.shape)
#     confs = None
#     img_temp = [img.copy()]
#     boxes_temp = [boxes.copy()]
#     labels_temp = [labels.copy()]
#     choice_list = [0, 1]
#     # choice = random.sample(choice_list, 1)[0]
#     h, w, c = img.shape
#     img_ori_size = [w, h]
#
#     choice = 0
#     judge_box("original", img_path, img, boxes)
#     if choice == 0:
#         choice = random.sample(choice_list, 1)[0]
#         #choice = 1
#         if choice == 0:
#             img, boxes, labels = random_rotation(img_path, img, boxes, labels, img_size, angle_value=180, angle_set=None)
#
#         choice = random.sample(choice_list, 1)[0]
#         choice = 1
#         if choice == 0 and len(boxes) > 0:
#             img, boxes, labels = cut_random_new(img_path, img, boxes, labels, img_size)
#             img, boxes = resize_image_and_correct_boxes(img, boxes, img_ori_size)
#
#         choice = random.sample(choice_list, 1)[0]
#         choice = 1
#         if choice == 0 and len(boxes) > 0:
#             img, boxes, labels = padding_random(img_path, img, boxes, labels, img_size)
#             img, boxes = resize_image_and_correct_boxes(img, boxes, img_ori_size)
#
#         cutsize = random.randint(1600, 2400)
#         img, boxes = cut_random_with_size(img, boxes, labels, class_num_dic, cut_size=(cutsize, cutsize),
#                                           cut_factor=0.8)
#         labels = [0 for i in range(len(boxes))]
#
#         choice = random.sample(choice_list, 1)[0]
#         choice = 1
#         if choice == 0 and len(boxes) > 2:
#             img, boxes, labels = replace_random(img, boxes, labels, max_num_replace=2, replace_box_scope=(100, 500))
#
#         choice = random.sample([0,1,2,3], 1)[0]
#         choice = 0
#         if choice != 0 and len(boxes) > 0:
#             img = add_background_box(img, boxes, background_lines, 10)
#         choice = random.sample([0, 1, 2, 3], 1)[0]
#         choice = 0
#         if choice != 0 and len(boxes) > 0:
#             img, boxes, labels = add_object_box_old(img, boxes, labels, object_lines, 5)
#         if len(boxes) != len(labels):
#             print("len boxes not equal len labels")
#         img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
#
#         choice = random.sample([0,1,2,3], 1)[0]
#         if choice == 0:
#             boxes = boxes.astype(np.float32)
#             return img, boxes, labels, confs
#
#
#         choice = random.sample(choice_list, 1)[0]
#         choice = 1
#         if choice == 0:
#             img = random_color_temperature(img)
#
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 0
#         if choice == 0:
#             img, boxes, labels = flip_rl(img, boxes, labels)
#             img_temp.append(img.copy())
#             boxes_temp.append(boxes.copy())
#             labels_temp.append(labels.copy())
#             judge_box("flip_rl", img_path, img, boxes)
#         # choice = random.sample(choice_list, 1)[0]
#         # if choice == 0:
#         #    img, boxes, labels = occlusion_random(img, boxes, labels)
#         #    judge_box("occlusion_random", img_path, img, boxes)
#         # choice = random.sample(choice_list, 1)[0]
#         # if choice == 0:
#         #     img = color_random(img)
#         # choice = random.sample(choice_list, 1)[0]
#         # if choice == 0:
#         #     img = img_addcontrast_brightness(img)
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 0
#         if choice == 0:
#             img, boxes, labels = flip_ud(img, boxes, labels)
#             img_temp.append(img.copy())
#             boxes_temp.append(boxes.copy())
#             labels_temp.append(labels.copy())
#             judge_box("flip_ud", img_path, img, boxes)
#         choice = random.sample(choice_list, 1)[0]
#         choice = 1
#         if choice == 0:
#             img, boxes, labels = rotation_random(img, boxes, labels)
#             img_temp.append(img.copy())
#             boxes_temp.append(boxes.copy())
#             labels_temp.append(labels.copy())
#             judge_box("rotation_random", img_path, img, boxes)
#         if use_label_smothing:
#             if np.sum((boxes_temp[0] == boxes).astype(int)) == boxes.shape[0] * boxes.shape[1]:
#                 pass
#             else:
#                 choice = random.sample(choice_list, 1)[0]
#                 if choice == 0 and len(labels_temp) > 1:
#                     id_0, id_1 = random.sample(range(len(labels_temp)), 2)
#                     rate = random.choice(range(18,22)) / 10
#                     img = 1/rate * img_temp[id_0] + (1-1/rate)*img_temp[id_1]
#                     img = np.asarray(img, np.uint8)
#                     boxes = np.append(boxes_temp[id_0], boxes_temp[id_1], axis=0)
#                     labels = np.append(labels_temp[id_0], labels_temp[id_1], axis=0)
#                     confs = np.append(1/rate * confs, (1-1/rate) * confs, axis=0)
#                 judge_box("label_smothing", img_path, img, boxes)
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 1
#         if choice == 0:
#             img = img_addcontrast_brightness(img)
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 1
#         if choice == 0:
#             img = img_blur(img)
#         choice = random.sample(choice_list, 1)[0]
#         choice = 1
#         if choice == 0:
#             img = img_addweight(img)
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 1
#         if choice == 0:
#             img = random_change_contrast(img)
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 1
#         if choice == 0:
#             img = random_color_distort(img)
#
#
#         choice = random.sample(choice_list, 1)[0]
#         # choice = 1
#         if choice == 0:
#             ratio = random.sample([[222, 246, 255], [220, 255, 255], [227, 238, 255], [218, 237, 255]], 1)[0]
#             img = change_temperature(img, ratio)
#
#         # boxes, labels, confs = merge_box(boxes, labels, confs, threshold = 0.2)
#     # img = img.astype(np.float32)
#     boxes = boxes.astype(np.float32)
#     #confs = confs.astype(np.float32)
#
#     return img, boxes, labels, confs


