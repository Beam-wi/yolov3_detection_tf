import os
import random
import cv2
import math
import time
import tensorflow as tf
import numpy as np
import utils.globalvar as globalvar
from utils.model import darknet_plus
import albumentations as A
from shutil import copyfile
from utils.model import darknet_plus as yolov3_classfy
from utils.data_utils import readImg
from utils.aug import imRotate


# 随机打乱训练数据集顺序
def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def rename_path(img_paths):
    for img_path in img_paths:
        img_path_new = img_path[:-4]
        img_path_new = img_path_new.replace(".png", "_")
        img_path_new = img_path_new.replace(".jpg", "_")
        img_path_new = img_path_new + ".jpg"
        print(img_path, img_path_new)
        os.rename(img_path, img_path_new)


# 从文件夹中获取训练数据，并写入文件中
def builddata_shuffle_and_overwrite(data_dir, file_path, class_name, name2id,  mode, max_num=1000.):
    data_dir = data_dir.strip()
    child_paths = os.listdir(data_dir)
    lines = []
    lines_temps = [[] for name in name2id.keys()]
    for child_path in child_paths:
        child_path = os.path.join(data_dir, child_path)
        for name in os.listdir(child_path):
            if name in list(name2id.keys()):
                idx = name2id[name]
                class_path = os.path.join(data_dir, child_path, name)
                if name not in ["background"]:  # 这里写死，所有背景必须放在这个名字文件夹下面
                    for img_name in os.listdir(class_path):
                        if img_name.split(".")[-1] == "jpg":
                            path = os.path.join(data_dir, child_path, name, img_name)
                            if os.path.getsize(path) >= 800:
                                line = path + " " + str(idx)
                                lines_temps[idx].append(line)
                else:
                    for background_name in os.listdir(class_path):
                        child_background_path = os.path.join(data_dir, child_path, name, background_name)
                        for img_name in os.listdir(child_background_path):
                            if img_name.split(".")[-1] == "jpg":
                                path = os.path.join(data_dir, child_path, name, background_name, img_name)
                                if os.path.getsize(path) >= 800:
                                    line = path + " " + str(idx)
                                    lines_temps[idx].append(line)
            else:
                globalvar.globalvar.logger.warning(f"{name} not in class names")

    if mode == "train":
        for lines_temp_id in range(len(lines_temps)):
            lines_temp = lines_temps[lines_temp_id]
            if len(lines_temp) == 0:
                continue
            random.shuffle(lines_temp)
            random.shuffle(lines_temp)
            random.shuffle(lines_temp)
            random.shuffle(lines_temp)
            repetion_times = max(math.floor(max_num / len(lines_temp)), 1)
            add_num = int(max(max_num - repetion_times * len(lines_temp), 0))
            globalvar.globalvar.logger.info(f"{class_name[lines_temp_id]}, max_num:{max_num}, lines_num:{len(lines_temp)}, repetion_times:{repetion_times}, add num:{add_num}")

            for time in range(repetion_times):
                if len(lines_temp) > max_num:
                    lines = lines + lines_temp[:int(max_num)]
                else:
                    lines = lines + lines_temp
            lines = lines + lines_temp[:add_num]

        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
    else:
        for lines_temp_id in range(len(lines_temps)):
            lines_temp = lines_temps[lines_temp_id]
            if len(lines_temp) == 0:
                continue
            lines = lines + lines_temp

    with open(file_path, 'w', encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    globalvar.globalvar.logger.info(f"{mode} data build over, data num is:{len(lines)}")


# 扩充函数
def data_augmentation(img, img_size, label_id, pic_path):
    def flip_rl(img):
        img = img[:, ::-1, :]
        return img

    def flip_ud(img):
        img = img[::-1, :, :]
        return img

    def img_addcontrast_brightness(imgrgb):
        a = random.sample([i / 10 for i in range(3, 18)], 1)[0]
        g = random.sample([i for i in range(0, 20)], 1)[0]
        h, w, ch = imgrgb.shape
        src2 = np.zeros([h, w, ch], imgrgb.dtype)
        img_bright = cv2.addWeighted(imgrgb, a, src2, 1 - a, g)
        return img_bright

    def rotation_random(img):
        def img_rotation(imgrgb, angle):
            rows, cols, channel = imgrgb.shape
            rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_rotation = cv2.warpAffine(imgrgb, rotation, (cols, rows), borderValue=128)
            return img_rotation

        angle = random.choice([90, 180])
        if angle == 90:
            img = img_rotation(img, -90)
        else:
            img = img_rotation(img, -180)
        return img

    def img_blur(imgrgb):
        choice_list = [3, 5]
        my_choice = random.sample(choice_list, 1)
        img_blur = cv2.blur(imgrgb, (my_choice[0], my_choice[0]))
        return img_blur

    def img_addweight(imgrgb):
        choice_list = [i * 10 for i in range(5, 15)]
        my_choice = random.sample(choice_list, 1)
        blur = cv2.GaussianBlur(imgrgb, (0, 0), my_choice[0])
        img_addweight = cv2.addWeighted(imgrgb, 1.2, blur, -0.2, 0)
        return img_addweight

    def img_cut_random(imgrgb, cut_rate=0.1):
        h, w, c = imgrgb.shape
        cut_h_u = random.choice(range(1, int(h * cut_rate)))
        cut_h_d = random.choice(range(1, int(h * cut_rate)))
        cut_w_l = random.choice(range(1, int(w * cut_rate)))
        cut_w_r = random.choice(range(1, int(w * cut_rate)))
        # img_cut = np.zeros((h, w, c), dtype=np.uint8)
        img_cut = np.random.rand(h, w, 3) * 255
        img_cut = img_cut.astype(np.uint8)
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        img_cut[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :] = imgrgb[
                                                                                               cut_h_u:h - cut_h_d,
                                                                                               cut_w_l:w - cut_w_r, :]
        return img_cut

    def radom_rotation(imgrgb):
        h, w, c = imgrgb.shape
        angle = random.randint(-45, 45)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(imgrgb, M, (w, h), borderValue=(0, 0, 0))
        return img

    def extend_and_cut_random(imgrgb, extend_rate=1.3, cut_rate=0.2):
        h, w, c = imgrgb.shape
        extend_rate = random.sample([1.1, 1.2, 1.3], 1)[0]
        h_extend = int(h * extend_rate)
        w_extend = int(w * extend_rate)
        pic = np.random.rand(h_extend, w_extend, 3) * 0
        pic = pic.astype(np.uint8)
        pic[int((h_extend - h) / 2):int((h_extend - h) / 2) + h, int((w_extend - w) / 2):int((w_extend - w) / 2) + w,
        :] = img
        cut_h_u = random.choice(range(1, int(h_extend * cut_rate)))
        cut_h_d = random.choice(range(1, int(h_extend * cut_rate)))
        cut_w_l = random.choice(range(1, int(w_extend * cut_rate)))
        cut_w_r = random.choice(range(1, int(w_extend * cut_rate)))
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        pic = pic[h_cut:h_extend - cut_h_d - cut_h_u + h_cut, w_cut:w_extend - cut_w_r - cut_w_l + w_cut, :]

        return pic

    def sp_noise(image, prob):
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    # def cut_random(imgrgb, cut_rate=0.1):
    #     h, w, c = imgrgb.shape
    #     cut_h_u = random.choice(range(1, max(int(h * cut_rate), 2)))
    #     cut_h_d = random.choice(range(1, max(int(h * cut_rate), 2)))
    #     cut_w_l = random.choice(range(1, max(int(w * cut_rate), 2)))
    #     cut_w_r = random.choice(range(1, max(int(w * cut_rate), 2)))
    #     h_cut = random.choice(range(0, cut_h_d + cut_h_u))
    #     w_cut = random.choice(range(0, cut_w_l + cut_w_r))
    #     imgrgb = imgrgb[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :]
    #
    #     return imgrgb

    def cut_random(image, ratio=0.21):
        ratio_ = round(random.uniform(*ratio), 2) if isinstance(ratio, (list, tuple)) else ratio
        ori_w, ori_h = image.shape[1], image.shape[0]
        h_min, h_max = min(int(ori_h*(1-ratio_)), int(ratio_*ori_h)), max(int(ori_h*(1-ratio_)), int(ratio_*ori_h))
        w_min, w_max = min(int(ori_w*(1-ratio_)), int(ratio_*ori_w)), max(int(ori_w*(1-ratio_)), int(ratio_*ori_w))
        # y0, y1 = random.randint(0, h_min-1), random.randint(h_max, ori_h-1)
        # x0, x1 = random.randint(0, w_min-1), random.randint(w_max, ori_w-1)
        y0, y1 = random.randint(0, h_min), random.randint(h_max, ori_h-1)
        x0, x1 = random.randint(0, w_min), random.randint(w_max, ori_w-1)
        new_img = image[y0: y1, x0:x1, :]
        return new_img

    def cut_random_27(imgrgb, pic_path, cut_rate=0.25):
        # print("cut label 27")
        # print(pic_path)
        h, w, c = imgrgb.shape
        min_size = random.sample([12, 16, 20], 1)[0]
        cut_h_u = random.choice(range(min_size, max(int(h * cut_rate), min_size + 2)))
        cut_h_d = random.choice(range(min_size, max(int(h * cut_rate), min_size + 2)))
        cut_w_l = random.choice(range(min_size, max(int(w * cut_rate), min_size + 2)))
        cut_w_r = random.choice(range(min_size, max(int(w * cut_rate), min_size + 2)))
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        imgrgb = imgrgb[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :]

        return imgrgb

    # img = cv2.resize(img, tuple(img_size))
    def cut_random_ll(img, cut_size_boundary=1, cut_rate=0.2):
        h, w = img.shape[0], img.shape[1]
        h_up = h * 0.2
        h_down = h * 0.8
        w_left = w * 0.2
        w_right = w * 0.8
        h_up_cut = random.choice(range(min(max(int(h_up - cut_size_boundary), 1), int(h * cut_rate))))
        h_down_cut = random.choice(range(max(min(int(h_down + cut_size_boundary), h - 1), int(h * (1 - cut_rate))), h))
        w_left_cut = random.choice(range(min(max(int(w_left - cut_size_boundary), 1), int(w * cut_rate))))
        w_right_cut = random.choice(
            range(max(min(int(w_right + cut_size_boundary), w - 1), int(w * (1 - cut_rate))), w))
        img = img[h_up_cut:h_down_cut, w_left_cut:w_right_cut, :]
        return img

    def random_color_distort(img, brightness_delta=13, hue_vari=3, sat_vari=0.5, val_vari=0.5):
        # def random_color_distort(img, brightness_delta=50, hue_vari=30, sat_vari=2, val_vari=0.2):
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

    def fill(image, value=128):
        ori_w, ori_h = image.shape[1], image.shape[0]
        if ori_w >= ori_h:
            tmp = int((ori_w - ori_h) / 2)
            image = cv2.copyMakeBorder(image, 0, tmp, 0, 0, cv2.BORDER_CONSTANT, value=[value, value, value])
            image = cv2.copyMakeBorder(image, tmp, 0, 0, 0, cv2.BORDER_CONSTANT, value=[value, value, value])

        else:
            tmp = int((ori_h - ori_w) / 2)
            image = cv2.copyMakeBorder(image, 0, 0, 0, tmp, cv2.BORDER_CONSTANT, value=[value, value, value])
            image = cv2.copyMakeBorder(image, 0, 0, tmp, 0, cv2.BORDER_CONSTANT, value=[value, value, value])

        return image

    def fill_192(image):
        ori_w, ori_h = image.shape[1], image.shape[0]
        size = 250
        resize = 96
        image_return = np.zeros((size * 2, size * 2, 3), dtype=int)
        x0 = size - int(ori_w / 2)
        x1 = size + ori_w - int(ori_w / 2)
        y0 = size - int(ori_h / 2)
        y1 = size + ori_h - int(ori_h / 2)
        x0_cut = size - resize
        x1_cut = size + resize
        y0_cut = size - resize
        y1_cut = size + resize
        image_return[y0:y1, x0:x1, :] = image
        image_return = image_return[y0_cut:y1_cut, x0_cut:x1_cut, :]
        image_return = np.asarray(image_return, np.uint8)
        return image_return

    def reduce_light(img):
        rate = random.sample([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 1)[0]
        img = img * rate
        img = np.floor(img)
        img = np.array(img, np.uint8)
        return img

    def random_stripe(img):
        H, W, C = img.shape
        stripe_w_list = [5, 10, 15, 20]
        stripe_h_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        stripe_w = random.sample(stripe_w_list, 1)[0]
        stripe_h = random.sample(stripe_h_list, 1)[0]
        if random.sample([0, 1], 1)[0] == 0:
            if stripe_h * 3 < H and stripe_w * 3 < W:
                stripe_h = int(H * stripe_h)
                add_value = random.randint(0, 50)
                start_y = random.randint(0, H - stripe_h)
                # start_x = random.randint(int(W / 2 - stripe_w / 2), int(W / 2 + stripe_w / 2))
                start_x = random.randint(0, W - stripe_w)
                img[start_y:start_y + stripe_h, start_x:start_x + stripe_w, :] = add_value
            else:
                pass
        else:
            pass
        stripe_h = random.sample(stripe_h_list, 1)[0]
        if random.sample([0, 1], 1)[0] == 0:
            if stripe_h * 3 < W and stripe_w * 3 < H:
                stripe_h = int(W * stripe_h)
                add_value = random.randint(0, 50)
                # start_y = random.randint(int(H/2 - stripe_w/2), int(H/2 + stripe_w/2))
                start_y = random.randint(0, H - stripe_w)
                start_x = random.randint(0, W - stripe_h)
                img[start_y:start_y + stripe_w, start_x:start_x + stripe_h, :] = add_value
            else:
                pass
        else:
            pass

        return img

    choice_list = [0, 1]

    if globalvar.globalvar.config.model_set["cutRatio"]:
        img = cut_random(img, ratio=globalvar.globalvar.config.model_set["cutRatio"])

    # # 随机旋转
    # if random.randint(0, 1):
    #     # angle = random.randint(-90, 90)
    #     angle = random.randint(-5, 5)
    #     img = imRotate(img, angle, border_value=128, adaption=False)
    #     # cv2.imwrite(
    #     #     f"/home/biwi/data/images4code/ai_model_ckpt/manu_train/sjht/zjhs/sjht-zjhs-53/tmp/111/{time.time()}.jpg",
    #     #     img)

    if globalvar.globalvar.config.model_set["fill_box2square"]:
        img = fill(img, value=128)
        # cv2.imwrite(
        #     f"/home/biwi/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-280-281/tmp/111/{time.time()}.jpg",
        #     img)

    img = cv2.resize(img, tuple(img_size))

    # # 翻转
    # choice = random.sample(choice_list, 1)[0]
    # if choice == 0:
    #     img = flip_rl(img)
    #
    # choice = random.sample(choice_list, 1)[0]
    # if choice == 0:
    #     img = flip_ud(img)

    choice = random.sample(choice_list, 1)[0]
    if choice == 0:
        img = img_addcontrast_brightness(img)

    # # 随机旋转90 180
    # choice = random.sample(choice_list, 1)[0]
    # if choice == 0:
    #     img = rotation_random(img)
    #
    # # 在 -45 到 45 之间旋转
    # choice = random.sample(choice_list, 1)[0]
    # if choice == 0:
    #     img = radom_rotation(img)
    #     # angle = random.randint(-90, 90)
    #     # img = imRotate(img, angle, border_value=128, adaption=True)
    #     # cv2.imwrite(f"/home/biwi/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-280-281/tmp/111/{time.time()}.jpg", img)

    choice = random.sample(choice_list, 1)[0]
    if choice == 0:
        img = img_blur(img)

    choice = random.sample(choice_list, 1)[0]
    if choice == 0:
        img = random_color_distort(img)

    transform = A.Compose(
        [

            A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.1, alpha_affine=120 * 0.03),
            # A.GridDistortion(p=1),
            # A.OpticalDistortion(distort_limit=1, shift_limit=0.1, p=1),
            # A.ElasticTransform(alpha=1, sigma=10, alpha_affine=20, interpolation=1, border_mode=0, value=0,
            #                    mask_value=None, always_apply=False, approximate=False, p=0.5)

        ])
    choice = random.sample(choice_list, 1)[0]
    choice = 1
    if choice == 0:
        data_transformed = transform(image=img)
        img = data_transformed["image"]

    return img


def parse_data(line, class_num, img_size, mode):
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    pic_path = s[0]
    y = s[1]
    img = cv2.imread(pic_path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        globalvar.globalvar.logger.info_ai(meg="read img failed:%s" % pic_path)
    if mode.decode() == 'train':
        try:
            img = data_augmentation(img, img_size, int(y), pic_path)
        except Exception as e:
            print(pic_path)
    else:
        img = data_augmentation(img, img_size, int(y), pic_path)

    img = np.asarray(img, np.float32)

    y_true = np.zeros(class_num)
    y_true[int(y)] = 1.
    img = np.asarray(img, np.float32)
    img = img / 255.
    y_true = np.asarray(y_true, np.float32)
    return img, y_true, np.asarray(int(y), np.int32)


def center_loss(features, label, alfa, nrof_classes):
    # embedding的维度
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
    label = tf.reshape(label, [-1])
    # 挑选出每个batch对应的centers [batch,nrof_features]
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    # 相同类别会累计相减
    centers = tf.scatter_sub(centers, label, diff)
    # 先更新完centers在计算loss
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def center_distance_loss(features, label, alfa, nrof_classes):
    # embedding的维度
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
    label = tf.reshape(label, [-1])
    # 挑选出每个batch对应的centers [batch,nrof_features]
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    # 相同类别会累计相减
    centers = tf.scatter_sub(centers, label, diff)
    # 先更新完centers在计算loss
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
        distance = tf.constant(0.)
        num_center = tf.cast(tf.shape(centers)[0], tf.int32)

        def loop_body(id, distance):
            center_one = tf.tile(centers[id:id + 1, :], [nrof_classes, 1])
            distance_one = tf.reduce_mean(tf.square(center_one - centers))
            distance = distance + distance_one
            return id + 1, distance

        _, distance = tf.while_loop(lambda id, *args: id < num_center, loop_body, [0, distance])

    return loss, centers, distance



def fill(image):
    ori_w, ori_h = image.shape[1], image.shape[0]
    if ori_w >= ori_h:
        tmp = int((ori_w - ori_h) / 2)
        image = cv2.copyMakeBorder(image, 0, tmp, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.copyMakeBorder(image, tmp, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    else:
        tmp = int((ori_h - ori_w) / 2)
        image = cv2.copyMakeBorder(image, 0, 0, 0, tmp, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.copyMakeBorder(image, 0, 0, tmp, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def get_test_data(data_path, classes_list, name2id):
    child_paths = os.listdir(data_path)
    class_name = classes_list
    lines_temps = [[] for _ in class_name]
    for child_path in child_paths:
        child_path = os.path.join(data_path, child_path)
        for name in os.listdir(child_path):
            if name in list(name2id.keys()):
                idx = name2id[name]
                class_path = os.path.join(data_path, child_path, name)
                for img_name in os.listdir(class_path):
                    if img_name.split(".")[-1] == "jpg":
                        path = os.path.join(data_path, child_path, name, img_name)
                        line = path + " " + str(idx)
                        lines_temps[idx].append(line)

            else:
                globalvar.globalvar.logger.info_ai(meg="not find the name:%s in class names" % name)

    return lines_temps

def model_init_classify(num_class_classify, model_path_classify):
    classify_size = globalvar.globalvar.config.model_set["classify_size"]
    gpu_options_classify = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    graph_classify = tf.Graph()
    with graph_classify.as_default():
        input_data_classify = tf.placeholder(tf.float32, [1, classify_size[1], classify_size[0], 3], name='input_data')
        input_data_classify_fp32 = tf.cast(input_data_classify, tf.float32) / 255.
        yolo_model_classfy = yolov3_classfy(num_class_classify)
        with tf.variable_scope('yolov3_classfication'):
            logits, center_feature = yolo_model_classfy.forward(input_data_classify_fp32, is_training=False)
        logits = tf.squeeze(logits)
        scores = tf.nn.softmax(logits)
        labels_classify = tf.squeeze(tf.argmax(scores, axis=0))
        score_classify = tf.gather(scores, labels_classify)
        saver_classify = tf.train.Saver(
            var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3_classfication"]))
    sess_classify = tf.Session(graph=graph_classify, config=tf.ConfigProto(gpu_options=gpu_options_classify, allow_soft_placement=True))
    saver_classify.restore(sess_classify, model_path_classify)

    return sess_classify, input_data_classify, score_classify, labels_classify


def test_data_from_dir(test_dir, test_model, save_result_path, classes_list, name2id, gpu_device):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # with tf.Session(config=config) as sess:
        # with tf.device(gpu_device[0]):
        #     input_data = tf.placeholder(tf.float32, [1, 192, 192, 3], name='input_data')
        #     yolo_model_classfy = darknet_plus(len(classes_list))
        #     with tf.variable_scope('yolov3_classfication'):
        #         logits, center_feature = yolo_model_classfy.forward(input_data, is_training=False)
        # saver = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3_classfication"]))
        # class_result = tf.identity(logits, name='class_result')
        # saver.restore(sess, test_model)
    sess_classify, input_data_classify, score_classify, labels_classify = model_init_classify(len(classes_list), test_model)

    globalvar.globalvar.logger.info_ai("load classify model from:%s"%test_model)

    lines_temps = get_test_data(test_dir, classes_list, name2id)
    data_result = [[] for _ in classes_list]
    for index, test_data in enumerate(lines_temps):
        test_num = len(test_data)
        test_id = 0
        for line in test_data:
            if test_id % 50 == 0:
                globalvar.globalvar.logger.info_ai("start test class:%s, num:%d, %d/%d"%(classes_list[index], test_num, test_id, test_num))
            test_id = test_id + 1
            # pic_path, class_id = line.split(" ")[0], line.split(" ")[1]
            pic_path, class_gt = line.split(" ")[0], line.split(" ")[0].split("/")[-2]
            img = cv2.imread(pic_path)
            if globalvar.globalvar.config.model_set["fill_box2square"]:
                img = fill(img)
            img = cv2.resize(img, (192, 192))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = np.asarray(img, np.float32)
            # img = img[np.newaxis, :] / 255.
            img = img[np.newaxis, :]

            score, label = sess_classify.run([score_classify, labels_classify], {input_data_classify:img})

            # score_classify, label_classify = self.sess_classify.run([self.score_classify, self.labels_classify],
            #                                            feed_dict={
            #                                                self.input_data_classify: img_classify})

            # labels_max = np.argmax(labels_, axis=1)[0]
            # labels_ = np.reshape(labels_, (-1))
            # y = softmax(labels_)
            # score = y[labels_max]
            # result = [pic_path, classes_list[int(class_id)], classes_list[labels_max], score]
            result = [pic_path, class_gt, classes_list[label], score]
            data_result[index].append(result)
    right_data_score = []
    wrong_data_score = []
    np.save(save_result_path, np.array(data_result))

    for index, data_result in enumerate(data_result):
        globalvar.globalvar.logger.info_ai("start statistic class:%s"%classes_list[index])
        right_data_score_temp = []
        wrong_data_score_temp = []
        for data in data_result:
            if data[1] == data[2]:
                right_data_score.append(data[3])
                right_data_score_temp.append(data[3])
            else:
                wrong_data_score.append(data[3])
                wrong_data_score_temp.append(data[3])
        right_data_score_temp = np.array(right_data_score_temp)
        wrong_data_score_temp = np.array(wrong_data_score_temp)
        globalvar.globalvar.logger.info_ai("%s test result, all data num:%d, right data num:%d, wrong data num:%d" %
                                           (classes_list[index], len(right_data_score_temp)+len(wrong_data_score_temp), len(right_data_score_temp), len(wrong_data_score_temp)))
    right_data_score = np.array(right_data_score)
    wrong_data_score = np.array(wrong_data_score)
    globalvar.globalvar.logger.info_ai("all data, all data num:%d, right data num:%d, wrong data num:%d"%
                                       (len(right_data_score)+len(wrong_data_score), len(right_data_score), len(wrong_data_score)))


# def test_data_from_dir(test_dir, test_model, save_result_path, classes_list, name2id, gpu_device):
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     with tf.Session(config=config) as sess:
#         with tf.device(gpu_device[0]):
#             input_data = tf.placeholder(tf.float32, [1, 192, 192, 3], name='input_data')
#             yolo_model_classfy = darknet_plus(len(classes_list))
#             with tf.variable_scope('yolov3_classfication'):
#                 logits, center_feature = yolo_model_classfy.forward(input_data, is_training=False)
#         saver = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3_classfication"]))
#         class_result = tf.identity(logits, name='class_result')
#         saver.restore(sess, test_model)
#         globalvar.globalvar.logger.info_ai("load classify model from:%s"%test_model)
#
#         lines_temps = get_test_data(test_dir, classes_list, name2id)
#         data_result = [[] for _ in classes_list]
#         for index, test_data in enumerate(lines_temps):
#             test_num = len(test_data)
#             test_id = 0
#             for line in test_data:
#                 if test_id % 50 == 0:
#                     globalvar.globalvar.logger.info_ai("start test class:%s, num:%d, %d/%d"%(classes_list[index], test_num, test_id, test_num))
#                 test_id = test_id + 1
#                 pic_path, class_id = line.split(" ")[0], line.split(" ")[1]
#                 img = cv2.imread(pic_path)
#
#                 if globalvar.globalvar.config.model_set["fill_box2square"]:
#                     img = fill(img)
#                 img = cv2.resize(img, (192, 192))
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#                 img = np.asarray(img, np.float32)
#                 img = img[np.newaxis, :] / 255.
#
#                 labels_ = sess.run([class_result], {input_data:img})[0]
#                 labels_max = np.argmax(labels_, axis=1)[0]
#                 labels_ = np.reshape(labels_, (-1))
#                 y = softmax(labels_)
#                 score = y[labels_max]
#                 result = [pic_path, classes_list[int(class_id)], classes_list[labels_max], score]
#                 data_result[index].append(result)
#         right_data_score = []
#         wrong_data_score = []
#         np.save(save_result_path, np.array(data_result))
#
#         for index, data_result in enumerate(data_result):
#             globalvar.globalvar.logger.info_ai("start statistic class:%s"%classes_list[index])
#             right_data_score_temp = []
#             wrong_data_score_temp = []
#             for data in data_result:
#                 if data[1] == data[2]:
#                     right_data_score.append(data[3])
#                     right_data_score_temp.append(data[3])
#                 else:
#                     wrong_data_score.append(data[3])
#                     wrong_data_score_temp.append(data[3])
#             right_data_score_temp = np.array(right_data_score_temp)
#             wrong_data_score_temp = np.array(wrong_data_score_temp)
#             globalvar.globalvar.logger.info_ai("%s test result, all data num:%d, right data num:%d, wrong data num:%d" %
#                                                (classes_list[index], len(right_data_score_temp)+len(wrong_data_score_temp), len(right_data_score_temp), len(wrong_data_score_temp)))
#         right_data_score = np.array(right_data_score)
#         wrong_data_score = np.array(wrong_data_score)
#         globalvar.globalvar.logger.info_ai("all data, all data num:%d, right data num:%d, wrong data num:%d"%
#                                            (len(right_data_score)+len(wrong_data_score), len(right_data_score), len(wrong_data_score)))

def get_wrong_data(save_result_path, wrong_data_save_path, classes_list, score=0.5, do_save=False):
    data_result = np.load(save_result_path, allow_pickle=True)
    all_num = 0
    wrong_num = 0
    right_num = 0
    not_find_num = 0
    for index, data_result in enumerate(data_result):
        if len(data_result) == 0:
            globalvar.globalvar.logger.info_ai(meg="test num is 0:%s"%classes_list[index])
            continue
        all_num = all_num + len(data_result)
        data_result = np.array(data_result)
        data_result_find_idx = np.where(data_result[:,3].astype(np.float32) >= score)
        data_result_find = data_result[data_result_find_idx]
        data_result_not_find_idx = np.where(data_result[:, 3].astype(np.float32) < score)
        data_result_not_find = data_result[data_result_not_find_idx]
        not_find_num = not_find_num + len(data_result_not_find)

        data_result_find_wrong_idx = np.where(data_result_find[:,1] != data_result_find[:,2])
        data_result_find_wrong = data_result_find[data_result_find_wrong_idx]
        wrong_num = wrong_num + len(data_result_find_wrong)
        data_result_find_right_idx = np.where(data_result_find[:,1] == data_result_find[:,2])
        data_result_find_right = data_result_find[data_result_find_right_idx]
        right_num = right_num + len(data_result_find_right)

        if do_save:
            for data in data_result_find_wrong:
                wrong_data_save_path_class = os.path.join(wrong_data_save_path, classes_list[index], "class_wrong")
                if not os.path.exists(wrong_data_save_path_class):
                    os.makedirs(wrong_data_save_path_class)
                save_name = "%f_%s_%s.jpg"%(float(data[3]), data[1], data[2])
                save_path_temp = os.path.join(wrong_data_save_path_class, save_name)
                copyfile(data[0], save_path_temp)
            for data in data_result_not_find:
                wrong_data_save_path_class = os.path.join(wrong_data_save_path, classes_list[index], "not_find")
                if not os.path.exists(wrong_data_save_path_class):
                    os.makedirs(wrong_data_save_path_class)
                save_name = "%f_%s_%s.jpg" % (float(data[3]), data[1], data[2])
                save_path_temp = os.path.join(wrong_data_save_path_class, save_name)
                copyfile(data[0], save_path_temp)
    globalvar.globalvar.logger.info_ai("score:%f, all num:%d, right num:%d, wrong num：%d, not find num:%d"%(score, all_num, right_num, wrong_num, not_find_num))
    globalvar.globalvar.logger.info_ai("score:%f, all num:%f, right num:%f, wrong num：%f, not find num:%f" % (score, float(all_num)/float(all_num), float(right_num)/float(all_num), float(wrong_num)/float(all_num), float(not_find_num)/float(all_num)))
    # return score, float(all_num)/float(all_num), float(right_num)/float(all_num), float(wrong_num)/float(all_num), float(not_find_num)/float(all_num)
