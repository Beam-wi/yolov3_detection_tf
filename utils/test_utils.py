import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from utils.misc_utils import parse_anchors
import utils.globalvar as globalvar
from utils.nms_utils import gpu_nms
from utils.model import yolov3_trt
from utils.model import darknet_plus as yolov3_classfy
#
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


def nms_cpu(boxes, scores, classes, classes_score, labels_nms_threshold, num_classes):
    boxes_temp = []
    scores_temp = []
    classes_temp = []
    classes_score_temp = []
    for c in range(num_classes):
        iou_thresh = labels_nms_threshold[c]
        indices = np.where(classes == c)[0]
        if indices.shape[0] == 0:
            continue
        boxes_cls = boxes[indices, :]
        scores_cls = scores[indices]
        classes_cls = classes[indices]
        classes_score_cls = classes_score[indices]
        all_detections_cls = np.column_stack((boxes_cls, scores_cls))
        retain_idx = nms(all_detections_cls, iou_thresh)
        boxes_temp.append(boxes_cls[retain_idx])
        scores_temp.append(scores_cls[retain_idx])
        classes_temp.append(classes_cls[retain_idx])
        classes_score_temp.append(classes_score_cls[retain_idx])

    boxes = np.concatenate(boxes_temp, axis=0)
    scores = np.concatenate(scores_temp, axis=0)
    classes = np.concatenate(classes_temp, axis=0)
    classes_score = np.concatenate(classes_score_temp, axis=0)

    return boxes, scores, classes, classes_score


class qianjian_detection:
    def __init__(self, project_path):
        self.model_path_detection = os.path.join(globalvar.globalvar.config.data_set["model_and_temp_file_save_path"], globalvar.globalvar.config.data_set["detection_model_save_dir_name"], globalvar.globalvar.config.model_set["detection_model_save_name"])
        assert os.path.exists(self.model_path_detection + ".meta"), "detection model not exit:%s" % self.model_path_detection
        self.model_path_classify = os.path.join(globalvar.globalvar.config.data_set["model_and_temp_file_save_path"], globalvar.globalvar.config.data_set["classify_model_save_dir_name"], globalvar.globalvar.config.model_set["classify_model_save_name"])
        if globalvar.globalvar.config.test_other_info_set["do_classify"]:
            assert os.path.exists(self.model_path_classify + ".meta"), "classify model not exit:%s" % self.model_path_classify
        self.score_thresh_detection = globalvar.globalvar.config.test_other_info_set["score_thresh_detection"]
        self.score_thresh_classify = globalvar.globalvar.config.test_other_info_set["score_thresh_classify"]
        self.labels_nms_threshold = globalvar.globalvar.config.test_other_info_set["labels_nms_threshold"]
        self.backbone_name = globalvar.globalvar.config.model_set["backbone_name"]
        self.train_with_gray = globalvar.globalvar.config.model_set["train_with_gray"]
        self.train_with_two_feature_map = globalvar.globalvar.config.model_set["train_with_two_feature_map"]
        # self.class_name_detection, self.name2id_classify = self.get_class_name(os.path.join(project_path, globalvar.globalvar.config.data_set["detection_class_name_path"])) if globalvar.globalvar.config.test_other_info_set['do_classify'] else self.get_class_name(globalvar.globalvar.config.data_set['class_name_path'])
        self.class_name_detection, self.name2id_classify = self.get_class_name(os.path.join(project_path, globalvar.globalvar.config.data_set["detection_class_name_path"])) if globalvar.globalvar.config.data_set['mergeLabel'] else self.get_class_name(globalvar.globalvar.config.data_set['class_name_path'])
        self.num_class_detection = len(self.class_name_detection)
        self.class_name_classify, self.name2id_classify = self.get_class_name(globalvar.globalvar.config.data_set["class_name_path"])
        self.num_class_classify = len(self.class_name_classify)
        self.new_size_detection = globalvar.globalvar.config.test_other_info_set["new_size_detection"] # w,h
        self.new_size_classify = globalvar.globalvar.config.model_set["classify_size"]
        # self.anchors_detection = parse_anchors(globalvar.globalvar.config.data_set["anchor_path"]) * 2.5
        self.anchors_detection = np.reshape(np.asarray(globalvar.globalvar.config.data_set["anchors"], np.float32), [-1, 2]) * globalvar.globalvar.config.data_set["anchorRotate"]
        self.size_min = globalvar.globalvar.config.test_other_info_set["size_min"]
        self.sess_detection = None
        self.sess_classify = None

    def get_class_name(self, class_name_path):
        class_name = []
        name2id = {}
        name2texture = {}
        textures = ["metal", "plastic01", "plastic02", "other"]
        name2color = {}
        color = ["black", "white", "silver", "other"]
        for line in open(class_name_path, "r").readlines():
            class_name.append(line.split("\n")[0].split(",")[0].split(" ")[0])

        id = 0
        for line in open(class_name_path, "r").readlines():
            for name in line.split("\n")[0].split(",")[0].split(" "):
                name2id[name] = id
                if len(line.split("\n")[0].split(",")) > 2:
                    name2texture_id = textures.index(line.split("\n")[0].split(",")[1])
                    name2color_id = color.index(line.split("\n")[0].split(",")[2])
                    name2texture[name] = name2texture_id
                    name2color[name] = name2color_id
            id = id + 1
        return class_name, name2id

    def model_init_detection(self):
        self.gpu_options_detection = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        self.graph_detection = tf.Graph()
        with self.graph_detection.as_default():
            self.input_data_detection = tf.placeholder(tf.float32,
                                                       [1, self.new_size_detection[1], self.new_size_detection[0], 3],
                                                       name='input_data')
            self.input_data_detection_fp32 = tf.cast(self.input_data_detection, tf.float32) / 255.
            yolo_model = yolov3_trt(self.num_class_detection, self.anchors_detection, self.backbone_name,
                                    self.train_with_two_feature_map)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data_detection_fp32, False,
                                                       train_with_gray=self.train_with_gray)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
            # pred_scores = pred_confs * pred_probs
            self.pred_boxes = pred_boxes
            self.pred_confs = pred_confs
            self.pred_probs = pred_probs
            pred_scores = pred_confs
            self.boxes_combine_detection, self.scores_combine_detection, self.labels_combine_detection = gpu_nms(
                pred_boxes, pred_scores, pred_probs, self.num_class_detection,
                max_boxes=50, score_thresh=self.score_thresh_detection,
                iou_thresh=globalvar.globalvar.config.test_other_info_set["iou_thresh"])
            saver_detection = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3"]))
        self.sess_detection = tf.Session(graph=self.graph_detection,
                                         config=tf.ConfigProto(gpu_options=self.gpu_options_detection,
                                                               allow_soft_placement=True))
        saver_detection.restore(self.sess_detection, self.model_path_detection)

    def model_init_classify(self):
        self.gpu_options_classify = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        self.graph_classify = tf.Graph()
        with self.graph_classify.as_default():
            self.input_data_classify = tf.placeholder(tf.float32,
                                                      [1, self.new_size_classify[1], self.new_size_classify[0], 3],
                                                      name='input_data')
            self.input_data_classify_fp32 = tf.cast(self.input_data_classify, tf.float32) / 255.
            yolo_model_classfy = yolov3_classfy(self.num_class_classify)
            with tf.variable_scope('yolov3_classfication'):
                logits, center_feature = yolo_model_classfy.forward(self.input_data_classify_fp32, is_training=False)
                self.logits = logits
                self.center_feature = center_feature
            logits = tf.squeeze(logits)
            scores = tf.nn.softmax(logits) if self.num_class_classify-1 else tf.nn.sigmoid(logits)
            self.scores = scores
            self.labels_classify = tf.squeeze(tf.argmax(scores, axis=0)) if self.num_class_classify-1 else tf.constant(0)
            self.score_classify = tf.gather(scores, self.labels_classify) if self.num_class_classify-1 else scores
            saver_classify = tf.train.Saver(
                var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3_classfication"]))
        self.sess_classify = tf.Session(graph=self.graph_classify,
                                        config=tf.ConfigProto(gpu_options=self.gpu_options_classify,
                                                              allow_soft_placement=True))
        saver_classify.restore(self.sess_classify, self.model_path_classify)


    def clamp_bboxs(self, boxes, img_size, to_remove=1):
        boxes[:, 0] = boxes[:, 0].clip(min=0, max=img_size[0] - to_remove)
        boxes[:, 1] = boxes[:, 1].clip(min=0, max=img_size[1] - to_remove)
        boxes[:, 2] = boxes[:, 2].clip(min=0, max=img_size[0] - to_remove)
        boxes[:, 3] = boxes[:, 3].clip(min=0, max=img_size[1] - to_remove)

        return boxes

    def fill_new(self, image):
        ori_w, ori_h = image.shape[1], image.shape[0]
        new_size = max(ori_h, ori_w)
        new_img = np.zeros([new_size, new_size, 3], dtype=np.float32)+128  # np.float32 np.uint8
        x0 = int((new_size - ori_w) / 2)
        y0 = int((new_size - ori_h) / 2)
        new_img[y0: y0 + ori_h, x0:x0 + ori_w, :] = image
        return new_img

    def extension_boxes(self, boxes, extension_ratio=1.2):
        x_center = (boxes[:, 2] + boxes[:, 0]) / 2
        y_center = (boxes[:, 3] + boxes[:, 1]) / 2
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]

        x0 = x_center - boxes_w * extension_ratio / 2
        x1 = x_center + boxes_w * extension_ratio / 2
        y0 = y_center - boxes_h * extension_ratio / 2
        y1 = y_center + boxes_h * extension_ratio / 2
        boxes_extension = np.concatenate([x0[:, np.newaxis], y0[:, np.newaxis], x1[:, np.newaxis], y1[:, np.newaxis]],
                                         axis=1)
        boxes_extension = np.asarray(boxes_extension, dtype=np.int32)

        return boxes_extension

    def forward(self, img_ori, do_classify=False):
        height_ori, width_ori = img_ori.shape[:2]
        # t1 = time.time()
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        # print(f"BGR2RGB: {time.time()-t1}")
        # t2 = time.time()
        img_detection = cv2.resize(img_ori, tuple(self.new_size_detection))
        img_detection = img_detection[np.newaxis, :]
        # print(f"ARGUMENT: {time.time()-t2}")
        # t3 = time.time()
        boxes_detection, scores_detection, labels_detection, pred_boxes_, pred_confs_, pred_probs_ = self.sess_detection.run(
            [self.boxes_combine_detection, self.scores_combine_detection, self.labels_combine_detection, self.pred_boxes, self.pred_confs, self.pred_probs],
            feed_dict={self.input_data_detection: img_detection})
        # print(f'DET INFER: {time.time()-t3}')

        # scores_detection = np.array([0.954188346862793, 0.9527560472488403, 0.9500169157981873, 0.9462890625, 0.94580078125, 0.9375, 0.93115234375, 0.95947265625, 0.958984375, 0.95751953125, 0.955651581287384, 0.95458984375, 0.95263671875, 0.95166015625, 0.94677734375], dtype=np.float32)
        # labels_detection = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        boxes_detection[:, 0] *= (width_ori / float(self.new_size_detection[0]))
        boxes_detection[:, 2] *= (width_ori / float(self.new_size_detection[0]))
        boxes_detection[:, 1] *= (height_ori / float(self.new_size_detection[1]))
        boxes_detection[:, 3] *= (height_ori / float(self.new_size_detection[1]))

        boxes_detection = self.clamp_bboxs(boxes_detection, [width_ori, height_ori])

        labels_detection_name = []
        for label in labels_detection:
            labels_detection_name.append(self.class_name_detection[label])
        labels_detection_name = np.array(labels_detection_name)

        if not do_classify: #如果不做分类，返回的分类名称就是检测嵌件中的名称，返回的得分都是1
            # scores_classify = [1. for i in range(len(boxes_detection))]
            # scores_classify = np.asarray(scores_classify)
            scores_classify = np.asarray(scores_detection)
            labels_classify = labels_detection
            # return boxes_detection, scores_detection, labels_detection_name, scores_classify, labels_detection_name

        else:
            labels_classify = []
            scores_classify = []

            # boxes_detection = np.array([[4438, 2487, 4482, 2536], [1516, 1761, 1554, 1810], [4507, 2321, 4551, 2365], [4300, 1306, 4344, 1347], [4005, 1404, 4049, 1442], [4082, 1379, 4125, 1425], [4165, 1831, 4209, 1872], [4011, 972, 4087, 1043], [1629, 143, 1699, 213], [1720, 726, 1790, 790], [2905, 1237, 2970, 1292], [2303, 1229, 2374, 1278], [2850, 237, 2927, 296], [2108, 381, 2195, 444], [3674, 662, 3762, 725]])

            if len(boxes_detection) > 0:
                boxes_detection_extension = self.extension_boxes(boxes_detection, extension_ratio=globalvar.globalvar.config.model_set["extension_ratio"])
                boxes_detection_extension = self.clamp_bboxs(boxes_detection_extension, [width_ori, height_ori])
            else:
                boxes_detection_extension = np.array([])
            # t4 = time.time()
            for box in boxes_detection_extension:
                x0, y0, x1, y1 = box
                img_cut = img_ori[int(y0): int(y1), int(x0): int(x1), :]
                #如果设置了填充为矩形，则分类前进行填充，再进行分类
                if globalvar.globalvar.config.model_set["fill_box2square"]:
                    img_cut = self.fill_new(img_cut)
                img_cut = cv2.resize(img_cut, tuple(self.new_size_classify))
                img_classify = img_cut[np.newaxis, :]

                # path = "/home/biwi/data/images4code/ai_model_ckpt/manu_train/sjht/lzbl/sjht-lzbl-271/tmp/tpdc_background_ysld06h_0.94170_0.61196_6_0_1626676802.5418835_test.png"
                # img_ori = cv2.imread(path)
                # img_ori = self.fill_new(img_ori)
                # img = cv2.resize(img_ori, (192, 192))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = np.asarray(img, np.float32)
                # img = img[np.newaxis, :]
                # score_classify, label_classify, logits_, center_feature_, scores_ = self.sess_classify.run([self.score_classify, self.labels_classify, self.logits, self.center_feature, self.scores],
                #                                                         feed_dict={self.input_data_classify: img_classify})
                #                                                         # feed_dict={self.input_data_classify: img})

                score_classify, label_classify = self.sess_classify.run([self.score_classify, self.labels_classify],
                                                                        feed_dict={self.input_data_classify: img_classify})
                                                                        # feed_dict={self.input_data_classify: img})

                scores_classify.append(score_classify)
                labels_classify.append(label_classify)
            # print(f"CLS INFER: {time.time()-t4}")

        labels_nms_threshold = self.labels_nms_threshold if isinstance(self.labels_nms_threshold, list) else [self.labels_nms_threshold for _ in range(self.num_class_classify)]
        if len(boxes_detection) > 0:
            labels_classify = np.array(labels_classify) if isinstance(labels_classify, list) else labels_classify
            scores_classify = np.asarray(scores_classify)
            boxes_detection, scores_detection, labels_classify, scores_classify = nms_cpu(boxes_detection,
            # boxes_detection, scores_detection, labels_classify, scores_classify = nms_cpu(self.extension_boxes(boxes_detection, extension_ratio=globalvar.globalvar.config.model_set["extension_ratio"]),
                                                                                          scores_detection,
                                                                                          labels_classify,
                                                                                          scores_classify,
                                                                                          labels_nms_threshold,
                                                                                          num_classes=self.num_class_classify)
            labels_classify = labels_classify.tolist()
            scores_classify = scores_classify.tolist()

        labels_classify_name = []
        for label in labels_classify:
            labels_classify_name.append(self.class_name_classify[label])

        scores_classify = np.array(scores_classify)

        labels_classify_name = np.array(labels_classify_name)

        return boxes_detection, scores_detection, labels_detection_name, scores_classify, labels_classify_name

    def det(self, img_ori, do_classify=False):
        height_ori, width_ori = img_ori.shape[:2]
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        img_detection = cv2.resize(img_ori, tuple(self.new_size_detection))
        img_detection = img_detection[np.newaxis, :]

        boxes_detection, scores_detection, labels_detection, pred_boxes_, pred_confs_, pred_probs_ = self.sess_detection.run(
            [self.boxes_combine_detection, self.scores_combine_detection, self.labels_combine_detection, self.pred_boxes, self.pred_confs, self.pred_probs],
            feed_dict={self.input_data_detection: img_detection})

        boxes_detection[:, 0] *= (width_ori / float(self.new_size_detection[0]))
        boxes_detection[:, 2] *= (width_ori / float(self.new_size_detection[0]))
        boxes_detection[:, 1] *= (height_ori / float(self.new_size_detection[1]))
        boxes_detection[:, 3] *= (height_ori / float(self.new_size_detection[1]))

        boxes_detection = self.clamp_bboxs(boxes_detection, [width_ori, height_ori])

        labels_detection_name = []
        for label in labels_detection:
            labels_detection_name.append(self.class_name_detection[label])
        labels_detection_name = np.array(labels_detection_name)

        if not do_classify:  # 如果不做分类，返回的分类名称就是检测嵌件中的名称，返回的得分都是1
            scores_classify = [1. for i in range(len(boxes_detection))]
            scores_classify = np.asarray(scores_classify)
            return boxes_detection, scores_detection, labels_detection_name, scores_classify, labels_detection_name

        labels_classify = []
        scores_classify = []
        if len(boxes_detection) > 0:
            boxes_detection_extension = self.extension_boxes(boxes_detection, extension_ratio=globalvar.globalvar.config.model_set["extension_ratio"])
            boxes_detection_extension = self.clamp_bboxs(boxes_detection_extension, [width_ori, height_ori])
        else:
            boxes_detection_extension = np.array([])

        for box in boxes_detection_extension:
            x0, y0, x1, y1 = box
            img_cut = img_ori[int(y0): int(y1), int(x0): int(x1), :]
            # 如果设置了填充为矩形，则分类前进行填充，再进行分类
            if globalvar.globalvar.config.model_set["fill_box2square"]:
                img_cut = self.fill_new(img_cut)
            img_cut = cv2.resize(img_cut, tuple(self.new_size_classify))
            img_classify = img_cut[np.newaxis, :]
            score_classify, label_classify = self.sess_classify.run(
                [self.score_classify, self.labels_classify],
                feed_dict={self.input_data_classify: img_classify})
                # feed_dict={self.input_data_classify: img})

            scores_classify.append(score_classify)
            labels_classify.append(label_classify)
        labels_nms_threshold = self.labels_nms_threshold if isinstance(self.labels_nms_threshold, list) else [self.labels_nms_threshold for _ in range(self.num_class_classify)]

        if len(boxes_detection) > 0:
            labels_classify = np.array(labels_classify)
            scores_classify = np.asarray(scores_classify)
            boxes_detection, scores_detection, labels_classify, scores_classify = \
                nms_cpu(boxes_detection, scores_detection, labels_classify,
                        scores_classify, labels_nms_threshold, num_classes=self.num_class_classify)
            labels_classify = labels_classify.tolist()
            scores_classify = scores_classify.tolist()

        labels_classify_name = []
        for label in labels_classify:
            labels_classify_name.append(self.class_name_classify[label])

        scores_classify = np.array(scores_classify)

        labels_classify_name = np.array(labels_classify_name)

        return boxes_detection, scores_detection, labels_detection_name, scores_classify, labels_classify_name


def show_box(img_show, true_boxes, true_labels, pred_boxes, pred_labels, scores_detection_select,
             scores_classify_select, img_test_path=None):
    for i in range(len(true_boxes)):
        x0, y0, x1, y1 = true_boxes[i]
        cv2.rectangle(img_show, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 4)
        cv2.putText(img_show, "%s" % (true_labels[i]), (int(x0), int(y0) - 16), 0, 2, [255, 255, 0], 5,
                    lineType=cv2.LINE_AA)
    #
    for i in range(len(pred_boxes)):
        x0, y0, x1, y1 = pred_boxes[i]
        cv2.rectangle(img_show, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 4)
        cv2.putText(img_show, "%s_%.2f_%.2f" % (pred_labels[i], scores_detection_select[i], scores_classify_select[i]),
                    (int(x0), int(y0) - 66), 0, 1, [0, 255, 0], 2,
                    lineType=cv2.LINE_AA)
    img_show = cv2.resize(img_show, (900, 900))
    cv2.imshow("test", img_show)
    cv2.waitKey(0)


def draw_box(img_show, true_boxes, true_labels, pred_boxes, pred_labels, scores_detection_select,
             scores_classify_select, tmw_dict, img_test_path, img_path_dir):
    """
    画图并保存图片
        真实框：蓝色:rgb(255,0,0)
        检测正确：绿色:rgb(0,255,0)
        检测误报：红色:rgb(0,0,255)
        漏检：紫色:rgb(255,0,255)
    args:
    tmw_dict:
            {'T': {'b': np.ndarray(n, 4),
               'l': [n]},
             'M': {'b': np.ndarray(m, 4),
                   'l': [m]},
             'W': {'b': np.ndarray(l, 4),
                   'l': [l]}
            }
    """
    pigment = {'T': (0, 255, 0), 'W': (0, 0, 255), 'M': (255, 0, 255)}
    pigment_ = {'c-hslqj02h': (0, 255, 255), 'c-hslqj02h-yc': (255, 0, 255), 'huiseslqj01h': (255, 255, 0), 'huiseslqj01h-yc': (255, 0, 0)}
    pred_boxes = np.array(pred_boxes)
    for i in range(len(true_boxes)):
        x0, y0, x1, y1 = true_boxes[i]
        cv2.rectangle(img_show, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 4)
        cv2.putText(img_show, "%s" % (true_labels[i]), (int(x0), int(y0) - 16), 0, 2, (255, 0, 0), 4,
                    lineType=cv2.LINE_AA)
    for k, v in tmw_dict.items():
        for i in range(v['b'].shape[0]):
            x0, y0, x1, y1 = v['b'][i]
            cv2.rectangle(img_show, (int(x0), int(y0)), (int(x1), int(y1)), pigment[k], 2)
            if k == 'M':
                # cv2.putText(img_show, "%s" % (v['l'][i]), (int(x0), int(y0) - 66), 0, 1, pigment[k], 2, lineType=cv2.LINE_AA)
                cv2.putText(img_show, "%s" % (v['l'][i]), (int(x0), int(y0) - 66), 0, 1, pigment[k], 2, lineType=cv2.LINE_AA)
            else:
                ind = np.where(pred_boxes==v['b'][i])[0][0]
                cv2.putText(img_show, "%s_%.2f_%.2f" % (v['l'][i], scores_detection_select[ind], scores_classify_select[ind]), (int(x0), int(y0) - 66), 0, 1, pigment[k], 2, lineType=cv2.LINE_AA)
                # cv2.putText(img_show, "%s_%.2f_%.2f" % (v['l'][i], scores_detection_select[ind], scores_classify_select[ind]), (int(x0), int(y0) - 66), 0, 2, pigment_[v['l'][i]], 2, lineType=cv2.LINE_AA)

    if not os.path.exists(img_path_dir):
        os.makedirs(img_path_dir)
    print(f"save img {img_test_path.split('/')[-1]}")
    cv2.imwrite(f"{img_path_dir}/{img_test_path.split('/')[-1]}", img_show)
    # cv2.imwrite(f"/home/biwi/data/workdirs/test/bdnb/bdnb-188-02/{img_test_path.split('/')[-1]}", img_show)


