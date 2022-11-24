#!/usr/bin/env python2
#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import os
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from torchvision import transforms as TT

import sys, os

# sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def get_engine(onnx_file_path, image_size, engine_file_path="", calib=None,
               ndtype="float16", dynamic_input=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
        # with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        #         1 << int(
        #             trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(
        #     network, TRT_LOGGER) as parser:
            start = time.time()

            # network.add_input("input_data", trt.float32, (-1, 3, 2006, 3009))
            builder.max_batch_size = image_size[0]  # always 1 for explicit batch
            if dynamic_input:
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 28  # 1*2^30=1073741824=1GB
            else:
                builder.max_workspace_size = 1 << 28  # 256MiB
            # builder.max_workspace_size = 4 << 30  # 256MiB

            if ndtype == 'int8':
                assert (builder.platform_has_fast_int8 == True), "not support int8"
                builder.int8_mode = True
                builder.int8_calibrator = calib
            elif ndtype == 'float16':
                assert (builder.platform_has_fast_fp16 == True), "not support fp16"
                builder.fp16_mode = True
            else:
                raise Exception('ndtype error !')

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                        onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                # amodel = model.read()
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            if dynamic_input:
                # optimization profile
                profile = builder.create_optimization_profile()
                # profile.set_shape("input_data", (1,192,192,3), (2,192,192,3), (8,192,192,3))
                # profile.set_shape("input_data", (1, 3, 120, 256), (1, 3, 512, 840), (1, 3, 768, 1024))  #, (1, 3, 1600, 2048) , (1, 3, 2016, 3008)
                # profile.set_shape("input_data", (1, 3, 768, 1024), (1, 3, 1600, 2048) , (1, 3, 2016, 3008))  #
                # profile.set_shape("input_data:0", (1, 768, 1024, 3),
                #                                 (1, 1600, 2048, 3),
                #                                 (1, 2016, 3008, 3))  # , (1, 3, 1600, 2048) , (1, 3, 2016, 3008)
                # profile.set_shape("input", (1, 3, 96, 96), (1, 3, 120, 120), (1, 3, 192, 192))
                # profile.set_shape("input", (1, 3, 2006, 3009), (3, 3, 2006, 3009), (6, 3, 2006, 3009))
                # profile.set_shape("input", (-1, 3, 692, 1024), (-1, 3, 1365, 2048), (-1, 3, 2006, 3009))
                # profile.set_shape("input", (1, 3, 2006, 3009), (3, 3, 2006, 3009), (6, 3, 2006, 3009))
                profile.set_shape("input", (1, 3, 692, 1024), (2, 3, 692, 1024), (3, 3, 692, 1024))
                # profile.set_shape("input_data", (1, 3, 2016, 3008), (2, 3, 2016, 3008), (8, 3, 2016, 3008))
                config.add_optimization_profile(profile)

                # # add sigmoid for output. for better results, you can shield below.
                # previous_output = network.get_output(0)
                # network.unmark_output(previous_output)
                # sigmoid_layer=network.add_activation(previous_output,trt.ActivationType.SIGMOID)
                # network.mark_output(sigmoid_layer.get_output(0))

                # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
                print('Completed parsing of ONNX file')
                print(
                    'Building an engine from file {}; this may take a while...'.format(
                        onnx_file_path))
                engine = builder.build_engine(network, config)
            else:
                # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
                network.get_input(0).shape = image_size
                print('Completed parsing of ONNX file')
                print(
                    'Building an engine from file {}; this may take a while...'.format(
                        onnx_file_path))
                engine = builder.build_cuda_engine(network)

            print(
                f"\nCompleted creating Engine. Take {(time.time() - start) / 60} minutes.")
            if not os.path.exists(os.path.dirname(engine_file_path)):
                os.makedirs(os.path.dirname(engine_file_path))
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(
                TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    # def __init__(self, training_data, cache_file, batch_size=64):
    def __init__(self, training_data, cache_file, image_size, batch_size=64,
                 prefix='', mode='det', dataset=None):
        """
        args:
            training_data: ann path for det or image path for cls.
            cache_file: the cache path if you have done this.
            image_size: must be len 4.
            batch_size: calibrator batch.
            prefix: only det image head.
            mode: 'det' or 'cls'.
            dataset: the data argument for you model,
                    you need add at the end of class.
                    now compatible for 'yolov3_dataset' and 'atss_dataset'.
        must method:
            get_batch_size
            get_batch
            read_calibration_cache
            write_calibration_cache
        """
        assert len(image_size) == 4, "image_size must be len 4!"
        if mode == 'det' and prefix == '':
            raise [ValueError, 'Prefix must be given when mode det.']
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.training_data = training_data
        self.image_size = image_size
        self.batch_size = batch_size
        self.prefix = prefix
        self.mode = mode
        self.dataset = dataset
        self.current_index = 0
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = self.load_det_data() if not mode == 'cls' else self.load_cls_data()

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(
            self.load_data(0).nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data):
            return None

        # Phase to print
        # current_batch = int(self.current_index / self.batch_size)
        # if current_batch % 10 == 0:
        #     print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        if self.current_index == 0:
            print("Calibrating batch, containing {:} images ...".format(
                len(self.data)))

        batch = self.load_data(self.current_index).ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        process_bar(self.current_index / (len(self.data) - 1), start_str='',
                    end_str=f"{len(self.data)}",
                    total_length=15)  # Draw progress bar
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def load_data(self, n):
        image = cv2.imread(self.data[n])
        assert isinstance(image,
                          np.ndarray), f'{self.data[n]} is not exist or not image.'
        # Return no batch image that fit your model.
        image = eval(f"self.{self.dataset}")(image)

        return image

    def load_det_data(self, ):
        data = list()
        with open(self.training_data, "r", encoding="utf-8") as file:
            dataset = [f"{self.prefix}/{x.split(' ')[0]}" for x in
                       file.read().split("\n")[:-1]]
        for sing in dataset:
            if '.jpg' in sing or '.png' in sing:
                data.append(sing)
            else:
                continue
        return data

    def load_cls_data(self, ):
        data = list()
        dataset = list()
        label_path = os.listdir(self.training_data)
        label_path.remove('wxqy')
        for label_path in [f"{self.training_data}/{label}" for label in
                           label_path]:
            dataset.extend(
                [f"{label_path}/{x}" for x in os.listdir(label_path)])
        for sing in dataset:
            if '.jpg' in sing or '.png' in sing:
                data.append(sing)
            else:
                continue
        return data

    def yolov3_dataset(self, img_ori):
        img = cv2.resize(img_ori,
                         (self.image_size[2], self.image_size[
                             1])) if not self.mode == 'cls' else self.fill_new(
            img_ori)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[np.newaxis, :] / 255.  # add dim and div 255.
        img = np.asarray(img, np.float32)

        return img

    def fill_new(self, image, size):
        """
        将图片变成正方行，再resize
        size: [w, h]
        """
        ori_w, ori_h = image.shape[1], image.shape[0]
        new_size = max(ori_h, ori_w)
        new_img = np.zeros([new_size, new_size, 3], dtype=np.float32)
        x0 = int((new_size - ori_w) / 2)
        y0 = int((new_size - ori_h) / 2)
        new_img[y0: y0 + ori_h, x0:x0 + ori_w, :] = image
        new_img = cv2.resize(new_img, (size[1], size[0]))
        return new_img

    def atss_dataset(self, image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = TT.Compose([TT.ToTensor(),
                                TT.Normalize(mean, std)])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[3], self.image_size[2]))

        return transform(image).numpy()


# Draw progress bar
def process_bar(percent, start_str='', end_str='', total_length=0):
    """
    demo:
        for i in range(101):
            time.sleep(0.1)
            end_str = '100%'
            process_bar(i/100, start_str='', end_str=end_str, total_length=15)
    """
    bar = ''.join(
        ["\033[31;41m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(
        percent * 100) + end_str
    print(bar, end='', flush=True)


# Draw the bounding boxes on the original input image and return it
def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories,
                bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width,
                    np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height,
                     np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12),
                  '{0} {1:.2f}'.format(all_categories[category], score),
                  fill=bbox_color)

    return image_raw





def main():
    """Create a TensorRT engine for ONNX-based model and run inference."""
    # float16
    # det
    # onnx_file_path = '/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210311_dy/onnx/5/model.onnx'
    # engine_file_path = "/mnt/data/bihua/data/model_repository/yolov3_tensorflow_/det_inserts_dy/6/model.plan"
    # # image_size = [1, -1, -1, 3]
    # image_size = [-1, -1, -1, 3]

    # # onnx_file_path = '/mnt/data/binovo/data/model/sjht-lzbl-271/onnx/det/4/model.onnx'
    # # engine_file_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/trt_model/det/4/model.plan"
    # onnx_file_path = '/tmp/Trt/onnx/model.onnx'
    # engine_file_path = "/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-281/1.0/model_and_temp_file/release_model_path/sjht_syht1_281_v1.0/trt_model/qj_detection/1/model.plan"
    # # image_size = [1, -1, -1, 3]
    # image_size = [1, 2016, 3008, 3]

    # mmdetection vfnet
    # onnx_file_path = '/mnt/data/bihua/data/models/sjht-orcode/mmdetection/vfnet/resnet18/20210512-5/onnx/1/model.onnx'
    # # onnx_file_path = '/mnt/data/bihua/data/models/sjht-orcode/mmdetection/vfnet/resnet18/20210512/onnx/3/model.onnx'
    # # onnx_file_path = '/mnt/data/bihua/data/models/sjht-orcode/mmdetection/vfnet/resnet50/20210512/onnx/model.onnx'
    # engine_file_path = "/mnt/data/bihua/data/model_repository/mmdetection/vfnet/resnet18/ORCODE20210512-5/1/model.plan"
    # # engine_file_path = "/mnt/data/bihua/data/models/sjht-orcode/mmdetection/vfnet/resnet18/20210512/trt/3/model.plan"
    # # engine_file_path = "/mnt/data/bihua/data/models/sjht-orcode/mmdetection/vfnet/resnet50/20210512/trt/model.plan"
    # # image_size = [1, 3, 1024, 1024]
    # image_size = [1, 3, 2016, 3040]
    # # image_size = [1, 3, 692, 1024]
    # # image_size = [1, 3, 2016, 3024]

    # cls
    # onnx_file_path = '/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106-1/onnx/model.onnx'
    # onnx_file_path = '/mnt/data/binovo/data/model/sjht-lzbl-271/onnx/cls/4/model.onnx'
    # engine_file_path = '/mnt/data/binovo/data/model/sjht-lzbl-271/trt_model/cls/4/model.plan'
    onnx_file_path = '/tmp/Trt/onnx/model.onnx'
    engine_file_path = '/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-281/1.0/model_and_temp_file/release_model_path/sjht_syht1_281_v1.0/trt_model/classfy_model/1/model.plan'
    image_size = [1, 192, 192, 3]

    # matching
    # onnx_file_path = '/mnt/data/bihua/data/model/yolo3Tensorflow/matching/onnx/2/model.onnx'
    # engine_file_path = "/mnt/data/bihua/data/model_repository/yolov3_tensorflow/matching_dy/5/model.plan"
    # image_size = [1, 3, -1, -1]

    # mmclassification
    # onnx_file_path = '/mnt/data/bihua/data/model/mmclassification/resnet50/SJHT20210330/onnx/epoch_100.onnx'
    # engine_file_path = "/mnt/data/bihua/data/model/mmclassification/resnet50/SJHT20210330/trt/model.plan"
    # image_size = [1, 3, 120, 120]





    # int8
    # det
    # onnx_file_path = '/mnt/data/zhouzhubin/tensorflow_model/sjht_bshw/det/onnx/model.onnx'
    # engine_file_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/sjht_bshw/trt_model_int8/qj_detection/1/model.plan"
    # train_data_file = "/mnt/data/bihua/data/ann/sjht/bshw/20210115_shbshw_train_add_background.txt"
    # cache_file = "cache/bshw_qj_det_calibration.cache"
    # prefix = "/mnt/data/bihua/data/images4code/images"
    # image_size = [1, 2016, 3008, 3]
    # mode = 'det'
    # calib = EntropyCalibrator(train_data_file, cache_file, image_size,
    #                           batch_size=1, prefix=prefix, mode=mode)

    # cls
    # onnx_file_path = '/mnt/data/zhouzhubin/tensorflow_model/sjht_bshw/cls/onnx/model.onnx'
    # engine_file_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/sjht_bshw/trt_model_int8/classfy_model/1/model.plan"
    # train_data_file = "/mnt/data/bihua/data/images4code/chenhong_data/box_img_shbshw/20210115_box_img"
    # cache_file = "cache/bshw_qj_cls_calibration.cache"
    # prefix = ''
    # image_size = [1, 192, 192, 3]
    # mode = 'cls'
    # calib = EntropyCalibrator(train_data_file, cache_file, image_size,
    #                           batch_size=1, prefix=prefix, mode=mode)

    # atss
    # onnx_file_path = '/mnt/data/zhouzhubin/tensorflow_model/cotton/atss/onnx/model.onnx'
    # engine_file_path = "/mnt/data/zhouzhubin/tensorflow_model/cotton/atss/trt8_2/model.plan"
    # train_data_file = "/mnt/data/zhouzhubin/tensorflow_model/cotton/atss/train_cotton_0604_2.txt"
    # cache_file = "cache/atss_det_calibration_0608.cache"
    # prefix = '/mnt/data/images4code'
    # image_size = [1, 3, 1280, 1280]
    # mode = 'det'
    # calib = EntropyCalibrator(train_data_file, cache_file, image_size,
    #                           batch_size=1, prefix=prefix, mode=mode,
    #                           dataset='atss_dataset')


    # 公共部分
    # int8
    # with get_engine(onnx_file_path, image_size, engine_file_path, calib=calib,
    #                 ndtype='int8') as engine, engine.create_execution_context() as context:
    #     print("finish!!!")

    # float16
    with get_engine(onnx_file_path, image_size, engine_file_path,
                    ndtype='float16',
                    dynamic_input=False) as engine, engine.create_execution_context() as context:
        print("finish!!!")


if __name__ == '__main__':
    main()
