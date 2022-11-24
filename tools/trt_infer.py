#!/usr/bin/env python3

import cv2
import json
import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from PIL import Image
from torchvision import transforms

def loadEngine2TensorRT(filepath):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(filepath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def cls_do_inference(engine, h_input):
    with engine.create_execution_context() as context:
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        # h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        h_output0 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        h_output1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(1 * h_input.size * h_input.dtype.itemsize)
        d_output0 = cuda.mem_alloc(h_output0.nbytes)
        d_output1 = cuda.mem_alloc(h_output1.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input), int(d_output0), int(d_output1)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output0, d_output0, stream)
        cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
    return h_output0, h_output1


def det_do_inference(engine, h_input):
    with engine.create_execution_context() as context:
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        h_input_ = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        h_output0 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        h_output1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
        h_output2 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(3)), dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(1 * h_input.size * h_input.dtype.itemsize)
        d_output0 = cuda.mem_alloc(h_output0.nbytes)
        d_output1 = cuda.mem_alloc(h_output1.nbytes)
        d_output2 = cuda.mem_alloc(h_output2.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        # bindings = [int(d_input), int(d_output0), int(d_output1), int(d_output2)]
        bindings = [int(d_input), int(d_output0), int(d_output1), int(d_output2)]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output0, d_output0, stream)
        cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
        cuda.memcpy_dtoh_async(h_output2, d_output2, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
    return h_output0, h_output1, h_output2




def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    shift = [(w-nw)//2, (h-nh)//2]
    return new_image, scale, shift


def torch_Normalized_gray(img):
    w, h = img.shape[1], img.shape[0]
    input_data = torch.from_numpy(img)
    # input_data = input_data.to(device)
    input_data = input_data.cuda()
    output_data = input_data / 255.
    # rgb_factor = np.array([0.2989, 0.5870, 0.1140])
    rgb_factor = torch.Tensor([0.2989, 0.5870, 0.1140]).unsqueeze(1).cuda()
    output_data_1 = torch.matmul(output_data, rgb_factor)
    output_data = torch.cat([output_data_1, output_data_1, output_data_1], dim=-1)
    out = output_data.cpu().data.numpy()
    out = out[np.newaxis, :]
    return out







if __name__ == "__main__":
    # det
    # # engine_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/tensorrt/yolov3.plan"
    # engine_path = "/mnt/data/bihua/data/model_repository/yolov3_tensorflow/cls/1/model.plan"
    # engine = loadEngine2TensorRT(engine_path)
    #
    # path = f"/mnt/data/bihua/LocalProject/yolo3Tensorflow/camera1_2020-12-23_07_53_02_441848.jpg"
    # img_ori = cv2.imread(path)
    # img = cv2.resize(img_ori, (192, 192))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.asarray(img, np.float32)
    # img = img[np.newaxis, :] / 255.
    # data_dict = dict()
    #
    # output = det_do_inference(engine, img)
    # # data_dict["pred_boxes"] = output[2].reshape(1, 145152).tolist()
    # data_dict["pred_confs"] = output[1].reshape(1, 36288, 1).tolist()
    # data_dict["pred_probs"] = output[0].reshape(1, 512, 1).tolist()
    # with open("./data/datadict_trt_.json", "w", encoding="utf-8") as file:
    #     json.dump(data_dict, file)
    # print("finish")

    # det
    # engine_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/tensorrt/yolov3.plan"
    # engine_path = "/mnt/data/bihua/data/model_repository/yolov3_tensorflow/det_workpieces/1/model.plan"
    # engine_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210105-1/tensorrt-20210218/model.plan"
    # engine_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210105-1/tensorrt/model.plan"
    engine_path = "/mnt/data/bihua/data/model/yolo3Tensorflow/test/model.plan"
    engine = loadEngine2TensorRT(engine_path)

    data_dict = dict()
    # path = f"/mnt/data/bihua/LocalProject/yolo3Tensorflow/camera1_2020-12-23_07_53_02_441848.jpg"
    path = f"/mnt/data/zhouzhubin/test_img/camera1_2021-03-31_08_36_56_459513.jpg"
    img_ori = cv2.imread(path)
    # # img = cv2.resize(img_ori, (768, 768))
    # img = cv2.resize(img_ori, (3008, 2016))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # img = np.asarray(img, np.float32)
    # img = np.asarray(img, np.int8)
    # img = img[np.newaxis, :] / 255.

    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (3008, 2016))
    img = torch_Normalized_gray(img)



    while True:
        start = time.time()
        output = det_do_inference(engine, img)
        print(f"time loss:{time.time()-start}")
    # # data_dict["pred_boxes"] = output[2].reshape(1, 145152).tolist()
    # data_dict["logits_"] = output[1].reshape(1, 5).tolist()  # (1, 5)
    # data_dict["center_feature_"] = output[0].reshape(1, 128).tolist()  # (1, 128)
    # with open("./data/datadict_cls_trt.json", "w", encoding="utf-8") as file:
    #     json.dump(data_dict, file)
    # print("finish")




