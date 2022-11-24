#! /usr/bin/evn/pytrhon3

import os
import sys
import time
import numpy as np

from ckpt2pb import det_ckpt2pb, cls_ckpt2pb
from onnx2rt import get_engine, EntropyCalibrator


def ckptConvPb(modelPath, imgSize, classNum, anchors=None, exportDir="/tmp/Trt"):
    START = time.time()
    pbPath = f"{exportDir}/pb"
    if len(os.listdir(pbPath)):
        os.system(f'rm -r {pbPath}/*')
    if isinstance(anchors, np.ndarray):
        # det
        det_ckpt2pb(modelPath, pbPath, classNum, anchors, img_size=imgSize,
                    use_predict=True)
    else:
        # cls
        cls_ckpt2pb(modelPath, pbPath, classNum, img_size=imgSize)
    print(f"pb convert success:\n  timeLoss: {time.time()-START}\n  "
          f"pbPath: {pbPath}")


def pbConvOnnx(exportDir="/tmp/Trt"):
    pbPath = f"{exportDir}/pb"
    onnxPath = f"{exportDir}/onnx/model.onnx"
    if os.path.exists(f"{exportDir}/onnx"):
        if len(os.listdir(f"{exportDir}/onnx")):
            os.system(f'rm -r {exportDir}/onnx/*')
    START = time.time()
    os.system(
        f'sudo python3 -m tf2onnx.convert --saved-model {pbPath} --output {onnxPath}')
    print(f"pb convert success:\n  timeLoss: {time.time()-START}\n  "
          f"onnxPath: {onnxPath}")


def onnxConvTrt(engineDir, imgSize, exportDir="/tmp/Trt", dynamicInput=False,
                trainDataFile=None, mode=None, prefix="", type='float16'):
    """
    type: 转trt计算值格式，'float16', 'int8' 或 'float32'
    mode: 'det' 或 'cls'.用int8时必须指定
    """
    onnxPath = f'{exportDir}/onnx/model.onnx'
    enginePath = f"{engineDir}/model.plan"
    if type == 'int8' and mode and trainDataFile:
        # int8
        cacheFile = f"{exportDir}/cache/calibration.cache"
        calib = EntropyCalibrator(trainDataFile, cacheFile, imgSize,
                                  batch_size=1, prefix=prefix, mode=mode)
        with get_engine(onnxPath, imgSize, enginePath, calib=calib, ndtype='int8',
                        dynamic_input=dynamicInput) as engine, engine.create_execution_context() as context:
            print("finish!!!")
    else:
        # float16
        with get_engine(onnxPath, imgSize, enginePath, ndtype='float16',
                        dynamic_input=dynamicInput) as engine, engine.create_execution_context() as context:
            print("finish!!!")


def main():
    """临时文件默认存在exportDir=/tmp/Trt下"""
    # det
    # modelPath = "/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-281/1.0/model_and_temp_file/model_detection/202107291536_detection_model_default_name"
    # engineDir = "/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-281/1.0/model_and_temp_file/release_model_path/sjht_syht1_281_v1.0/trt_model/qj_detection/1"
    # imgSize = (1, 2016, 3008, 3)
    # classNum = 1
    # anchors = np.reshape(np.asarray
    #     ([15.00, 30.00, 19.00, 19.00, 30.00, 15.00, 25.00, 50.00, 36.00, 36.00, 50.00, 25.00, 43.00, 86.00, 60.00, 60.00, 86.00, 43.00], np.float32), [-1, 2]) * 2.5

    # cls
    modelPath = "/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-281/1.0/model_and_temp_file/model_classify/202107291536_classify_model_default_name"
    engineDir = "/mnt/data/binovo/data/images4code/ai_model_ckpt/manu_train/sjht/syht1/sjht-syht1-281/1.0/model_and_temp_file/release_model_path/sjht_syht1_281_v1.0/trt_model/classfy_model/1"
    imgSize = (1, 192, 192, 3)
    classNum = 4
    anchors = None

    ckptConvPb(modelPath, imgSize, classNum, anchors=anchors)
    pbConvOnnx()
    onnxConvTrt(engineDir, imgSize)


if __name__ == '__main__':
    main()
