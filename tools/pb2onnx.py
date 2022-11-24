# !/usr/bin/env python3
import os
import time


"""
方法一、
import tensorflow as tf
import tf2onnx

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, '/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/pb/pb')  # ['GPU_server_1'],
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=["input_data"], output_names=["class_result", "feature_result"])
    model_proto = onnx_graph.make_model("test")
    with open("/mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/onnx/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())





方法二、
shell
python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/pb/pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/onnx/model.onnx
python3 -m tf2onnx.convert --saved-model /mnt/data/binovo/data/model/sjht-lzbl-271/cls --output /mnt/data/binovo/data/model/sjht-lzbl-271/onnx/model.onnx
也可以用脚本执行shell指令
import os
import time

a = time.time()
os.system('sudo python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/pb/pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/onnx/model.onnx')
print(time.time() - a)
"""





"""
sudo python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/pb/frozen/frozen_graph.pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/onnx/yolov3.onnx
sudo python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/pb/pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20200106/onnx/model.onnx
sudo python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210105/pb/pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/det/inserts/20210105/onnx/model.onnx
sudo python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20210111-8/pb/pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/cls/20210111-8/onnx/2/model.onnx --inputs input0:0[-1,192,192,3]
sudo python3 -m tf2onnx.convert --saved-model /mnt/data/bihua/data/model/yolo3Tensorflow/lbhy/pb --output /mnt/data/bihua/data/model/yolo3Tensorflow/lbhy/onnx/model.onnx
"""




if __name__ == "__main__":
    pb_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/pb/det/4"
    onnx_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/onnx/det/4/model.onnx"
    # pb_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/pb/cls/4"
    # onnx_path = "/mnt/data/binovo/data/model/sjht-lzbl-271/onnx/cls/4/model.onnx"
    a = time.time()
    # os.system(
    #     f'sudo python3 -m tf2onnx.convert --saved-model {pb_path} --output {onnx_path}')
    os.system(
        f'sudo python3 -m tf2onnx.convert --saved-model {pb_path} --output {onnx_path}')
    print(f"loss time: {time.time() - a}")