import uff
import tensorrt as trt



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

model_file = '/mnt/data/bihua/data/model/yolo3Tensorflow/uff/yolov3.uff'
engine_file_path = '/mnt/data/bihua/data/model/yolo3Tensorflow/tensorrt/yolov3.plan'
# Create the builder, network, and parser:
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser, builder.create_builder_config() as config:
    config.max_workspace_size = 1 << 20  # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    # parser.register_input("Placeholder", (1, 28, 28))
    # parser.register_output("fc2/Relu")
    parser.register_input("input_data", (1, 768, 768, 3))
    parser.register_output("boxes_result/confs_result/probs_result")
    parser.parse(model_file, network)
    # if not parser.parse(model_file, network):
    #     print('ERROR: Failed to parse the uff file.')
    #     for error in range(parser.num_errors):
    #         print(parser.get_error(error))
    # engine = builder.build_cuda_engine(network)
    engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())


