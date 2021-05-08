import os
import sys
import argparse
import tensorflow as tf
from utils.config_parse import get_config
from utils.logging_util import Logger
import utils.globalvar as globalvar
from utils.misc_utils import read_class_names
from utils.classify_utils import test_data_from_dir, get_wrong_data

project_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    #################
    # ArgumentParser
    #################
    parser = argparse.ArgumentParser(description="YOLO-V3 training procedure.")
    parser.add_argument("--config_project", type=str, default="./data/config/config_project.yaml",
                        help="项目配置文件.")
    parser.add_argument("--config_common", type=str, default="./data/config/config_common.yaml",
                        help="默认配置文件.")
    args = parser.parse_args()

    config_common = get_config(args.config_common)
    config_project = get_config(args.config_project).as_dict()
    config_common.update(config_project)

    # 创建部分存储路径
    if not os.path.exists(config_common.data_set["log_save_path"]):
        try:
            os.makedirs(config_common.data_set["log_save_path"])
            print("make log save dir:%s" % config_common.data_set["log_save_path"])
        except Exception as ex:
            print("make log save dir:%s, failed, cause of %s" % (config_common.data_set["log_save_path"], ex))
            sys.exit(1)

    # 创建日志文件
    logger = Logger(config_common.data_set["log_save_path"], "classify_test")
    logger.clean_log_dir()

    logger.info_ai(meg="project name is:%s" % config_common.project_name)
    logger.info_ai(meg="config info", get_ins={"config": config_common})

    if not os.path.exists(config_common.data_set["test_result_save_path"]):
        try:
            os.makedirs(config_common.data_set["test_result_save_path"])
            logger.info_ai(meg="make test result data save dir:%s" % config_common.data_set["test_result_save_path"])
        except Exception as ex:
            logger.info_ai(meg="make test result data save dir:%s, failed, cause of %s" % (
            config_common.data_set["test_result_save_path"], ex))
        sys.exit(1)

    # 设置全局变量
    globalvar.set_logger(logger)
    globalvar.set_config(config_common)

    # 判断必须文件是否存在
    assert os.path.exists(config_common.data_set["class_name_path"]), "class_name_path not exit:%s" % config_common.data_set["class_name_path"]
    assert os.path.exists(config_common.data_set["log_save_path"]), "log_save_path not exit:%s" % config_common.data_set["log_save_path"]
    assert os.path.exists(config_common.data_set["classify_model_save_path"]), "classify_model_save_path not exit:%s" % config_common.data_set["classify_model_save_path"]
    assert os.path.exists(config_common.data_set["log_save_path"]), "log_save_path not exit:%s" % config_common.data_set["log_save_path"]
    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = config_common.model_set["gpu_device"]
    gpus = []
    for id in config_common.model_set["gpu_device"].strip().split(","):
        gpus.append("/gpu:%s" % id)

    gpu_device = gpus
    classes = read_class_names(config_common.data_set["class_name_path"])
    classes_list = list(classes.values())
    logger.info_ai(meg="train class name get from class name file", get_ins={"classes_list": classes_list})
    name2id = {}
    id = 0
    for name in classes_list:
        name2id[name] = id
        id = id + 1
    logger.info_ai(meg="get class name id", get_ins={"name2id": name2id})

    test_result_temp_data_path = os.path.join(config_common.data_set["test_result_save_path"], "classify_test_result")
    if not os.path.exists(test_result_temp_data_path):
        try:
            os.makedirs(test_result_temp_data_path)
            print("make test_result_temp_data_path:%s" % test_result_temp_data_path)
        except Exception as ex:
            print("make test_result_temp_data_path:%s, failed, cause of %s" % (test_result_temp_data_path, ex))
            sys.exit(1)

    test_result_wrong_data_path = os.path.join(config_common.data_set["test_result_save_path"], "classify_test_result")
    if not os.path.exists(test_result_wrong_data_path):
        try:
            os.makedirs(test_result_wrong_data_path)
            print("make test_result_wrong_data_path:%s" % test_result_wrong_data_path)
        except Exception as ex:
            print("make test_result_wrong_data_path:%s, failed, cause of %s" % (test_result_wrong_data_path, ex))
            sys.exit(1)

    save_result_path = os.path.join(test_result_temp_data_path, "%s.npy"%config_common.model_set["classify_model_save_name"])
    model_path_classify = os.path.join(config_common.data_set["classify_model_save_path"],
                                            config_common.model_set["classify_model_save_name"])
    assert os.path.exists(model_path_classify + ".meta"), "classify model not exit:%s" % model_path_classify

    test_data_from_dir(config_common.data_set["classify_val_data_save_path_temp"], model_path_classify, save_result_path, classes_list, name2id, gpu_device)

    all_num_data, right_num_data, wrong_num_data, not_find_num_data = [], [], [], []

    wrong_data_save_path = os.path.join(test_result_wrong_data_path, "%f"%config_common.model_set["classify_model_test_score"])
    get_wrong_data(save_result_path, wrong_data_save_path, classes_list, score=config_common.model_set["classify_model_test_score"], do_save=True)
