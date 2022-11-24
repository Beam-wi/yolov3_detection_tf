# -*- coding:utf-8 -*-

import os
import re
import sys
import logging

from logging.handlers import TimedRotatingFileHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mvsGetPhotoArr.utils import cfgs


def get_logger(log_name):  # ,log_save_file="algorithm_model"
    if not os.path.exists(cfgs.LOG_DIR):
        os.mkdir(cfgs.LOG_DIR)
    log_save_file = os.path.join(cfgs.LOG_DIR, log_name)
    # if not (os.path.exists(log_save_file)):  # 检查文件是否已经存在
    #     fo = open(log_save_file, "w")
    #     fo.close()
    # fsize = os.path.getsize(log_save_file)
    # fsize = fsize / float(1024 * 1024)
    # if fsize > 4:
    #     os.remove(log_save_file)
    #     print("log file is remove log file..", log_save_file)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    # file log handler
    # fh = logging.FileHandler(log_save_file)
    fh = TimedRotatingFileHandler(
        filename=log_save_file, when="MIDNIGHT", interval=1, backupCount=7
    )
    fh.suffix = "%Y-%m-%d.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
    fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    fh.setLevel(logging.DEBUG)
    # console hander
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s %(filename)s\t[line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    # 在记录日志之后移除句柄
    # logger.removeHandler(fh)
    # logger.removeHandler(ch)
    return logger