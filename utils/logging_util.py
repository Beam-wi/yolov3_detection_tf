#coding=utf-8
import logging
import os
import datetime

class Logger:
    def __init__(self, log_save_dir, task_name, save_log_num=7):
        self.log_save_dir = log_save_dir
        self.save_log_num = save_log_num
        self.build_logger(task_name)
        self.logger_id = 0
    def build_logger(self, task_name):
        if not os.path.exists(self.log_save_dir):
            os.mkdir(self.log_save_dir)
        else:
            pass
        # build_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        build_time = datetime.datetime.now().strftime('%Y%m%d')
        log_save_file_name = os.path.join(self.log_save_dir, build_time + "_%s_aiInfo.log"%task_name)
        self.logger = logging.getLogger("aiInfer")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(filename)s\t[line:%(lineno)d] %(levelname)s %(message)s')
        #创建文件日志
        file_handler = logging.FileHandler(log_save_file_name,encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        #创建控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def find_log_file(self, parent_dir):
        log_files = []
        def find_all_file_(parent_dir):
            if os.path.isdir(parent_dir):
                dir_list = os.listdir(parent_dir)
                paths = [os.path.join(parent_dir, tt) for tt in dir_list]
                for path in paths:
                    find_all_file_(path)
            else:
                if parent_dir.split("_")[-1] == "aiInfo.log":
                    log_files.append(parent_dir)
                else:
                    pass
        find_all_file_(parent_dir)
        return log_files


    def clean_log_dir(self):
        log_files = sorted(self.find_log_file(self.log_save_dir))
        delete_logs = []
        if len(log_files) > self.save_log_num:
            delete_logs = log_files[:-self.save_log_num]
        for delete_log in delete_logs:
            os.remove(delete_log)
            self.info("delete log:{0}".format(delete_log))

    def info_ai(self, meg=None, api_name=None, get_ins=None, get_outs=None, *args, **kwargs):
        if api_name == "callAiModel":
            self.logger_id += 1
        meg_add = "\nlogger id:{0}".format(str(self.logger_id))
        if meg:
            meg_add += "\nmessage:{0}".format(str(meg))
        if api_name:
            meg_add += "\napi name:{0}".format(str(api_name))
        if get_ins:
            for key, value in get_ins.items():
                meg_add += "\n......get in parameter {0}:\n{1}".format(key, str(value))
        if get_outs:
            for key, value in get_outs.items():
                meg_add += "\n......get out parameter {0}:\n{1}".format(key, str(value))
        # meg_add = meg_add.encode('utf-8')
        self.info(meg_add)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    def log(self, level, *args, **kwargs):
        self.logger.log(level, *args, **kwargs)
    # def basicConfig(self, **kwargs):
    #     self.logger.basicConfig(**kwargs)



if __name__ == '__main__':
    import numpy as np
    def add(a, b):
        return a + b
    logger_test = Logger("E:/YOLOv4_tensorflow/log")

    a = np.array([[1,2], [3,4]])
    b = np.array([[11,21], [31,41]])
    c = add(a, b)
    logger_test.info_ai(meg="call add", api_name="add", get_ins={"a":a, "b":b})
    logger_test.clean_log_dir()