
class globalvar:
    logger = None
    config = None
    background_add_lines = None
    data_train = None
    data_val = None
    label_statics = None

def set_logger(logger):
    globalvar.logger = logger

def set_config(config):
    globalvar.config = config

def set_background_add_lines(background_add_lines):
    globalvar.background_add_lines = background_add_lines

def set_data_train(data_train):
    globalvar.data_train = data_train

def set_data_val(data_val):
    globalvar.logger = data_val

def set_label_statics(label_statics):
    globalvar.label_statics = label_statics



def get_logger():
    return globalvar.logger

def get_config():
    return globalvar.config

def get_background_add_lines():
    return globalvar.background_add_lines

def get_data_train():
    return globalvar.data_train

def get_data_val():
    return globalvar.data_val

def get_label_statics():
    return globalvar.label_statics