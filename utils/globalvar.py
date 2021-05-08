
class globalvar:
    logger = None
    config = None
    background_add_lines = None
def set_logger(logger):
    globalvar.logger = logger

def set_config(config):
    globalvar.config = config

def set_background_add_lines(background_add_lines):
    globalvar.background_add_lines = background_add_lines

def get_logger():
    return globalvar.logger

def get_config():
    return globalvar.config

def get_background_add_lines():
    return globalvar.background_add_lines