from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from typing import Any, Dict, Text
import six
import yaml
import os


# copy from  https://github.com/google/automl/tree/master/efficientdet
class Config(object):
    """A config utility class."""

    def __init__(self, config_dict=None):
        self.update(config_dict)

    def __setattr__(self, k, v):
        try:
            self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)
        except:
            self.__dict__[k] = Config(v) if isinstance(v, dict) else v
            print("this key can not copy, so just assign, key name:%s"%k)

    def __getattr__(self, k):
        return self.__dict__[k]

    def __getitem__(self, k):
        return self.__dict__[k]

    def __repr__(self):
        return repr(self.as_dict())

    def __str__(self):
        try:
            return yaml.dump(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())

    def _update(self, config_dict, allow_new_keys=True):
        """Recursively update internal members."""
        if not config_dict:
            return

        for k, v in six.iteritems(config_dict):
            if k not in self.__dict__:
                if allow_new_keys:
                    self.__setattr__(k, v)
                else:
                    raise KeyError('Key `{}` does not exist for overriding. '.format(k))
            else:
                # if isinstance(self.__dict__[k], dict):
                if isinstance(self.__dict__[k], type(self)):
                    self.__dict__[k]._update(v, allow_new_keys)
                else:
                    self.__dict__[k] = copy.deepcopy(v)

    def get(self, k, default_value=None):
        return self.__dict__.get(k, default_value)

    def update(self, config_dict):
        """Update members while allowing new keys."""
        self._update(config_dict, allow_new_keys=True)

    def keys(self):
        return self.__dict__.keys()

    def override(self, config_dict_or_str, allow_new_keys=False):
        """Update members while disallowing new keys."""
        if isinstance(config_dict_or_str, str):
            if not config_dict_or_str:
                return
            elif config_dict_or_str.endswith('.yaml'):
                config_dict = self.parse_from_yaml(config_dict_or_str)
            else:
                raise ValueError(
                    'Invalid string {}, must end with .yaml or contains "=".'.format(
                        config_dict_or_str))
        elif isinstance(config_dict_or_str, dict):
            config_dict = config_dict_or_str
        else:
            raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

        self._update(config_dict, allow_new_keys=allow_new_keys)

    def parse_from_module(self, module_name: Text) -> Dict[Any, Any]:
        """Import config from module_name containing key=value pairs."""
        config_dict = {}
        module = __import__(module_name)

        for attr in dir(module):
            # skip built-ins and private attributes
            if not attr.startswith('_'):
                config_dict[attr] = getattr(module, attr)

        return config_dict

    def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
        """Parses a yaml file and returns a dictionary."""
        # with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
        #     config_dict = yaml.load(f, Loader=yaml.FullLoader)
        #     return config_dict
        with open(yaml_file_path, 'r', encoding='UTF-8') as f:
            config_content = yaml.safe_load(f)

        return config_content

    def save(self, filePath):
        eval(f"self.save_to_{filePath.split('.')[-1]}(filePath)")

    def save_to_yaml(self, yaml_file_path):
        """Write a dictionary into a yaml file."""
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.as_dict(), f)

    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            else:
                config_dict[k] = copy.deepcopy(v)
        return config_dict


def get_config(config_path):
    assert os.path.exists(config_path), "config path is not exist:%s"%config_path
    config = Config()
    config.override(config_path, allow_new_keys=True)
    
    return config

