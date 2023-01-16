import os
import logging
import yaml
from fastnumbers import fast_real
from configparser import ConfigParser
from pathlib import Path

# Conventional constants
LOG_DIR = 'logs'
CONFIG_DIR = 'config'
CLASSIFICATION_KEY = 'classification'
REGRESSION_KEY = 'regression'
EPSILON = 1e-16


# Standard messages
def load_messages(messages_section='Messages'):
    messages_fpath = os.path.join(os.path.join(Path(__file__).parent, 'config'), 'messages.properties')
    config = ConfigParser()
    config.read(messages_fpath)
    return dict(config[messages_section])


# Configuration
def load_user_config_if_exists(app_path):
    user_config_filepath = None
    config_dir_path = os.path.join(app_path, CONFIG_DIR)
    if os.path.exists(config_dir_path):
        for filename in os.listdir(config_dir_path):
            if filename.endswith(".yaml") and ('settings' in filename or 'config' in filename):
                user_config_filepath = os.path.join(config_dir_path, filename)
                break
    return user_config_filepath


class Config:

    def __init__(self, user_config_fpath=None, standard_config_fpath=None):
        if standard_config_fpath == None:
            standard_config_fpath = os.path.join(os.path.join(Path(__file__).parent, 'config'), 'settings.yaml')
        standard_config_dict = yaml.load(open(standard_config_fpath, 'r'), Loader=yaml.SafeLoader)
        if user_config_fpath != None:
            user_config_dict = yaml.load(open(user_config_fpath, 'r'), Loader=yaml.SafeLoader)
            Config.override_dict(standard_config_dict, user_config_dict)
        else:
            Config.override_dict(standard_config_dict)
        self.full_conf = standard_config_dict

    @staticmethod
    def override_dict(standard_dict, user_dict=None):
        for key, value in standard_dict.items():
            if isinstance(value, dict):
                if user_dict is not None and key in user_dict:
                    Config.override_dict(value, user_dict[key])
                else:
                    Config.override_dict(value)
            else:
                if user_dict is not None and key in user_dict:
                    standard_dict[key] = user_dict[key]



# Logging
def build_log_file_path(app_path, app_name):
    log_dir_path = os.path.join(app_path, LOG_DIR)
    if not (os.path.exists(log_dir_path)):
        os.makedirs(log_dir_path)
    return os.path.join(log_dir_path, f'{app_name}.log')


def file_logger(file_path, app_name):
    logger = logging.getLogger(f'TheDeepChecker: {app_name} Logs')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def console_logger(app_name):
    logger = logging.getLogger(f'TheDeepChecker: {app_name} Logs')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
