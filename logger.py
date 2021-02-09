import logging
from os import makedirs
from os.path import isfile, dirname


class Logger(object):
    def __init__(self, file_name):
        if isfile(file_name):
            logging.basicConfig(filename=file_name, filemode='a',
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                level=logging.INFO)
        else:
            dir_name = dirname(file_name)
            makedirs(dir_name, exist_ok=True)
            makedirs(dir_name + "/models/", exist_ok=True)
            makedirs(dir_name + "/Tensorboard/", exist_ok=True)
            logging.basicConfig(filename=file_name, filemode='w',
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                level=logging.INFO)
            logging.info("============ created logger =============")

    def log(self, msg):
        logging.info(msg)
