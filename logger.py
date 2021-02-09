import logging


class Logger(object):
    def __init__(self, file_name):
        logging.basicConfig(filename=file_name, filemode='a',
                            format='%(astime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    def log(self, msg):
        logging.info(msg)
