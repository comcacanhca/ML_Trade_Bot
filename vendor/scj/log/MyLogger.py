import datetime
import logging
import os

today = datetime.datetime.now().strftime('%Y-%m-%d')
log_file_name = os.getcwd().split('SCJ999')[0]+f'SCJ999/log/{today}/'

if os.path.exists(log_file_name)==False:
    os.makedirs(name=log_file_name)

file_name = 'MyLogger'
class MyLogger:
    logger_instance = None
    loggerFileHandle = None
    StreamHandle = None
    nowLevel = logging.INFO

    @classmethod
    def setLogger(cls, filename, level = nowLevel):
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        if level==logging.INFO:
            print(f'LogPath: {log_file_name + filename + "_info.log"}')
            cls.loggerFileHandle = logging.FileHandler(log_file_name + filename + '_info.log', mode='a', encoding='utf-8')
        else:
            cls.loggerFileHandle = logging.FileHandler(log_file_name+filename+"_debug.log", mode='a', encoding='utf-8')

        # debug_log.setLevel(logging.DEBUG)
        # info_log.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(levelname)s: %(asctime)s - %(message)s', datefmt='%H:%M:%S')

        # Set the formatter for the file handler
        cls.loggerFileHandle.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(cls.loggerFileHandle)
        return logger

    @classmethod
    def set_Level(cls, level):
        MyLogger.get_logger().set_Level(level)

    @classmethod
    def get_logger(cls, filename=None):
        if not cls.logger_instance:
            if filename:
                cls.logger_instance = MyLogger.setLogger(filename)
            else:
                cls.logger_instance = MyLogger.setLogger(file_name)
        return cls.logger_instance

def write_test(text, show=True):
    text = f"{text}"
    with open(log_file_name+'/write_file.txt', mode='a') as f:
        f.writelines(f'{datetime.datetime.now()}: '+text+'\n')
        if show==True:
            print(f'{datetime.datetime.now()}: '+text)

class MyFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(log_file_name+filename, mode, encoding, delay)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg+'\n')
            stream.flush()
        except (Exception,) as e:
            self.handleError(record)
            raise e