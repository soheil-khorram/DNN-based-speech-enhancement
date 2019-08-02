import os
import datetime


class Logger:

    @staticmethod
    def set_path(path):
        Logger.path = path
        dir_path = os.path.dirname(Logger.path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def write_log(text):
        print(text)
        with open(Logger.path, 'a') as file_id:
            file_id.write(text + '\n')

    @staticmethod
    def write_date_time():
        Logger.write_log('Date-time: ' + str(datetime.datetime.now()))
