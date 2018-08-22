import os

from Data import Data

class DataInitiator(object):
    @staticmethod
    def path2omni(path):
        data = Data()
        data.path = path
        data.name = os.path.basename(path)
        split = data.name.split('.')[0].split('_')
        data.pid = int(split[0])
        data.glasses = bool(split[1])
        data.place = split[2]
        data.target = int(split[3])
        return data

    @staticmethod
    def path2katayama(path):
        data = Data()
        data.path = path
        data.name = os.path.basename(path)
        split = data.name.split('.')[0].split('_')
        data.pid = int(split[0])
        data.target = int(split[1])
        return data
