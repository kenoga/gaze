import os

class DataInitiator(object):
    @staticmethod
    def path2omni(path):
        self.path = path
        self.name = os.path.basename(path)
        split = self.name.split('.')[0].split('_')
        self.pid = int(split[0])
        self.glasses = bool(split[1])
        self.place = split[2]
        self.target = int(split[3])

    @staticmethod
    def path2katayama(path):
        self.path = path
        self.name = os.path.basename(path)
        split = self.name.split('.')[0].split('_')
        self.pid = int(split[0])
        self.target = int(split[1])
