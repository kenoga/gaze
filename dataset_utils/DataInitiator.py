import os

from Data import Data

class DataInitiator(object):
    def init(self, path):
        pass
 
class OmniDataInitiator(DataInitiator):
     def init(self, path):
        data = Data()
        data.path = path
        data.name = os.path.basename(path)
        split = data.name.split('.')[0].split('_')
        data.pid = int(split[0])
        data.glasses = bool(int(split[1]))
        data.place = split[2]
        data.target = int(split[3])
        data.id = int(split[4])
        return data

class KatayamaDataInitiator(DataInitiator):
    def init(self, path):
        data = Data()
        data.path = path
        data.name = os.path.basename(path)
        split = data.name.split('.')[0].split('_')
        data.pid = int(split[0])
        data.target = int(split[1])
        data.id = int(split[2].replace("image", ""))
        return data

  
