import os

class ImagePath():
    def __init__(self, path):
       # format: 
        self.img_name = os.path.basename(path)
        split = self.img_name.split('.')[0].split('_')
        self.path = path
        self.pid = int(split[0])
        self.glasses = bool(split[1])
        self.place = split[2]
        self.target = int(split[3])
        self.locked = None
        self.mirror = False
        self.for_test = False
        
class FujikawaImagePath():
    def __init__(self, path):
       # format: 
        self.img_name = os.path.basename(path)
        self.path = path
        self.locked = None
        self.for_test = False
        self.mirror = False