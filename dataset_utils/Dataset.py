
class Dataset(object):
    def __init__(self, dataset_dir, img_format="jpg"):
        self.dataset_dir = dataset_dir
        self.img_format = img_format
        self.data = self._load_data()
        self.with_face_direction = False
    
    def _load_data(self):
        subdirs = sorted([os.path.join(dataset_dir, subdir) \
                           for subdir in os.listdir(dataset_dir) \
                           if os.path.isdir(subdir)])

        for subdir in subdirs:
            datapaths = sorted([path for path in glob.glob(os.path.join(subdir, '*.%s' % format))])
            data = [Data(datapath) for datapath in datapaths]
        return data
    
    def load_face_bb_lmk(self, face_bb_lmk_dict):
        # delete data that has not face
        self.data = [d for d in data if d.name in face_bb_lmk_dict]
        for d in self.data:
            bb = face_bb_lmk_dict[d.name]['bb']
            lmk = face_bb_lmk_dict['landmarks']
            d.set_bb(bb)
            d.set_lmk(lmk)
    
    def load_face_direction_feature(self, face_dir_dict):
        # delete data that has not face
        self.with_face_direction = True
        self.data = [d for d in data if d in face_bb_lmk_dict]
        for d in self.data:
            d.face_direction = face_dir_dict[d.name]
    
    def set_label(self, locked_targets):
        for d in self.data:
            if d.target in locked_targets:
                d.locked = True
            else:
                d.locked = False

    def filter_noise(self, noise_set):
        self.data = [d for d in self.data if d not in noise_set]
    
    def filter_noise2(self, noise_dict):
        self.data = [d for d in self.data \
                    if d.name not in self.annotation_dict \
                    or noise_set[ipath.name] not in {'closed-eyes', 'other'}]
    
    def filter_pid(self, pids):
        self.data = [d for d in self.data if d.pid in pids]
    
    def filter_target(self, ignored):
        self.data = [d for d in self.data if d.target not in ignored]
    
    def filater_place(self, places):
        self.data = [d for d in self.data if d.place in places]
        
    