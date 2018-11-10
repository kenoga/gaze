
class ExperimentResult():
    def __init__(self):
        self.content = {}
        self.content['train'] = {}
        self.content['val'] = {}
        self.content['test'] = {}
        self.content['train']['loss'] = []
        self.content['train']['accuracy'] = []
        self.content['val'] = {}
        self.content['val']['loss'] = []
        self.content['val']['accuracy'] = []
        self.content['val']['precision'] = []
        self.content['val']['recall'] = []
        self.content['val']['fscore'] = []
        self.content['test'] = {}
        self.content['test']['imgs'] = []
        self.content['test']['y'] = []
        self.content['test']['t'] = []
        self.content['test']['miss'] = []

    def add_train_result(self, loss, accuracy):
        self.content['train']['loss'].append(float(loss))
        self.content['train']['accuracy'].append(float(accuracy))

    def add_validation_result(self, loss, accuracy, precision, recall, fscore):
        self.content['val']['loss'].append(float(loss))
        self.content['val']['accuracy'].append(float(loss))
        self.content['val']['precision'].append(float(loss))
        self.content['val']['recall'].append(float(loss))
        self.content['val']['fscore'].append(float(loss))

    def add_test_result(self, loss, accuracy, precision, recall, fscore, t, y, paths, probs):
        self.content['test']['loss'].append(float(loss))
        self.content['test']['accuracy'].append(float(loss))
        self.content['test']['precision'].append(float(loss))
        self.content['test']['recall'].append(float(loss))
        self.content['test']['fscore'].append(float(loss))
        self.content['test']['t'] = t
        self.content['test']['y'] = y
        self.content['test']['paths'] = paths
        self.content['test']['probs'] = probs
