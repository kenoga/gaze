import numpy as np

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
        self.content['test']['loss'] = []
        self.content['test']['accuracy'] = []
        self.content['test']['precision'] = []
        self.content['test']['recall'] = []
        self.content['test']['fscore'] = []

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

    def print_header(self):
        print('epoch  train_loss  train_accuracy  val_loss  val_accuracy  val_precision  val_recall  val_fscore  updated  Elapsed-Time')

    def print_tmp_result(self, epoch_i, tr_loss, tr_acc, val_loss, val_acc, val_pre, val_rec, val_fscore, updated, time):
        print('{:>5}  {:^10.4f}  {:^14.4f}  {:^8.4f}  {:^12.4f}  {:^13.4f}  {:^10.4f}  {:^10.4f}  {:^7s}  {:^12.2f}' \
              .format( \
                epoch_i, \
                np.mean(tr_loss), \
                np.mean(tr_acc), \
                np.mean(val_loss), \
                np.mean(val_acc), \
                np.mean(val_pre),
                np.mean(val_rec),
                np.mean(val_fscore),
                str(updated),
                time))
