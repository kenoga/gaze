# -*- coding: utf-8 -*-
import os, sys
import pickle
import chainer

sys.path.append(os.path.abspath(".."))

from training.data_loader.dataset_loader import DatasetLoader
from evaluation.conf_repr_translator import ConfigRepresentationTranslator


class Tester(object):
    def __init__(self):
        self.test_did = 5
        self.translator = ConfigRepresentationTranslator()

    def test(self, npz_path, output_dir):
        repr = os.path.basename(npz_path).split(".")[0]
        #### Load Setting
        Network, network_params, Iterator, iterator_params, Tester, dataset_path = self.translator.repr2setting(repr)

        #### Iterator Setting
        Iterator.set_params(**iterator_params)

        #### Tester Setting
        tester = Tester(Network, network_params)
        tester._setup()
        tester.load_npz(npz_path)

        #### DatasetLoader Setting
        dataset_loader = DatasetLoader(dataset_path, Iterator)

        #### Loading Test Datasets
        test_datasets = dataset_loader.load_by_dialog_id(self.test_did)
        
        #### Make Output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Test
        for i, test_dataset in enumerate(test_datasets):
            print(test_dataset)
            with chainer.using_config('train', False):
                score, (ts_all, ys_all) = tester.test(test_dataset, True)
            print(score)
            ys_all = chainer.functions.softmax(ys_all)
            with open(os.path.join(output_dir, "%02d_%02d_%s.pickle" % (test_dataset.dialog_id, test_dataset.session_id, test_dataset.seat_id)), "w") as fw:
                pickle.dump((chainer.cuda.to_cpu(ts_all.data), chainer.cuda.to_cpu(ys_all.data)), fw)

if __name__ == "__main__":
    npz_dir = os.path.join(".", "training", "output", "single_frame_test_dialog_id_05", "npz")
    npz_path = os.path.join(npz_dir, "ff1_fc2_0064_03_02.npz")
    output_dir = os.path.join("evaluation", "test_result", "test")
    tester = Tester().test(npz_path, output_dir)
