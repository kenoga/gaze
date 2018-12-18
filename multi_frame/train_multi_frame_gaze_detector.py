# -*- coding: utf-8 -*-

import os
import copy
import random
import datetime
import pickle
import chainer
import chainer.cuda
import cupy as cp
import chainer.functions as F
from chainer import optimizers, serializers
import numpy as np
from dataset_loader import DatasetsIteratorForCrossValidation
from network.lstm import LSTM, GRU

def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)
    # set Chainer(CuPy) random seed
    cp.random.seed(seed)


# モデルを更新する。
def update_model(model, optimizer, xs, ts):

    # 誤差逆伝播
    model.cleargrads()
    loss.backward()

    # バッチ単位で古い記憶を削除し、計算コストを削減する。
    loss.unchain_backward()

    # バッチ単位で更新する。
    optimizer.update()
    return loss

def train(model, optimizer, train_datasets):
    losses = []
    for train_dataset in train_datasets:
        batch_losses = []
        for batch in train_dataset:
            xs, ts = batch
            with chainer.using_config('train', True):
                loss = model.compute_loss(xs, ts)
            batch_losses.append(loss.data)
            # 誤差逆伝播
            model.cleargrads()
            loss.backward()
            # バッチ単位で古い記憶を削除し、計算コストを削減する。
            loss.unchain_backward()
            # バッチ単位で更新する。
            optimizer.update()
        losses.append(sum(batch_losses)/len(batch_losses))
        model.reset_state()
    return sum(losses)/len(losses)

def validate(model, datasets):
    losses = []
    for dataset in datasets:
        batch_losses = []
        for batch in dataset:
            xs, ts = batch
            with chainer.using_config('train', False):
                loss = model.compute_loss(xs, ts)
            batch_losses.append(loss.data)
        losses.append(sum(batch_losses)/len(batch_losses))
        model.reset_state()
    return sum(losses)/len(losses)

def test(model, dataset):
    ys_all = None
    ts_all = None
    losses = []
    for batch in dataset:
        xs, ts = batch
        with chainer.using_config('train', False):
            ys = model(xs)
        concat_ys = F.concat(ys, axis=0)
        concat_ts = F.concat(ts, axis=0)
        if ys_all is None:
            ys_all = concat_ys
        else:
            ys_all = F.vstack((ys_all, concat_ys))
            
        if ts_all is None:
            ts_all = concat_ts
        else:
            ts_all = F.hstack((ts_all, concat_ts))
    f1_score = F.f1_score(ys_all, ts_all)[0][1].data
    return f1_score

# 各conf + validation datsetのidから一意の実験idを作成する
# 結果のファイル名に利用する
def get_exp_id(conf, val_dialog_id):
    if conf["network"] == LSTM:
        network = "lstm"
    elif conf["network"] == GRU:
        network = "gru"
    else:
        network = "unknown"
        
    return "%s_%s_%04d_%02d_%02d_%02d" % (network, conf["input_type"], conf["rnn_hidden"], conf["batch_size"], conf["window_size"], val_dialog_id)


def main(conf):
    if conf["gpu"] >= 0:
        xp = chainer.cuda.cupy
    else:
        xp = np
    
    cross_validation_dataset_loader = DatasetsIteratorForCrossValidation(conf["dataset_path"], conf["batch_size"], conf["window_size"], test_dialog_id=5, xp=xp)
    
    for train_datasets, val_datasets in cross_validation_dataset_loader:
        val_dialog_id = cross_validation_dataset_loader.current_val_dialog_id
        exp_id = get_exp_id(conf, val_dialog_id)
        print(exp_id)
              
        # Model Settings
        model = conf["network"](conf["rnn_layer"], conf["rnn_input"], conf["rnn_hidden"], conf["rnn_output"], conf["dropout"])
        if conf["gpu"] >= 0:
            chainer.cuda.get_device(conf["gpu"]).use()
            model.to_gpu()
        # Optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        # Training
        train_losses = []
        val_losses = []
        min_val_loss = None
        min_val_model = None
        for epoch in range(conf["epoch"]):
            # エポックの最初でシャッフルする。
            random.shuffle(train_datasets)
            
            train_loss = train(model, optimizer, train_datasets)
            val_loss = validate(model, val_datasets)
            print('epoch:{}, loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))
            
            if min_val_loss is None or val_loss <= min_val_loss:
                min_val_model = copy.deepcopy(model)
                min_val_loss = val_loss

            train_losses.append(train_loss)
            val_losses.append(val_loss)
              
            with open(conf["log_path"], "a") as fr:
                report = ", ".join(["EPOCH_REPORT", exp_id, "%.4f" % val_loss])
                print(report)
                fr.write(report + "\n")
            
        # 全エポックでもっともval_lossが低くなったモデルのfスコアを計算してモデルを保存
        model_path = os.path.join(conf["result_path"], "%s.npz" % exp_id)
        serializers.save_npz(model_path, min_val_model) 
        
        train_loss_path = os.path.join(conf["result_path"], './train_loss_%s.pickle' % exp_id)
        pickle.dump(train_losses, open(train_loss_path, 'wb'))
        
        val_loss_path =  os.path.join(conf["result_path"], './val_loss_%s.pickle' % exp_id)
        pickle.dump(val_losses, open(val_loss_path, 'wb'))
        
        f1_scores = [test(model, val_dataset) for val_dataset in val_datasets]
        ave_score = sum(f1_scores) / len(f1_scores)
        with open(conf["log_path"], "a") as fr:
            report = ", ".join(["VALIDATION_REPORT", exp_id, "%.4f" % min_val_loss, "%.4f" % ave_score])
            print(report)
            fr.write(report + "\n")
        

if __name__ == '__main__':
    conf = {}
    conf["rnn_layer"] = 1
    conf["rnn_output"] = 2
    conf["dropout"] = 0.5
    conf["gpu"] = 0
    conf["epoch"] = 1
    conf["result_path"] = "./result/"
    conf["log_path"] = "./log/train_log_{0:%Y%m%d%H%M%S}.txt".format(datetime.datetime.now())
    
#     networks = [LSTM]
#     dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2")]
#     batch_sizes = [8]
#     window_sizes = [16]
#     rnn_hiddens = [32]
    
    set_random_seed(1107)

    networks = [LSTM, GRU]
    batch_sizes = [1,2,4,8,16]
    dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2"), ("./dataset/dataset_fc3.pickle", 2, "fc3")]
    window_sizes = [2,4,8,16]
    rnn_hiddens = [16,32,64,128,256]
    
    params = [networks, batch_sizes, dataset_path_and_rnn_inputs, window_sizes, rnn_hiddens]
    exp_size = reduce(lambda x, y: x * y, [len(param) for param in params])
    exp_i = 0
    for network in networks:
        for dataset_path_and_rnn_input in dataset_path_and_rnn_inputs:
            for batch_size in batch_sizes:
                for window_size in window_sizes:
                    for rnn_hidden in rnn_hiddens:
                        exp_i += 1
                        print("exp_i: %d / %d" % (exp_i, exp_size))
                        conf["network"] = network
                        conf["dataset_path"] = dataset_path_and_rnn_input[0]
                        conf["rnn_input"] = dataset_path_and_rnn_input[1]
                        conf["input_type"] = dataset_path_and_rnn_input[2]
                        conf["batch_size"] = batch_size
                        conf["window_size"] = window_size
                        conf["rnn_hidden"] = rnn_hidden
                        main(conf)
