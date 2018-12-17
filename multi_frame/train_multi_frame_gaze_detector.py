# -*- coding: utf-8 -*-

import pickle
import chainer
import chainer.cuda
import chainer.functions as F
from chainer import optimizers, serializers
import numpy as np
from dataset_loader import DatasetsIteratorForCrossValidation
from networks.lstm import LSTM, GRU



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
            loss = model.compute_loss(xs, ts)
            batch_losses.append(loss.data)
            loss.unchain_backward()
        losses.append(sum(batch_losses)/len(batch_losses))
        model.reset_state()
    return sum(losses)/len(losses)

def test(model, dataset):
    ys_all = None
    ts_all = None
    losses = []
    for batch in dataset:
        xs, ts = batch
        ys = model(xs)
        concat_ys = F.concat(ys, axis=0)
        concat_ts = F.concat(ts, axis=0)
#         import pdb; pdb.set_trace()
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

def main(conf):
    if conf["gpu"] >= 0:
        xp = chainer.cuda.cupy
    else:
        xp = np
    
    cross_validation_dataset_loader = DatasetsIteratorForCrossValidation(conf["dataset_path"], conf["batch_size"], conf["window_size"], xp)
    
    for train_datasets, val_datasets, test_datasets in cross_validation_dataset_loader:
        for test_dataset in test_datasets:
            print(test_dataset)
        # Model Settings
        model = GRU(conf["lstm_layer"], conf["lstm_input"], conf["lstm_hidden"], conf["lstm_output"], conf["dropout"], train=True)
        if conf["gpu"] >= 0:
            model.to_gpu()

        # Optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        # Training
        train_losses = []
        val_losses = []
        for epoch in range(conf["epoch"]):
            # エポックの最初でシャッフルする。
            # random.shuffle(train_dataset)
            
            train_loss = train(model, optimizer, train_datasets)
            val_loss = validate(model, val_datasets)
            print('epoch:{}, loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))

            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        for test_dataset in test_datasets:
            f1_score = test(model, test_dataset)
            print("%s -> %.3f" % (test_dataset, f1_score))
                  
        # 保存する。
        serializers.save_npz('./test.npz', model)
        pickle.dump(train_losses, open('./train_loss.pickle', 'wb'))
        pickle.dump(val_losses, open('./val_loss.pickle', 'wb'))

if __name__ == '__main__':
    conf = {}
    conf["batch_size"] = 32
    conf["window_size"] = 8
    conf["lstm_layer"] = 1
    conf["lstm_input"] = 128
    conf["lstm_hidden"] = 16
    conf["lstm_output"] = 2
    conf["dropout"] = 0.5
    conf["gpu"] = 0
    conf["epoch"] = 10
    conf["dataset_path"] = "./dataset_fc2.pickle"
    main(conf)
