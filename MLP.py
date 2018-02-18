#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @brief multi-layer perceptron for classification and regression
# @section Requirements:  python3,  chainer 2
# @version 0.01
# @date Oct. 2017
# @author Shizuo KAJI (shizuo.kaji@gmail.com)
# @licence MIT

from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training,datasets,iterators
from chainer.training import extensions
import numpy as np

from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

# activation function
activ = {
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'linear': F.identity,
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
}

# Neural Network definition
class MLP(chainer.Chain):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.activ=activ[args.activation]
        self.layers = args.layers
        self.out_ch = args.out_ch
        self.dropout_ratio = args.dropout_ratio
        self.l = []
        # with self.init_scope():
        #     self.l.append(L.Linear(None,args.unit))
        #     for i in range(1,self.layers):
        #         self.l.append(L.Linear(args.unit,args.unit))
        #     self.l.append(L.Linear(args.unit,args.out_ch))            
        self.add_link('layer{}'.format(0), L.Linear(None,args.unit))
        for i in range(1,self.layers):
            self.add_link('layer{}'.format(i), L.Linear(args.unit,args.unit))
        self.add_link('fin_layer', L.Linear(args.unit,args.out_ch))

    def __call__(self, x, t):
        # h = F.dropout(self.activ(self.l[0](x)),ratio=self.dropout_ratio)
        # for i in range(1,self.layers):
        #     h = F.dropout(self.activ(self.l[i](h)),ratio=self.dropout_ratio)
        # h = self.l[-1](h)
        
        h = F.dropout(self.activ(self['layer{}'.format(0)](x)),ratio=self.dropout_ratio)
        for i in range(1,self.layers):
            h = F.dropout(self.activ(self['layer{}'.format(i)](h)),ratio=self.dropout_ratio)
        h = self['fin_layer'](h)

        if self.out_ch > 1:    # classification
            h = F.sigmoid(h)
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        else:   #regression
            loss = F.mean_squared_error(t, h)
            chainer.report({'loss': loss}, self)
        if chainer.config.train:
            return loss
        return h


def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Perceptron classifier/regressor')
    parser.add_argument('dataset', help='Path to data file')
    parser.add_argument('--activation', '-a', choices=activ.keys(), default='sigmoid',
                        help='Activation function')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--dropout_ratio', '-dr', type=float, default=0,
                        help='dropout ratio')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--snapshot', '-s', type=int, default=-1,
                        help='snapshot interval')
    parser.add_argument('--label_index', '-l', type=int, default=0,
                        help='Column number of the target variable')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--out_ch', '-oc', type=int, default=1,
                        help='num of output channels. set to 1 for regression')
    parser.add_argument('--optimizer', '-op', default='MomentumSGD',
                        help='optimizer {MomentumSGD,AdaDelta,AdaGrad,Adam,RMSprop}')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--skip_rows', '-sr', type=int, default=1,
                        help='num of rows skipped in the data')
    parser.add_argument('--skip_column', '-sc', type=int, nargs="*", default=[],
                        help='set of indices of columns to be skipped in the data')
    parser.add_argument('--layers', '-nl', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--unit', '-nu', type=int, default=100,
                        help='Number of units in the hidden layers')
    parser.add_argument('--test_every', '-t', type=int, default=5,
                        help='use one in every ? entries in the dataset for validation')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-5,
                        help='weight decay for regularization')
    args = parser.parse_args()

    ##
    print('GPU: {} Minibatch-size: {} # epoch: {}'.format(args.gpu,args.batchsize,args.epoch))

    # Set up a neural network to train
    model = MLP(args)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimiser
    if args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=0.003, momentum=0.9)
    elif args.optimizer == 'AdaDelta':
        optimizer = chainer.optimizers.AdaDelta(rho=0.95, eps=1e-06)
    elif args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad(lr=0.001, eps=1e-08)
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop(lr=0.01, alpha=0.99, eps=1e-08)
    else:
        print("Wrong optimiser")
        exit(-1)
    optimizer.setup(model)
    if args.weight_decay>0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    print('layers: {}, units: {}, optimiser: {}, Weight decay: {}, dropout ratio: {}'.format(args.layers,args.unit,args.optimizer,args.weight_decay,args.dropout_ratio))

    # select numpy or cupy
    xp = chainer.cuda.cupy if args.gpu >= 0 else np
    label_type = np.int32 if args.out_ch>1 else np.float32

    # read csv file
    csvdata = np.loadtxt(args.dataset, delimiter=",", skiprows=args.skip_rows)
    ind = np.ones(csvdata.shape[1], dtype=bool)  # indices for unused columns
    ind[args.label_index] = False
    for i in args.skip_column:
        ind[i] = False
    x = csvdata[:,ind]
    t = csvdata[:,args.label_index][:,np.newaxis]
    print('target column: {}, excluded columns: {}'.format(args.label_index,np.where(ind==False)[0].tolist()))
    print("variable shape: {}, label shape: {}, label type: {}".format(x.shape, t.shape, label_type))
    x = np.array(x, dtype=np.float32)
    if args.out_ch > 1:
        t = np.array(np.ndarray.flatten(t), dtype=label_type)
    else:
        t = np.array(t, dtype=label_type)

## train-validation data
# random spliting
    #train, test = datasets.split_dataset_random(datasets.TupleDataset(x, t), int(0.8*t.size))
# splitting by modulus of index
    train_idx = [i for i in range(t.size) if (i+1) % args.test_every != 0]
    var_idx = [i for i in range(t.size) if (i+1) % args.test_every == 0]
    n = len(train_idx)
    train_idx.extend(var_idx)
    train, test = datasets.split_dataset(datasets.TupleDataset(x, t), n, train_idx)

# dataset iterator
    train_iter = iterators.SerialIterator(train, args.batchsize, shuffle=True)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    frequency = args.epoch if args.snapshot == -1 else max(1, args.snapshot)
    log_interval = 1, 'epoch'
    val_interval = frequency/10, 'epoch'

    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=log_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # ChainerUI
    trainer.extend(CommandsExtension())
    save_args(args, args.outdir)
    trainer.extend(extensions.LogReport(trigger=log_interval))

    if not args.predict:
        trainer.run()
    else:
        test = datasets.TupleDataset(x, t)

    ## prediction
    print("predicting: {} entries...".format(len(test)))
    nvar, = test[0][0].shape
    x = xp.zeros((len(test), nvar)).astype(np.float32)
    for i in range(len(test)):
        x[i,:] = xp.asarray(test[i][0])
    if args.out_ch > 1:
        t = xp.zeros(len(test)).astype(label_type)
        for i in range(len(test)):
            t[i] = xp.asarray(test[i][1])
    else:
        t = xp.zeros((len(test), 1)).astype(label_type)
        for i in range(len(test)):
            t[i,:] = xp.asarray(test[i][1])

    with chainer.using_config('train', False):
        y = model(x,t)
    if args.gpu >= 0:
        pred = chainer.cuda.to_cpu(y.data)
    else:
        pred = y.data
    if args.out_ch > 1:    # classification
        p=np.argmax(pred,axis=1)
        result = np.vstack((t,p)).astype(np.int32).transpose()
        print(result.tolist())
        np.savetxt(args.outdir+"/nn-out.csv", result, delimiter="," ,header="truth,prediction")
    else:
        rmse = F.mean_squared_error(pred,t)
        result = np.vstack((t[:,0],pred[:,0])).transpose()
        np.savetxt(args.outdir+"/nn-out.csv", result , delimiter="," ,header="truth,prediction")
        # draw a graph
        left = np.arange(len(test))
        plt.plot(left, t[:,0], color="royalblue")
        plt.plot(left, pred[:,0], color="crimson", linestyle="dashed")
        plt.title("RMSE: {}".format(np.sqrt(rmse.data)))
        plt.show()

if __name__ == '__main__':
    main()
