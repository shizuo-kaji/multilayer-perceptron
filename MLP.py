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
        initializer = chainer.initializers.HeNormal()
        self.activ=activ[args.activation]
        self.dropout_ratio = args.dropout_ratio
        self.unit = args.unit
        self.batchnorm = args.batchnorm
        self.regression = args.regression
        for i in range(len(self.unit)):
            self.add_link('layer{}'.format(i), L.Linear(None,args.unit[i],initialW=initializer))
            if args.batchnorm:
                self.add_link('bnlayer{}'.format(i), L.BatchNormalization(args.unit[i]))

    def __call__(self, x, t):
        h = x
        for i in range(len(self.unit)-1):
            h = self['layer{}'.format(i)](h)
            if self.batchnorm:
                h = self['bnlayer{}'.format(i)](h)
            h = self.activ(h)
            h = F.dropout(h,ratio=self.dropout_ratio)
        h = self['layer{}'.format(len(self.unit)-1)](h)
        if self.regression:
            loss = F.mean_squared_error(t, h)
            chainer.report({'loss': loss}, self)
        else:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
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
    parser.add_argument('--labelcol', '-l', type=int, nargs="*", default=[0,1,2,3],
                        help='column indices of target variables')
    parser.add_argument('--initmodel', '-i',
                        help='Initialize the model from given file')
    parser.add_argument('--gpu', '-g', type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op', default='MomentumSGD',
                        help='optimizer {MomentumSGD,AdaDelta,AdaGrad,Adam,RMSprop}')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--skip_rows', '-sr', type=int, default=0,
                        help='num of rows skipped in the data')
    parser.add_argument('--skip_column', '-sc', type=int, nargs="*", default=[],
                        help='set of indices of columns to be skipped in the data')
    parser.add_argument('--unit', '-nu', type=int, nargs="*", default=[128,64,32,4],
                        help='Number of units in the hidden layers')
    parser.add_argument('--test_every', '-t', type=int, default=5,
                        help='use one in every ? entries in the dataset for validation')
    parser.add_argument('--regression', action='store_true',
                        help="set for regression, otherwise classification")
    parser.add_argument('--batchnorm','-bn', action='store_true',
                        help="perform batchnormalization")
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-5,
                        help='weight decay for regularization')
    args = parser.parse_args()

    ##
    if not args.gpu:
        if chainer.cuda.available:
            args.gpu = 0
        else:
            args.gpu = -1          

    print('GPU: {} Minibatch-size: {} # epoch: {}'.format(args.gpu,args.batchsize,args.epoch))

    # Set up a neural network to train
    model = MLP(args)
    if args.initmodel:
        print('Load model from: ', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Set up an optimizer
    if args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    elif args.optimizer == 'AdaDelta':
        optimizer = chainer.optimizers.AdaDelta(rho=0.95, eps=1e-06)
    elif args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad(lr=0.01, eps=1e-08)
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-08)
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop(lr=0.01, alpha=0.99, eps=1e-08)
    else:
        print("Wrong optimiser")
        exit(-1)
    optimizer.setup(model)
    if args.weight_decay>0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    print('units: {}, optimiser: {}, Weight decay: {}, dropout ratio: {}'.format(args.unit,args.optimizer,args.weight_decay,args.dropout_ratio))

    # select numpy or cupy
    xp = chainer.cuda.cupy if args.gpu >= 0 else np
    label_type = np.float32 if args.regression else np.int32

    # read csv file
    csvdata = np.loadtxt(args.dataset, delimiter=",", skiprows=args.skip_rows)
    ind = np.ones(csvdata.shape[1], dtype=bool)  # indices for unused columns
    ind[args.labelcol] = False
    for i in args.skip_column:
        ind[i] = False
    x = np.array(csvdata[:,ind],dtype=np.float32)
    t = csvdata[:,args.labelcol]
    t = np.array(t, dtype=label_type)
    if not args.regression:
        t = t[:,0]
    print('target column: {}, excluded columns: {}'.format(args.labelcol,np.where(ind==False)[0].tolist()))
    print("variable shape: {}, label shape: {}, label type: {}".format(x.shape, t.shape, label_type))

## train-validation data
# random spliting
    #train, test = datasets.split_dataset_random(datasets.TupleDataset(x, t), int(0.8*t.size))
# splitting by modulus of index
    train_idx = [i for i in range(len(t)) if (i+1) % args.test_every != 0]
    var_idx = [i for i in range(len(t)) if (i+1) % args.test_every == 0]
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

    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(
            model, 'model_epoch_{.updater.epoch}'), trigger=(frequency/5, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.dump_graph('main/loss'))

    if args.optimizer in ['MomentumSGD','AdaGrad','RMSprop']:
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(args.epoch/5, 'epoch'))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'main/accuracy', 'validation/main/loss',
         'validation/main/accuracy', 'elapsed_time', 'lr'
         ]),trigger=log_interval)

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
    x, t = chainer.dataset.concat_examples(test, args.gpu)

    with chainer.using_config('train', False):
        y = model(x,t)
    if args.gpu >= 0:
        pred = chainer.cuda.to_cpu(y.data)
        t = chainer.cuda.to_cpu(t)
    else:
        pred = y.data
    if args.regression:
        left = np.arange(t.shape[0])
        for i in range(len(args.labelcol)):
            rmse = F.mean_squared_error(pred[:,i],t[:,i])
            plt.plot(left, t[:,i], color="royalblue")
            plt.plot(left, pred[:,i], color="crimson", linestyle="dashed")
            plt.title("RMSE: {}".format(np.sqrt(rmse.data)))
            plt.savefig(args.outdir+'/result{}.png'.format(i))
            plt.close()
        result = np.hstack((t,pred))
        np.savetxt(args.outdir+"/result.csv", result , fmt='%1.5f', delimiter=",", header="truth,prediction")
    else:
        p=np.argmax(pred,axis=1)
        result = np.vstack((t,p)).astype(np.int32).transpose()
        print(result.tolist())
        np.savetxt(args.outdir+"/result.csv", result, delimiter="," ,header="truth,prediction")

if __name__ == '__main__':
    main()
