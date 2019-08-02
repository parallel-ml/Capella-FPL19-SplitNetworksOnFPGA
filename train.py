"""
Trains split ResNet using ImageNet dataset.
"""
import argparse, time

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory

from splitnet import SplitResNet

# number of GPUs to use
num_gpus = 3
ctx = [mx.gpu(i) for i in range(num_gpus)]

net = SplitResNet(1000)
net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

jitter_param = 0.4
lighting_param = 0.1
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]

# data augmentation
train_data = mx.io.ImageRecordIter(
    path_imgrec         = '/data/youn/datasets/imagenet/rec/train.rec',
    path_imgidx         = '/data/youn/datasets/imagenet/rec/train.idx',
    preprocess_threads  = 32,
    shuffle             = True,
    batch_size          = 256,

    data_shape          = (3, 224, 224),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
    rand_mirror         = True,
    random_resized_crop = True,
    max_aspect_ratio    = 4. / 3.,
    min_aspect_ratio    = 3. / 4.,
    max_random_area     = 1,
    min_random_area     = 0.08,
    brightness          = jitter_param,
    saturation          = jitter_param,
    contrast            = jitter_param,
    pca_noise           = lighting_param,
)

val_data = mx.io.ImageRecordIter(
    path_imgrec         = '/data/youn/datasets/imagenet/rec/val.rec',
    path_imgidx         = '/data/youn/datasets/imagenet/rec/val.rec',
    preprocess_threads  = 32,
    shuffle             = False,
    batch_size          = 256,

    resize              = 256,
    data_shape          = (3, 224, 224),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
)

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [30, 60, 90, np.inf]

# Nesterov accelerated gradient descent
optimizer = 'nag'
# Set parameters
optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
                                 'validation-top1-err', 'validation-top5-err'])

def test(ctx, val_data):
    acc_top1_val = mx.metric.Accuracy()
    acc_top5_val = mx.metric.TopKAccuracy(5)
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        acc_top1_val.update(label, outputs)
        acc_top5_val.update(label, outputs)

    _, top1 = acc_top1_val.get()
    _, top5 = acc_top5_val.get()
    return (1 - top1, 1 - top5)

epochs = 120
lr_decay_count = 0
log_interval = 50

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    acc_top1.reset()
    acc_top5.reset()

    if lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        with ag.record():
            outputs = [net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        ag.backward(loss)
        trainer.step(batch_size)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        if log_interval and not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            err_top1, err_top5 = (1-top1, 1-top5)
            print('Epoch[%d] Batch [%d]     Speed: %f samples/sec   top1-err=%f     top5-err=%f'%(
                      epoch, i, batch_size*opt.log_interval/(time.time()-btic), err_top1, err_top5))
            btic = time.time()

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    err_top1, err_top5 = (1-top1, 1-top5)

    err_top1_val, err_top5_val = test(ctx, val_data)
    train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])

    print('[Epoch %d] training: err-top1=%f err-top5=%f'%(epoch, err_top1, err_top5))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    print('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))