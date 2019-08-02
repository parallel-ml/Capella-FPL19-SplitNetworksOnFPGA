"""
Simple example showing how SplitResNet can be trained.

Uses FashionMNIST dataset for training.
"""

import time

import mxnet as mx
from mxnet import gluon, nd, init, autograd
from mxnet.gluon.data.vision import datasets, transforms

from splitnet import SplitResNet

mnist_train = datasets.FashionMNIST(train=True)

# convert images to (channel, height, width) format
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])

mnist_train = mnist_train.transform_first(transformer)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)

net = SplitResNet(10)
net.initialize(init=mx.init.Xavier())
net.hybridize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

for epoch in range(10):
    print(f'epoch : {epoch}')
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()

    for data, label in train_data:
        data_cpu = nd.empty(data.shape, mx.context.cpu(0))
        data.copyto(data_cpu)
        data = data_cpu
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)

        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    # calculate validation accuracy
    for data, label in valid_data:
        data_cpu = nd.empty(data.shape, mx.context.cpu(0))
        data.copyto(data_cpu)
        data = data_cpu
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data),
            valid_acc/len(valid_data), time.time()-tic))
    net.save_parameters_split(epoch)

