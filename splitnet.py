from __future__ import absolute_import, print_function

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision

class ResNetSplitBlock(gluon.HybridBlock):
    '''
    Exactly the same architecture as
    mxnet.gluon.model_zoo.vision.resnet.BasicBlockV1,
    but cut in half.
    '''
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(ResNetSplitBlock, self).__init__(**kwargs)

        self.body = gluon.nn.HybridSequential(prefix='SplitBlock_')
        self.body.add(gluon.nn.Conv2D(channels=channels, kernel_size=(3, 3), strides=stride, padding=1,
                in_channels=in_channels, use_bias=False))
        self.body.add(gluon.nn.BatchNorm())
        self.body.add(gluon.nn.Activation('relu'))
        self.body.add(gluon.nn.Conv2D(channels=channels, kernel_size=(3, 3), strides=1, padding=1,
                in_channels=in_channels, use_bias=False))
        self.body.add(gluon.nn.BatchNorm())

        if downsample:
            self.downsample = gluon.nn.HybridSequential(prefix='')
            self.downsample.add(gluon.nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(gluon.nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(residual+x, act_type='relu')

        return x

def resnet18_v1_split():
    '''
    Exactly the same architecture as
    mxnet.gluon.model_zoo.vision.resnet18_v1,
    but cut in half & without dense layer.

    Input size is a tensor of size (BATCH_SIZE, 3, 224, 224) that
    represents a batch of images of size BATCH_SIZE.
    '''
    net = gluon.nn.HybridSequential()
    body = gluon.nn.HybridSequential()
    net.add(body)

    body.add(gluon.nn.Conv2D(channels=32, kernel_size=(7, 7), strides=2, padding=3,
            use_bias=False))
    body.add(gluon.nn.BatchNorm())
    body.add(gluon.nn.Activation('relu'))
    body.add(gluon.nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    block = gluon.nn.HybridSequential()
    block.add(ResNetSplitBlock(channels=32, stride=1))
    block.add(ResNetSplitBlock(channels=32, stride=1))
    body.add(block)

    block = gluon.nn.HybridSequential()
    block.add(ResNetSplitBlock(channels=64, stride=2, downsample=True))
    block.add(ResNetSplitBlock(channels=64, stride=1))
    body.add(block)

    block = gluon.nn.HybridSequential()
    block.add(ResNetSplitBlock(channels=128, stride=2, downsample=True))
    block.add(ResNetSplitBlock(channels=128, stride=1))
    body.add(block)

    block = gluon.nn.HybridSequential()
    block.add(ResNetSplitBlock(channels=256, stride=2, downsample=True))
    block.add(ResNetSplitBlock(channels=256, stride=1))
    body.add(block)

    body.add(gluon.nn.GlobalAvgPool2D())

    return net

class SplitResNet(gluon.HybridBlock):
    def __init__(self, output_size, **kwargs):
        super(SplitResNet, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = resnet18_v1_split()
            self.conv2 = resnet18_v1_split()
            self.dense = gluon.nn.Dense(output_size)

    def hybrid_forward(self, F, data):
        conv1_out = F.relu(self.conv1(data))
        conv2_out = F.relu(self.conv2(data))
        concat = F.concat(conv1_out, conv2_out, dim=1)
        return self.dense(concat)

    def save_parameters_split(self, chkpt):
        self.conv1.save_parameters(f'conv1-{chkpt}.params')
        self.conv2.save_parameters(f'conv2-{chkpt}.params')
        self.dense.save_parameters(f'dense-{chkpt}.params')

