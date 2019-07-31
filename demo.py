from __future__ import absolute_import, print_function

import argparse, json, os, time, sys, threading, queue
from os.path import join
from PIL import Image
import numpy as np

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download

import vta
from vta.testing import simulator
from vta.top import graph_pack

import splitnet

class WorkerThread(threading.Thread):
    '''
    Thread class that communicates to a remote
    pynq board using TVM's RPC server.

    This thread is responsible for compiling a network (split or full), deploying to FPGA,
    and feeding inputs to it.
    '''

    def __init__(self, pynq_addr, thread_id, jobs, outputs):
        '''
        Args:
          pynq_addr: ip address of pynq board to connect to
          thread_id: id of this thread; used to concatenate features later
          jobs: queue of np arrays; inputs to feed to model is enqueued here
          outputs: outputs of model is enqueued here 
        '''
        super(WorkerThread, self).__init__()
        self.pynq_addr = pynq_addr
        self.id = thread_id
        self.jobs = jobs
        self.outputs = outputs
        self.compile_model()
        
    def compile_model(self):
        if device == 'vta':
            self.remote = rpc.connect(self.pynq_addr, 9091)
            vta.reconfig_runtime(self.remote)
            vta.program_fpga(self.remote, bitstream=None)
        else:
            self.remote = rpc.LocalSession()

        self.ctx = self.remote.ext_dev(0) if device == 'vta' else self.remote.cpu(0)

        # Load pre-configured AutoTVM schedules
        with autotvm.tophub.context(target):

            # Populate the shape and data type dictionary for ResNet input
            dtype_dict = {'data': 'float32'}
            shape_dict = {'data': (env.BATCH, 3, 224, 224)}

            gluon_model = vision.resnet18_v1(pretrained=True, ctx=ctx).features if args.nonsplit else splitnet.resnet18_v1_split()

            # Measure build start time
            build_start = time.time()

            # Start front end compilation
            mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

            # Update shape and type dictionary
            shape_dict.update({k: v.shape for k, v in params.items()})
            dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

            # Perform quantization in Relay
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                relay_prog = relay.quantize.quantize(mod['main'], params=params)

            # Perform graph packing and constant folding for VTA target
            if target.device_name == 'vta':
                assert env.BLOCK_IN == env.BLOCK_OUT
                relay_prog = graph_pack(
                    relay_prog,
                    env.BATCH,
                    env.BLOCK_OUT,
                    env.WGT_WIDTH,
                    start_name=start_pack,
                    stop_name=stop_pack)

            # Compile Relay program with AlterOpLayout disabled
            with relay.build_config(opt_level=3, disabled_pass={'AlterOpLayout'}):
                if target.device_name != 'vta':
                    graph, lib, params = relay.build(
                        relay_prog, target=target,
                        params=params, target_host=env.target_host)
                else:
                    with vta.build_config():
                        graph, lib, params = relay.build(
                            relay_prog, target=target,
                            params=params, target_host=env.target_host)

            self.params = params

            # Measure Relay build time
            build_time = time.time() - build_start
            print(f'inference graph for thread {self.id} built in {0:.4f}s!'.format(build_time))

            # Send the inference library over to the remote RPC server
            temp = util.tempdir()
            lib.save(temp.relpath('graphlib.o'))
            self.remote.upload(temp.relpath('graphlib.o'))
            lib = self.remote.load_module('graphlib.o')

            # Graph runtime
            self.m = graph_runtime.create(graph, lib, self.ctx)

    def run(self):
        print(f'thread {self.id} started')
        while True:
            # get output
            data = self.jobs.get()

            if isinstance(data, str):
                break

            if args.i:
                print(f'thread {self.id} received job')

            self.m.set_input(**self.params)
            self.m.set_input('data', data)

            timer = self.m.module.time_evaluator('run', self.ctx, number=num, repeat=rep)

            if device != 'vta':
                simulator.clear_stats()
            tcost = timer()

            # Get classification results
            tvm_output = self.m.get_output(0).asnumpy()

            # enqueue result
            self.outputs.put((tvm_output, self.id))


def concat(features):
    '''
    Concatenate numpy arrays.

    Args:
      features: sequence of numpy arrays to concatenate
    '''
    if args.nonsplit:
        return nd.array(features[0])

    arr = np.concatenate(features, axis=1)
    return nd.array(arr)


def predict(model, features, k, categories):
    '''
    Make a top-k predication given an output model, features, and categories

    Args:
      model: a mxnet.gluon model
      features: input to feed to model
      k: top-k predictions to report
      categories: prediciton categories
    '''
    predictions = model(features).softmax()
    top_pred = predictions.topk(k=3)[0].asnumpy()
    if args.i:
        print('')
        print('result')
        print('======')
        for index in top_pred:
            probability = predictions[0][int(index)]
            category = categories[int(index)]
            print('{}: {:.2f}%'.format(category, probability.asscalar()*100))

# parse command line arguments
parser = argparse.ArgumentParser(description='demo for split ResNet')
parser.add_argument('--nonsplit', action='store_true', default=False)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--i', action='store_true', default=False)
args = parser.parse_args()

# get dense layer, categories for imagenet
ctx = mx.cpu()
dense = vision.resnet18_v1(pretrained=True, ctx=ctx).output
mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/image_net_labels.json')
categories = np.array(json.load(open('image_net_labels.json', 'r')))

assert tvm.module.enabled('rpc')
# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# device, `vta` or `cpu`
device = 'cpu' if args.cpu else 'vta'
target = env.target if device == 'vta' else env.target_vta_cpu

start_pack = 'nn.max_pool2d'
stop_pack = 'nn.global_avg_pool2d'

# Perform inference and gather execution statistics
num = 1 # number of times we run module for a single measurement
rep = 1 # number of measurements (we derive std dev from this)

# ip addresses of pynq boards, hardcoded for demo
if args.nonsplit:
    pynqs = ['192.168.1.15']
else:
    pynqs = ['192.168.1.15', '192.168.1.33']

threads = []
RESULT, ID = range(2)

# where worker threads push results
outputs = queue.Queue()

# initialize threads
for i in range(len(pynqs)):
    ip_addr = pynqs[i]
    t = WorkerThread(ip_addr, thread_id=i, jobs=queue.Queue(), outputs=outputs)
    threads.append(t)

features = [None] * len(threads)

for t in threads:
    t.start()

def stats(limit=10):
    '''
    Runs inference on limit number of images of animals
    downloaded from Google and report the mean, standard deviation
    of inference delay.

    Args:
      limit: number of images to download
    '''
    from google_images_download import google_images_download
    import statistics

    response = google_images_download.googleimagesdownload()
    arguments = {'keywords': 'animals', 'limit': limit, 'print_urls': True}
    paths = response.download(arguments)
    delays = []

    for path in paths[0]['animals']:
        delays.append(run_inference(path))
    
    print(f'\nran inference {limit} times')
    print('===========================')
    print(f'mean: {statistics.mean(delays)} s')
    print(f'stdev: {statistics.stdev(delays)} s')

    # clean up
    for path in paths[0]['animals']:
        os.remove(path)

    for t in threads:
        t.jobs.put('e')
        t.join()


def run_inference(img_path):
    '''
        Runs inference on a single
        image on model deployed to FPGA.

        Args:
          img_path: string path of image to run inference on
    '''
    # Prepare test image for inference
    image = Image.open(img_path).resize((224, 224))
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, env.BATCH, axis=0)

    start_time = time.time()
    for t in threads:
        t.jobs.put(image)

    for i in range(len(threads)):
        output = outputs.get()
        features[output[ID]] = output[RESULT]

    predict(dense, concat(features), 3, categories)
    predict_time = time.time() - start_time
    return predict_time

    print('\nperformed inference in %.4fs/sample' % (predict_time / (num*rep)))

# interactive mode
if args.i:
    while True:
        img_path = input('\nenter a path to image file or e to exit: ')
        if img_path == 'e':
            for t in threads:
                t.jobs.put('e')
                t.join()
                print(f'thread {t.id} joined')
            break

        predict_time = run_inference(img_path)
        print('\nperformed inference in %.4fs/sample' % (predict_time / (num*rep)))
else:
    stats(50)
    











