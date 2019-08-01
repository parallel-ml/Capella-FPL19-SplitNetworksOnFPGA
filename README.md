# FPL19 Split Networks on FPGA

Implementation of a split, distributed CNN (ResNet V1 18), deployed to 2 [PYNQ FPGA boards](https://pynq.io) using [TVM/VTA](https://tvm.ai).

## Motivation

Implementation of deep neural networks (DNNs) are hard to achieve on edge devices because DNNs
often require more resources than those provided by individual edge devices.

The idea of this project is to create an edge-tailored model by splitting a DNN into independent narrow DNNs to run
separately on multiple edge devices in parallel.

The outputs from the split networks are then
concatenated and fed through the fully connected layers to perform inference.

## Code Description
- `splitnet.py` contains split models built with [MxNet Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/index.html). Only `resnet18_v1_split` is implemented so far.
  - `resnet18_v1_split` returns a split version of `mxnet.gluon.model_zoo.vision.resnet18_v1`; initialized with random weights.
- `demo.py` demonstrates how to deploy split networks to 2 PYNQ FPGA boards with TVM/VTA and how to concatenate the results.
- `autotune.py` uses TVM's autotuning tool to achieve fast performance when running `resnet18_v1_split` on PYNQ FPGA. Currently broken.

## Setup

### PYNQ Boards
To deploy the split networks, first acquire 2 PYNQ boards
and set them up following instructions [here](https://pynq.readthedocs.io/en/latest/getting_started/pynq_z1_setup.html).

After PYNQ boards are set up, follow instructions [here](https://docs.tvm.ai/vta/install.html#pynq-side-rpc-server-build-deployment) to
launch TVM-based RPC servers on both boards. You should see the following output when starting the RPC server:

```
INFO:root:RPCServer: bind to 0.0.0.0:9091
```

The RPC server should be listening on port `9091`.

### Local 
The following instructions apply to your local machine. CNN models are developed, compiled
& uploaded to PYNQ boards *from your local machine* via RPC.

First, install TVM with LLVM enabled. Follow the instructions [here](https://docs.tvm.ai/install/from_source.html).

Install the necessary python dependencies:

```
pip3 install --user numpy decorator attrs
```

Next, you need to add a configuration file for VTA:

```
cd <tvm root>
cp vta/config/pynq_sample.json vta/config/vta_config.json
```

When the TVM compiler compiles the convolutional operators in a neural network, it queries a log file to
get the best knob parameters to achieve fast performance. Normally, for a particular network, this log file
is generated using TVM's autotuning tool (`autotune.py`).

However, since this tool seems to be broken, log file
for `resnet18_v1_split` was manually created.

Move this log file to where the compiler can find it:

```
cd <project root>
cp vta_v0.05.log ~/.tvm/tophub/vta_v0.05.log
```

## Usage
After setup has been complete on both the PYNQ and host end, you are
now ready to deploy the split networks. `demo.py` is a minimal example that shows you how to do this.

First, install additional Python dependencies:

```
pip3 install --user numpy decorator attrs
```

Then run the demo:

```
python3 demo.py [--cpu] [--nonsplit] [--i]
```

### Options:
- `--cpu` Run model on local machine instead of PYNQ boards.

- `--nonsplit` Run the non-split version of the model.

- `--i` Run the interactive version of the demo. This allows you to enter paths to image files to feed to model.

By default, the demo downloads 50 images of animals from Google Images, feeds them to the model, and reports the mean and standard deviation (in sec) of the inference delays. 
