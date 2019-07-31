# FPL19 Split Networks on FPGA

Implementation of a split, distributed CNN (ResNet V1 18), deployed to 2 PYNQ FPGA boards using [TVM/VTA](https://tvm.ai).

Model is built using the [MxNet Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/index.html) frontend.

## Usage
To run the demo, run:

```
python3 demo.py [--cpu] [--nonsplit] [--i]
```

Options:

`--cpu` Run model on local machine instead of FPGA.

`--nonsplit` Run the non-split version of the model.

`--i` Run the interactive version of the demo. This allows you to enter paths to image files to feed to model.

By default, the demo downloads 50 images of animals from Google Images, feed them to the model, and reports the mean and standard deviation (in sec) of the inference delays. 
