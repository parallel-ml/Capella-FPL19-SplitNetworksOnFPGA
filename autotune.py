import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

import topi
import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack

import splitnet

model = "resnet18_v1"

"""
Code to autotune a split network using the TVM autotuning tool.

Maybe someday it will work.
"""

# Tracker host and port can be set by your environment
tracker_host = os.environ.get("TVM_TRACKER_HOST", '0.0.0.0')
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

device_host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
device_port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

# Make sure that TVM was compiled with RPC=1
assert tvm.module.enabled('rpc')

# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "resnet18_v1"
start_pack="nn.max_pool2d"
stop_pack="nn.global_avg_pool2d"

# Tuning option
log_file = "%s.%s.log" % (device, network)
tuning_option = {
    'log_filename': log_file,

    'tuner': 'random',
    'n_trial': 1000,
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET, host=tracker_host, port=tracker_port,
            number=5,
            timeout=1000,
            check_correctness=True,
        ),
    ),
}

def compile_model():
    # Populate the shape and data type dictionary
    dtype_dict = {"data": 'float32'}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    # gluon_model = vision.get_model(model, pretrained=True)
    gluon_model = splitnet.resnet18_v1_split()
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    with relay.quantize.qconfig(global_scale=8.0,
                                skip_conv_layers=[0]):
        relay_prog = relay.quantize.quantize(mod["main"], params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            relay_prog,
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack)
    
    return relay_prog, params

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def register_vta_tuning_tasks():
    from tvm.autotvm.task.topi_integration import TaskExtractEnv, deserialize_args

    @tvm.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.const(a_min, x.dtype)
        const_max = tvm.const(a_max, x.dtype)
        x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.task.register("topi_nn_conv2d", override=True)
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        args = deserialize_args(args)
        A, W = args[:2]

        with tvm.target.vta():
            res = topi.nn.conv2d(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.current_target().device_name == 'vta':
            s = topi.generic.schedule_conv2d_nchw([res])
        else:
            s = tvm.create_schedule([res.op])
        return s, [A, W, res]

def tune_and_evaluate(tuning_opt):

    if env.TARGET != "sim":
        # remote = rpc.connect(device_host, device_port)
        # Get remote from fleet node
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, tracker_port, timeout=10000)
        # Reconfigure the JIT runtime and FPGA.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # In simulation mode, host the RPC server locally.
        remote = rpc.LocalSession()

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_model()
    tasks = autotvm.task.extract_from_program(func=relay_prog,
                                              params=params,
                                              ops=(tvm.relay.op.nn.conv2d,),
                                              target=target,
                                              target_host=env.target_host)

    # We should have extracted 10 convolution tasks
    assert len(tasks) == 10
    print("Extracted {} conv2d tasks:".format(len(tasks)))
    for tsk in tasks:
        print("\t{}".format(tsk))

    # We do not run the tuning in our webpage server since it takes too long.
    # Comment the following line to run it by yourself.
    # return

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            if target.device_name != "vta":
                graph, lib, params = relay.build(
                    relay_prog, target=target,
                    params=params, target_host=env.target_host)
            else:
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_prog, target=target,
                        params=params, target_host=env.target_host)

        # Export library
        print("Upload...")
        temp = util.tempdir()
        lib.save(temp.relpath("graphlib.o"))
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # Generate the graph runtime
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_runtime.create(graph, lib, ctx)

        # upload parameters to device
        image = tvm.nd.array(
            (np.random.uniform(size=(1, 3, 224, 224))).astype('float32'))
        m.set_input(**params)
        m.set_input('data', image)

        # evaluate
        print("Evaluate inference time cost...")
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

tune_and_evaluate(tuning_option)




