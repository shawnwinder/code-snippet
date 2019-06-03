from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import os
import cv2
import lmdb
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import yaml
import sys

from caffe2.python import (
    workspace,
    model_helper,
    core, brew,
    optimizer,
)
from caffe2.proto import caffe2_pb2



###############################################################################
# net_test_0
"""
# db_path = '/mnt/disk1/zhibin/experiment_data/fine_grained/mafat_lmdb/training_lmdb_encoded/general_class_training_lmdb/'
# db_path = '/mnt/disk1/zhibin/experiment_data/fine_grained/mafat_lmdb/training_lmdb_encoded/general_class_sub_class_training_lmdb/'
db_path = '/mnt/disk1/zhibin/experiment_data/fine_grained/mafat_lmdb/training_lmdb_encoded/feature_training_lmdb/'
num_labels = 13
label_type = 1
batch_size = 20
label_prob = 1

workspace.ResetWorkspace()
model = model_helper.ModelHelper(name='foo')

device_opt = caffe2_pb2.DeviceOption()
device_opt.device_type = caffe2_pb2.CUDA
device_opt.cuda_gpu_id = 2
with core.DeviceScope(device_opt):
    db_reader = model.CreateDB(
        'db_reader',
        db=db_path,
        db_type='lmdb'
    )
    brew.image_input(
        model,
        db_reader,
        ['data', 'raw_label'],
        batch_size=batch_size,
        use_gpu_transform=False,
        scale=224,
        crop=224,
        mean_per_channel=[128., 128., 128.],
        std_per_channel=[128., 128., 128.],
        mirror=True,
        is_test=False,
        label_type=label_type,
        num_labels=num_labels,
    )
    brew.fc(model, 'data', 'fc', dim_in=224*224*3, dim_out=num_labels)

    # compute multi-label loss
    # model.net.ExpandDims('raw_label', 'expanded_label', dims=[0,1],)
    # model.net.ReduceSum('expanded_label', 'reduced_sum', axes=(3,), keepdims=0)
    # model.net.MergeDim('reduced_sum', 'reduced_sum')
    # model.net.MergeDim('reduced_sum', 'reduced_sum')
    model.net.ReduceBackSum('raw_label', 'reduced_sum')
    model.net.Div(['raw_label', 'reduced_sum'], 'label', broadcast=True, axis=0)

    model.net.SoftmaxWithLoss(
        ['fc', 'label'],
        ['softmax', 'avg_loss'],
        scale = float(num_labels),
        # scale = 12.0,
        label_prob = label_prob,
    )


workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)

print("fc: ", workspace.FetchBlob('fc'))
print("raw_label: ", workspace.FetchBlob('raw_label'))
print("reduced_sum: ", workspace.FetchBlob('reduced_sum'))
print("label: ", workspace.FetchBlob('label'))
print("softmax:", workspace.FetchBlob("softmax"))
print("avg_loss:", workspace.FetchBlob("avg_loss"))
"""


###############################################################################
# op_test_0
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "SoftmaxWithLoss",
        ["logits", "labels"],
        ["softmax", "avgloss"],
    )

workspace.FeedBlob(
    "logits",
    np.asarray([[.1, .4], [.2, .3]]).astype(np.float32),
    device_option=device_opt,
)

workspace.FeedBlob(
    "labels",
    np.asarray([[1., 0.], [0., 1.]]).astype(np.float32),
    device_option=device_opt,
)
print("logits:\n", workspace.FetchBlob("logits"))
print("labels:\n", workspace.FetchBlob("labels"))

workspace.RunOperatorOnce(op)
print("softmax:\n", workspace.FetchBlob("softmax"))
print("avgloss:\n", workspace.FetchBlob("avgloss"))
"""


###############################################################################
# op_test_1
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "ReduceBackSum",
        # "ReduceFrontSum",
        ["X"],
        ["reduced_sum"],
        # num_reduce_dims=1,
        # keep_dims=1,  # no this arg!
    )

workspace.FeedBlob(
    "X",
    np.arange(1,11).reshape(2, 5).astype(np.float32),
    device_option=device_opt
)
print("X:", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("ReduceSum:", workspace.FetchBlob("reduced_sum"))
'''


###############################################################################
# op_test_2
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
            "Add",
            ["X", "Y"],
            ["Z"],
            broadcast=1,
    )
workspace.FeedBlob(
    "X",
    np.random.randint(10, size=(2,3,3)).astype(np.float32),
    device_option = device_opt,
)
workspace.FeedBlob(
    "Y",
    np.asarray([2]).astype(np.float32),
    device_option = device_opt,
)
print("X:", workspace.FetchBlob("X"))
print("Y:", workspace.FetchBlob("Y"))

workspace.RunOperatorOnce(op)
print("Z:", workspace.FetchBlob("Z"))
"""


###############################################################################
# op_test_3
'''
workspace.ResetWorkspace()
op = core.CreateOperator(
        "ExpandDims",
        ["X"],
        ["X"],
        dims=[0,1],
)
#workspace.FeedBlob("X", np.random.randint(10, size=(2,5)).astype(np.float32))
workspace.FeedBlob("X", np.asarray([[1., 2., 3.]]).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X:", workspace.FetchBlob("X"))
# print("Y:", workspace.FetchBlob("Y"))
'''


###############################################################################
# op_test_4
'''
workspace.ResetWorkspace()
op = core.CreateOperator(
        "MergeDim",
        ["X"],
        ["X"],
)
# workspace.FeedBlob("X", np.random.randint(10, size=(2,5)).astype(np.float32))
workspace.FeedBlob("X", np.asarray([[1., 2., 3.]]).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X:", workspace.FetchBlob("X"))
# workspace.FeedBlob("X", workspace.FetchBlob("Y"))
# workspace.RunOperatorOnce(op)
# print("Y:", workspace.FetchBlob("Y"))
'''


###############################################################################
# op_test_5
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Div",
        ["X", "Y"],
        ["Z"],
        # broadcast=1,
        # axis=0,
    )

workspace.FeedBlob(
    "X",
    np.arange(1, 25).reshape(4, 6).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))
workspace.FeedBlob(
    "Y",
    # 2 * np.ones((4, 6)).astype(np.float32),
    2 * np.arange(1, 25).reshape(4, 6).astype(np.float32),
    device_option=device_opt,
)
print("Y:\n", workspace.FetchBlob("Y"))

workspace.RunOperatorOnce(op)
print("Z:\n", workspace.FetchBlob("Z"))
'''


###############################################################################
# op_test_6
'''
workspace.ResetWorkspace()
op = core.CreateOperator(
    "UnsortedSegmentMean",
    ["X", "IDX"],
    ["Y"],
)

workspace.FeedBlob( "X", np.asarray([[1., 0., 1., 0., 0.],
                                     [0., 1., 1., 0., 1.]]).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.FeedBlob("IDX", np.asarray([1, 1]).astype(np.int32))
print("IDX:", workspace.FetchBlob("IDX"))

workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
'''


###############################################################################
# net_test_1
"""
db_path = '/mnt/disk1/zhibin/experiment_data/fine_grained/mafat_lmdb/training_lmdb_encoded/general_class_training_lmdb/'
num_class = 2
num_labels = 0
label_type = 0
batch_size = 5
label_prob = 0

workspace.ResetWorkspace()
model = model_helper.ModelHelper(name='foo')

device_opt = caffe2_pb2.DeviceOption()
device_opt.device_type = caffe2_pb2.CUDA
device_opt.cuda_gpu_id = 2
with core.DeviceScope(device_opt):
    db_reader = model.CreateDB(
        'db_reader',
        db=db_path,
        db_type='lmdb'
    )
    brew.image_input(
        model,
        db_reader,
        ['data', 'label'],
        batch_size=batch_size,
        use_gpu_transform=False,
        scale=224,
        crop=224,
        mean_per_channel=[128., 128., 128.],
        std_per_channel=[128., 128., 128.],
        mirror=True,
        is_test=False,
        label_type=label_type,
        num_labels=num_labels,
    )
    brew.fc(model, 'data', 'fc', dim_in=224*224*3, dim_out=num_class)

    model.net.SoftmaxWithLoss(
        ['fc', 'label'],
        ['softmax', 'avg_loss'],
        scale = 1.0,
        label_prob = label_prob,
    )


workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)

print("fc: ", workspace.FetchBlob('fc'))
print("label: ", workspace.FetchBlob('label'))
print("softmax:", workspace.FetchBlob("softmax"))
print("avg_loss:", workspace.FetchBlob("avg_loss"))
"""


###############################################################################
# op_test_7
'''
workspace.ResetWorkspace()
op = core.CreateOperator(
    "Concat",
    ["X", "Y"],
    ["C", "_"],
    axis=0,
)

workspace.FeedBlob("X", np.random.randint(10, size=(2,3,4,4)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.FeedBlob("Y", np.random.randint(10, size=(10,3,4,4)).astype(np.float32))
print("Y:", workspace.FetchBlob("Y"))

workspace.RunOperatorOnce(op)
C = workspace.FetchBlob("C")
print("C:", C)
print("C.shape", C.shape)
'''


###############################################################################
# op_test_8
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "SigmoidCrossEntropyWithLogits",
    ["logits", "label"],
    ["xent"],
)

workspace.FeedBlob("logits", np.asarray([[0.5, 0.76, 1.32, 0.891, 0.21],
                                         [0.56, 0.98, 0.123, 0.7663, 1.331]]).astype(np.float32))
print("logits:", workspace.FetchBlob("logits"))
workspace.FeedBlob("label", np.asarray([[0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1]]).astype(np.float32))
print("label:", workspace.FetchBlob("label"))

workspace.RunOperatorOnce(op)
xent = workspace.FetchBlob("xent")
print("xent:", xent)
print("xent.shape", xent.shape)
"""


###############################################################################
# op_test_9
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "Cast",
    ["X"],
    ["Y"],
    to=core.DataType.FLOAT,
)

workspace.FeedBlob("X", np.asarray([[0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1]]).astype(np.float32))
print("X:", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
Y = workspace.FetchBlob("Y")
print("Y:", Y)
"""


###############################################################################
# op_test_10
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "BatchGather",
    ["Data", "Indices"],
    ["Output"],
)

workspace.FeedBlob("Data", np.asarray([[1, 0, 3, 4, 2]]).astype(np.int32))
print("Data:\n", workspace.FetchBlob("Data"))

# workspace.FeedBlob("Indices", np.asarray([0]).astype(np.int32).reshape(-1, 1))
# workspace.FeedBlob("Indices", np.asarray([0, 1]).astype(np.int32))
workspace.FeedBlob("Indices", np.asarray([0, 1]).astype(np.int32))
print("Indices:\n", workspace.FetchBlob("Indices"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("Output")
print("Output:\n", Output)
"""


###############################################################################
# op_test_11
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "Gather",
    ["Data", "Indices"],
    ["Output"],
)

workspace.FeedBlob("Data", np.asarray([1, 0, 3, 4, 2]).astype(np.int32))
print("Data:\n", workspace.FetchBlob("Data"))

# workspace.FeedBlob("Indices", np.asarray([0]).astype(np.int32).reshape(-1, 1))
# workspace.FeedBlob("Indices", np.asarray([0, 1]).astype(np.int32))
workspace.FeedBlob("Indices", np.asarray([3]).astype(np.int32))
print("Indices:\n", workspace.FetchBlob("Indices"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("Output")
print("Output:\n", Output)
"""


###############################################################################
# op_test_12
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "OneHot",
    ["indices", "index_size"],
    ["one_hot"],
)

workspace.FeedBlob("indices", np.asarray([0, 1, 2]).astype(np.long))
print("Indices:\n", workspace.FetchBlob("indices"))
workspace.FeedBlob("index_size", np.array([3]).astype(np.long))
print("index_size:\n", workspace.FetchBlob("index_size"))

workspace.RunOperatorOnce(op)
one_hot = workspace.FetchBlob("one_hot")
print("one_hot:\n", one_hot)
print("one_hot type:\n", type(one_hot[0][0]))
"""


###############################################################################
# op_test_12
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "FlattenToVec",
        ["data"],
        ["flattened_data"],
    )

workspace.FeedBlob(
    "data",
    np.asarray([[1, 0, 3, 4, 2],
                [0, 3, 0, 4, 1],
                [7, 2, 6, 9, 0]]).astype(np.float32),
    device_option = device_opt,
)

print("data:\n", workspace.FetchBlob("data"))

workspace.RunOperatorOnce(op)
flattened_data = workspace.FetchBlob("flattened_data")
print("flattened_data:\n", flattened_data)
"""


###############################################################################
# op_test_13
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "BooleanMask",
    ["data", "mask"],
    ["masked_data", "masked_indices"],
)

workspace.FeedBlob("data", np.asarray([1,2,3,4,5,6,7,8,9]).astype(np.long))
print("data:\n", workspace.FetchBlob("data"))

workspace.FeedBlob("mask", np.asarray([1,0,0,0,1,0,0,0,1]).astype(np.bool))
# workspace.FeedBlob("mask", np.asarray([1,0,0,0,1,0,0,0,1]).astype(np.long))
print("mask:\n", workspace.FetchBlob("mask"))

workspace.RunOperatorOnce(op)
masked_data = workspace.FetchBlob("masked_data")
print("masked_data:\n", masked_data)
"""


###############################################################################
# op_test_14
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "GivenTensorIntFill",
        [],
        ["Output"],
        values=np.random.randint(10, size=10),
        shape=[10, 1],
    )

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("Output")
print("Output:\n", Output)
"""


###############################################################################
# op_test_15
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Mul",
        ["A", "B"],
        ["C"],
        broadcast=1,
        # axis=0,
    )

workspace.FeedBlob("A", np.ones((10, 4)).astype(np.int32),
                   device_option=device_opt)
print("A: ", workspace.FetchBlob("A"))
workspace.FeedBlob("B", np.arange(1, 11).reshape(10, 1).astype(np.int32),
                   device_option=device_opt)
print("B: ", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("C")
print("Output:\n", Output)
'''


###############################################################################
# op_test_16
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Reshape",
        ["data"],
        ["reshaped", "old_shape"],
        shape=(-1, 1),
    )

workspace.FeedBlob(
    "data",
    np.asarray([1, 0, 3, 4, 2]).astype(np.float32),
    device_option=device_opt,
)
print("data:\n", workspace.FetchBlob("data"))

workspace.RunOperatorOnce(op)
print("reshaped:\n", workspace.FetchBlob("reshaped"))
"""


###############################################################################
# net_test_2
"""
workspace.ResetWorkspace()
batch_size = 3
num_classes = 5
net = core.Net("gather_test")
# flatten data
flattened_data = net.FlattenToVec(
    ['data'],
    ['flattened_data'],
)

# flatten label
label_size = net.ConstantFill(
    [],
    ['label_size'],
    value=num_classes,
    dtype=core.DataType.INT64,
)
label = net.Cast(
    ['label_raw'],
    ['label'],
    from_type=core.DataType.INT32,
    to=core.DataType.INT64,
)
one_hot_label = net.OneHot(
    ['label', 'label_size'],
    ['one_hot_label'],
)
flattened_one_hot_label = net.FlattenToVec(
    ['one_hot_label'],
    ['flattened_one_hot_label'],
)
flattened_one_hot_label_bool = net.Cast(
    ['flattened_one_hot_label'],
    ['flattened_one_hot_label_bool'],
    from_type=core.DataType.FLOAT,
    to=core.DataType.BOOL,
)

# apply label mask to data
masked_data, _ = net.BooleanMask(
    ['flattened_data', 'flattened_one_hot_label_bool'],
    ['masked_data', 'masked_labels'],
)

# calculate masked data loss
log_masked_data = net.Log(
    ['masked_data'],
    ['log_masked_data'],
)
negative_log_masked_data = net.Negative(
    ['log_masked_data'],
    ['negative_log_masked_data'],
)
loss = net.ReduceBackSum(
    ['negative_log_masked_data'],
    ['loss'],
)

# add bp
net.AddGradientOperators([loss])



workspace.FeedBlob("data", np.asarray([[1, 0, 3, 4, 2],
                                    [0, 3, 0, 4, 1],
                                    [7, 2, 6, 9, 0]]).astype(np.float32))
print("data:\n", workspace.FetchBlob("data"))

workspace.FeedBlob("label_raw", np.asarray([0, 1, 2]).astype(np.int32))
print("label_raw:\n", workspace.FetchBlob("label_raw"))

workspace.RunNetOnce(net)
print("masked_data:\n", workspace.FetchBlob("masked_data"))
print("log_masked_data: \n", workspace.FetchBlob("log_masked_data"))
print("loss: \n", workspace.FetchBlob("loss"))
print("net proto: \n{}".format(net.Proto()))
"""


###############################################################################
# net_test_3
"""
workspace.ResetWorkspace()
batch_size = 3
num_classes = 5
net = core.Net("gather_test_2")
# flatten data
flattened_data = net.Reshape(
    ['data'],
    ['flattened_data', 'old_shape'],
    shape=(-1, 1),
)

# flatten label
label_stride = net.GivenTensorIntFill(
    [],
    ['label_stride'],
    values=np.arange(batch_size) * num_classes,
    shape=[batch_size],
)
label = net.Sum(
    ['label', 'label_stride'],
    ['label'],
    broadcast=1,
)

# gather
scores = net.Gather(
    ['flattened_data', 'label'],
    ['scores'],
)

# calculate masked data loss
log_scores = net.Log(
    ['scores'],
    ['log_scores'],
)
negative_log_scores = net.Negative(
    ['log_scores'],
    ['negative_log_scores'],
)
loss_array = net.ReduceBackSum(
    ['negative_log_scores'],
    ['loss_array'],
)
loss = net.ReduceBackSum(
    ['loss_array'],
    ['loss'],
)

# add bp
net.AddGradientOperators([loss])



workspace.FeedBlob("data", np.asarray([[1, 0, 3, 4, 2],
                                    [0, 3, 0, 4, 1],
                                    [7, 2, 6, 9, 0]]).astype(np.float32))
print("data:\n", workspace.FetchBlob("data"))

workspace.FeedBlob("label", np.asarray([0, 3, 1]).astype(np.int32))
print("label:\n", workspace.FetchBlob("label"))

workspace.RunNetOnce(net)
print("flattened_data:\n", workspace.FetchBlob("flattened_data"))
print("scores:\n", workspace.FetchBlob("scores"))
print("log_scores: \n", workspace.FetchBlob("log_scores"))
print("negative_log_scores: \n", workspace.FetchBlob("negative_log_scores"))
print("loss: \n", workspace.FetchBlob("loss"))
# print("net proto: \n{}".format(net.Proto()))
"""


###############################################################################
# net_test_4
"""
workspace.ResetWorkspace()
batch_size = 3
num_classes = 5
net = core.Net("gather_test_3")
# flatten data
flattened_data = net.Reshape(
    ['data'],
    ['flattened_data', 'old_shape'],
    shape=(1, -1),
)

# flatten label
label_stride = net.GivenTensorIntFill(
    [],
    ['label_stride'],
    values=np.arange(batch_size) * num_classes,
    shape=[batch_size],
)
strided_label = net.Sum(
    ['label', 'label_stride'],
    ['strided_label'],
    broadcast=1,
)

# gather
scores = net.BatchGather(
    ['flattened_data', 'strided_label'],
    ['scores'],
)

# calculate masked data loss
log_scores = net.Log(
    ['scores'],
    ['log_scores'],
)
negative_log_scores = net.Negative(
    ['log_scores'],
    ['negative_log_scores'],
)
loss = net.ReduceBackSum(
    ['negative_log_scores'],
    ['loss'],
)

# add bp
net.AddGradientOperators([loss])



workspace.FeedBlob("data", np.asarray([[1, 0, 3, 4, 2],
                                    [0, 3, 0, 4, 1],
                                    [7, 2, 6, 9, 0]]).astype(np.float32))
print("data:\n", workspace.FetchBlob("data"))

workspace.FeedBlob("label", np.asarray([0, 3, 1]).astype(np.int32))
print("label:\n", workspace.FetchBlob("label"))

workspace.RunNetOnce(net)
print("flattened_data:\n", workspace.FetchBlob("flattened_data"))
print("scores:\n", workspace.FetchBlob("scores"))
print("log_scores: \n", workspace.FetchBlob("log_scores"))
print("negative_log_scores: \n", workspace.FetchBlob("negative_log_scores"))
print("loss: \n", workspace.FetchBlob("loss"))
print("net proto: \n{}".format(net.Proto()))
"""


###############################################################################
# net_test_5
"""
workspace.ResetWorkspace()
batch_size = 3
num_classes = 5
net = core.Net("gather_test_4")
# flatten data
flattened_data = net.FlattenToVec(
    ['data'],
    ['flattened_data'],
)

#==============================================================================
# "Gather" op make the generated blob sparse, the opsite op is "ScatterAssign"
# while the "BatchGather" op generate dense blob
#==============================================================================
'''
selected_data = net.Gather(
    ['flattened_data', 'label'],
    ['selected_data'],
)
'''

# net.AddGradientOperators([selected_data])
net.AddGradientOperators([flattened_data])

workspace.FeedBlob("data", np.asarray([[1, 0, 3, 4, 2],
                                    [0, 3, 0, 4, 1],
                                    [7, 2, 6, 9, 0]]).astype(np.float32))
print("data:\n", workspace.FetchBlob("data"))

workspace.FeedBlob("label", np.asarray([0, 3, 1]).astype(np.int32))
print("label:\n", workspace.FetchBlob("label"))

workspace.RunNetOnce(net)
print("flattened_data:\n", workspace.FetchBlob("flattened_data"))
# print("selected_data:\n", workspace.FetchBlob("selected_data"))
print("net proto: \n{}".format(net.Proto()))
"""


###############################################################################
# op_test_17
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Squeeze",
        ["data"],
        ["squeezed"],
        dims=[0,]
    )

workspace.FeedBlob(
    "data",
    np.asarray([[1, 0, 3, 4, 2]]).astype(np.float32),
    device_option = device_opt,
)
print("data:\n", workspace.FetchBlob("data"))
print("data shape:\n", workspace.FetchBlob("data").shape)

workspace.RunOperatorOnce(op)
print("squeezed:\n", workspace.FetchBlob("squeezed"))
print("squeezed shape:\n", workspace.FetchBlob("squeezed").shape)
"""


###############################################################################
# op_test_18
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Split",
        ["data"],
        ["part1", "part2"],
        split=(2,1),
        axis=0,
    )

workspace.FeedBlob(
    "data",
    # np.random.randint(20, size=(4,3,4,4)).astype(np.float32),
    np.arange(1,13).reshape(3,4).astype(np.float32),
    device_option=device_opt,
)

print("data:\n", workspace.FetchBlob("data"))

workspace.RunOperatorOnce(op)
print("part1:\n", workspace.FetchBlob("part1"))
print("part2:\n", workspace.FetchBlob("part2"))
'''


###############################################################################
# op_test_19
"""
workspace.ResetWorkspace()
op = core.CreateOperator(
    "SumSqrElements",
    ["data"],
    ["sum"],
    # average=False,
    # average=True,
)

# workspace.FeedBlob("data", np.random.randint(3, size=(16,3,4,4)).astype(np.float32))
workspace.FeedBlob("data", np.ones(shape=(16,3,4,4)).astype(np.float32))
print("data:\n", workspace.FetchBlob("data"))

workspace.RunOperatorOnce(op)
print("sum:\n", workspace.FetchBlob("sum"))
"""


###############################################################################
# op_test_20
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "ReduceBackMean",
        ["X"],
        ["reduced_mean"],
    )

workspace.FeedBlob(
    "X",
    np.asarray([1, 2, 3., 4]).astype(np.float32),
    device_option=device_opt,
)
print("X:", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("ReduceMean:", workspace.FetchBlob("reduced_mean"))
"""


###############################################################################
# op_test_21
"""
np_a = np.random.randint(10, size=(3, 4)).astype(np.float32)
print("np_a:\n", np_a)
np_b = np.random.randint(10, size=(3, 4)).astype(np.float32)
print("np_b:\n", np_b)
print("np_a x np_b:\n", np.dot(np_a, np_b.T))

workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "BatchMatMul",
        ["A", "B"],
        ["C"],
        trans_b=1,
    )

workspace.FeedBlob(
    "A",
    np_a,
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))

workspace.FeedBlob(
    "B",
    np_b,
    device_option=device_opt,
)
print("B:\n", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
print("C:\n", workspace.FetchBlob("C"))
"""


###############################################################################
# op_test_22
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "DiagonalFill",
        [],
        ["diag_out"],
        value=1.0,
        dtype=core.DataType.FLOAT,
        shape=(16, 16),
    )

workspace.FeedBlob(
    "X",
    np.asarray([1, 2, 3, 4]).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("diag_out:\n", workspace.FetchBlob("diag_out"))
"""


###############################################################################
# op_test_23
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Softmax",
        ["logits"],
        ["softmax"],
    )

workspace.FeedBlob(
    "logits",
    np.asarray([[.1, .4], [.2, .3]]).astype(np.float32),
    device_option=device_opt,
)

workspace.RunOperatorOnce(op)
print("softmax:\n", workspace.FetchBlob("softmax"))
"""


###############################################################################
# op_test_23
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "CrossEntropy",
        ["logits", "labels"],
        ["xent"],
    )

workspace.FeedBlob(
    "logits",
    np.asarray([[.1, .4], [.2, .3]]).astype(np.float32),
    device_option=device_opt,
)
print("logits:\n", workspace.FetchBlob("logits"))

workspace.FeedBlob(
    "labels",
    np.asarray([[1., 0.], [0., 1.]]).astype(np.float32),
    device_option=device_opt,
)

print("labels:\n", workspace.FetchBlob("labels"))

workspace.RunOperatorOnce(op)
print("xent:\n", workspace.FetchBlob("xent"))
"""


###############################################################################
# op_test_24
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "AveragedLoss",
        ["xent"],
        ["avg_loss"],
    )

workspace.FeedBlob(
    "xent",
    np.asarray([2.5, 3.5]).astype(np.float32),
    device_option=device_opt,
)
print("xent:\n", workspace.FetchBlob("xent"))

workspace.RunOperatorOnce(op)
print("avg_loss:\n", workspace.FetchBlob("avg_loss"))
"""


###############################################################################
# net_test_6
"""
workspace.ResetWorkspace()
model = model_helper.ModelHelper(name='test_6')

device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    model.net.DiagonalFill([],['diag'], value=1.0, shape=(3, 4))
    model.net.Reshape(['diag'], ['diag_reshaped', 'old_shape'], shape=(4,3))

workspace.RunNetOnce(model.net)
print("diag:\n", workspace.FetchBlob("diag"))
print("diag_reshaped:\n", workspace.FetchBlob("diag_reshaped"))
print("old_shape:\n", workspace.FetchBlob("old_shape"))
"""


###############################################################################
# op_test_25
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Normalize",
        ["A"],
        ["A_norm"],
        axis=0,
    )

workspace.FeedBlob(
    "A",
    # np.asarray([[1, 1], [2, 2], [3, 3]]).astype(np.float32),
    np.arange(1, 25).reshape(4,6).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))

workspace.RunOperatorOnce(op)
print("A_norm:\n", workspace.FetchBlob("A_norm"))
'''


###############################################################################
# op_test_26
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Print",
        ["A"],
        [],
    )

workspace.FeedBlob(
    "A",
    np.asarray([[1, 1], [2, 2], [3, 3]]).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))

workspace.RunOperatorOnce(op)
"""


###############################################################################
# op_test_27
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Shape",
        ["A"],
        ["A_shape"],
    )

workspace.FeedBlob(
    "A",
    np.asarray([[1, 1], [2, 2], [3, 3]]).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))

workspace.RunOperatorOnce(op)
print("A_shape:\n", workspace.FetchBlob("A_shape"))
"""


###############################################################################
# op_test_28
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Concat",
        ["A", "B"],
        ["concat", "split_info"],
        axis=1,
    )

workspace.FeedBlob(
    "A",
    np.zeros((10, 1), dtype=np.int32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))

workspace.FeedBlob(
    "B",
    np.ones((10, 1), dtype=np.int32),
    device_option=device_opt,
)
print("B:\n", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
print("concat:\n", workspace.FetchBlob("concat"))
print("split_info:\n", workspace.FetchBlob("split_info"))
"""


###############################################################################
# op_test_29
# Notice: currently, this op is not supported by GPU
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "ConcatTensorVector",
        ["tensor_vector"],
        ["concat"]
    )


workspace.FeedBlob(
    "tensor_vector",
    np.asarray([[1,2,3], [4,5,6]]).astype(np.int32),
    device_option=device_opt,
)
print("tensor_vector:\n", workspace.FetchBlob("tensor_vector"))

workspace.RunOperatorOnce(op)
print("concat:\n", workspace.FetchBlob("concat"))
"""


###############################################################################
# op_test_30
# Notice: currently, this op is not supported by GPU
"""
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Append",
        ["A", "B"],
        ["A"],
    )

workspace.FeedBlob(
    "A",
    np.asarray([1,2,3]).astype(np.int32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))

workspace.FeedBlob(
    "B",
    np.asarray([4,5,6]).astype(np.int32),
    device_option=device_opt,
)
print("B:\n", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
print("A:\n", workspace.FetchBlob("A"))
"""


###############################################################################
# op_test_31
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Size",
        ["A"],
        ["A_size"],
    )

workspace.FeedBlob(
    "A",
    np.asarray([[1,2,3], [3,5,6]]).astype(np.int32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))


workspace.RunOperatorOnce(op)
print("A_size:\n", workspace.FetchBlob("A_size"))
"""


###############################################################################
# net_test_7
"""
workspace.ResetWorkspace()
model = model_helper.ModelHelper(name='test_7')

device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    A = model.net.GivenTensorFill(
        [],
        ['A'],
        values=np.random.randn(10,3,4,4),
        shape=(10,3,4,4)
    )
    N, C, H, W = (10, 3, 4, 4)

    A_reshape, _ = model.net.Reshape(
        A,
        ['A_reshape', 'A_old_shape'],
        shape=(N, C, H*W),
    )
    A_transposed = model.net.Transpose(
        [A_reshape],
        ['A_transposed'],
        axes=[0, 2, 1],
    )

    model.net.Print(model.net.Shape(A_reshape, 'A_reshaped_shape'), [])
    model.net.Print(model.net.Shape(A_transposed, 'A_transposed_shape'), [])

    outer_product = model.net.BatchMatMul(
        ["A_reshape", "A_transposed"],
        ["outer_product"],
    )

    model.AddGradientOperators([outer_product])


workspace.RunNetOnce(model.net)
print("construct model successfully")
print("outer_product shape:\n", workspace.FetchBlob("outer_product").shape)
"""


###############################################################################
# op_test_32
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Transpose",
        ["A"],
        ["A_transposed"],
        axes=[0, 2, 1],
    )

workspace.FeedBlob(
    "A",
    np.arange(24).reshape(2, 3, 4).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))


workspace.RunOperatorOnce(op)
print("A_transposed:\n", workspace.FetchBlob("A_transposed"))
print("A_transposed shape:\n", workspace.FetchBlob("A_transposed").shape)
"""


###############################################################################
# op_test_33
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Sqrt",
        ["A"],
        ["A_sqrt"],
    )

workspace.FeedBlob(
    "A",
    np.arange(4).reshape(2, 2).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))


workspace.RunOperatorOnce(op)
print("A_sqrt:\n", workspace.FetchBlob("A_sqrt"))
'''


###############################################################################
"""
# op_test_34
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Normalize",
        ["A"],
        ["A_l2"],
    )

workspace.FeedBlob(
    "A",
    np.arange(4).reshape(2, 2).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))


workspace.RunOperatorOnce(op)
print("A_l2:\n", workspace.FetchBlob("A_l2"))
"""


###############################################################################
# op_test_35
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "WeightedSum",
        ["X0", "W0", "W1", "X1"],
        ["weighted_sum"],
    )

workspace.FeedBlob(
    "X0",
    np.asarray([10]).astype(np.float32),
    device_option=device_opt,
)
print("X0:\n", workspace.FetchBlob("X0"))

workspace.FeedBlob(
    "W0",
    np.asarray([1]).astype(np.float32),
    device_option=device_opt,
)
print("W0:\n", workspace.FetchBlob("W0"))

workspace.FeedBlob(
    "X1",
    np.asarray([10]).astype(np.float32),
    device_option=device_opt,
)
print("X1:\n", workspace.FetchBlob("X1"))


workspace.FeedBlob(
    "W1",
    np.asarray([2]).astype(np.float32),
    device_option=device_opt,
)
print("W1:\n", workspace.FetchBlob("W1"))

workspace.RunOperatorOnce(op)
print("weighted_sum:\n", workspace.FetchBlob("weighted_sum"))
"""


###############################################################################
# net_test_8
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    model = model_helper.ModelHelper(name='test_8')

    ''' Add an database input data '''
    db_reader = model.CreateDB(
        'db_reader',
        # db="/home/zhibin/wangxiao/datasets/mamc_lmdb/cars196_encoded_train_lmdb",
        db="/home/zhibin/wangxiao/datasets/mamc_lmdb/cars196_train_lmdb",
        # db="/home/zhibin/wangxiao/datasets/caffe2_lmdb/stanford_cars_encoded_train_lmdb",
        # db="/home/zhibin/wangxiao/datasets/caffe2_lmdb/stanford_cars_train_lmdb",
        db_type='lmdb',
    )

    data, label = model.net.TensorProtosDBInput(
        [db_reader],
        ['data', 'label'],
        batch_size = 1,
    )
    '''
    data, label = brew.image_input(
        model,
        db_reader,
        ['data', 'label'],
        # batch_size=32,
        batch_size=1,
        use_gpu_transform=True,
        scale=256,
        crop=224,
        mean_per_channel=[128., 128., 128.],
        std_per_channel=[128., 128., 128.],
        mirror=True,
        is_test=False,
    )
    '''

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)

data = workspace.FetchBlob('data')
print("data: {}\n".format(data))
print("data shape: {}\n".format(data.shape))

label = workspace.FetchBlob('label')
print("label: {}\n".format(label))
print("label shape: {}\n".format(label.shape))
"""


###############################################################################
# op_test_36
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        # "Sub",
        "Add",
        ["A", "B"],
        ["C"],
        broadcast=1,
        axis=0,
    )

workspace.FeedBlob(
    "A",
    # (3 * np.ones((2, 3, 2, 2))).astype(np.float32),
    3 * np.ones((3,4)).astype(np.float32),
    device_option=device_opt,
)
print("A:\n", workspace.FetchBlob("A"))
workspace.FeedBlob(
    "B",
    1 * np.ones((3,4)).astype(np.float32),
    device_option=device_opt
)
print("B:\n", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("C")
print("Output:\n", Output)
"""


###############################################################################
"""
# op_test_37
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Div",
        ["A", "B"],
        ["C"],
        broadcast=1,
        # axis=0,
    )

workspace.FeedBlob(
    "A",
    (12 * np.ones((2, 3, 2, 2))).astype(np.float32),
    device_option=device_opt
)
print("A:\n", workspace.FetchBlob("A"))
workspace.FeedBlob(
    "B",
    # np.array([2,3,4]).astype(np.float32),
    # (2 * np.ones((3,))).astype(np.float32),
    np.asarray([6]).astype(np.float32),
    device_option=device_opt
)
print("B:\n", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("C")
print("Output:\n", Output)
"""


###############################################################################
# net_test_9
"""
model = model_helper.ModelHelper(name='fc_test')

device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    brew.fc(model, 'A', 'B', dim_in=12, dim_out=6)
    reshaped_A, _ = model.net.Reshape(
        ['A'],
        ['A_reshpaed', 'old_shape_of_A'],
        shape=(5, 12),
    )
    brew.fc(model, reshaped_A, 'B_reshaped', dim_in=12, dim_out=6)

workspace.FeedBlob(
    "A",
    (12 * np.ones((5, 3, 2, 2))).astype(np.float32),
    device_option=device_opt
)
print("A:\n", workspace.FetchBlob("A"))

workspace.RunNetOnce(model.param_init_net)
workspace.RunNetOnce(model.net)
print("B:\n", workspace.FetchBlob("B"))
print("B_reshaped:\n", workspace.FetchBlob("B_reshaped"))
"""


###############################################################################
# op_test_38
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "MatMul",
        ["A", "B"],
        ["C"],
    )

workspace.FeedBlob(
    "A",
    (2 * np.ones((2, 3))).astype(np.float32),
    device_option=device_opt
)
print("A:\n", workspace.FetchBlob("A"))
workspace.FeedBlob(
    "B",
    (3 * np.ones((3, 2))).astype(np.float32),
    device_option=device_opt
)
print("B:\n", workspace.FetchBlob("B"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("C")
print("Output:\n", Output)
"""


###############################################################################
# op_test_39
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "BatchSparseToDense",
        ["lenghts", "indices", "values"],
        ["dense"],
        dense_last_dim=5,
    )

workspace.FeedBlob(
    "lengths",
    np.asarray([2, 3, 1]).astype(np.int32),
    device_option=device_opt,
)
print("lengths:\n", workspace.FetchBlob("lengths"))

workspace.FeedBlob(
    "indices",
    np.asarray([0, 1, 2, 3, 4, 5]).astype(np.int32),
    device_option=device_opt,
)
print("indices:\n", workspace.FetchBlob("indices"))


workspace.FeedBlob(
    "values",
    np.asarray([6, 7, 8, 9, 10, 11]).astype(np.float32),
    device_option=device_opt,
)
print("values:\n", workspace.FetchBlob("values"))

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("dense")
print("Output:\n", Output)
"""


###############################################################################
# net_test_10
"""
model = model_helper.ModelHelper(name='stop_gradient_test')
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    # add input & model
    data = model.net.GivenTensorFill(
        [],
        ["data"],
        values = np.random.rand(3, 4),
        shape=[3, 4],
    )
    label = model.net.GivenTensorIntFill(
        [],
        ["label"],
        values = np.random.randint(3, size=3),
        shape=[3],
    )
    fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=5)

    var = model.net.GivenTensorFill(
        [],
        ["var"],
        values = np.ones((5,10)),
        shape=[5, 10],
    )

    mm = model.net.MatMul(
        [fc1, var],
        ['mm'],
    )

    fc2 = brew.fc(model, mm, 'fc2', dim_in=10, dim_out=4)
    # add loss & bp
    softmax, loss = model.net.SoftmaxWithLoss(
        [fc2, label],
        ['softmax', 'loss'],
    )
    model.AddGradientOperators([loss])
    optimizer.add_weight_decay(model, 0.001)
    optimizer.build_multi_precision_sgd(
        model,
        base_learning_rate = 0.01,
        policy = "step",
        stepsize = 10,
        momentum = 0.9,
        nesterov = 1,
        gamma = 0.96,
    )

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
for i in range(5):
    workspace.RunNet(model.net)
    print("========================================")
    label = workspace.FetchBlob('label')
    print("label: {}".format(label))
    var = workspace.FetchBlob('var')
    print("var: {}".format(var))
    fc1 = workspace.FetchBlob('fc1')
    print("fc1: {}".format(fc1))
"""


###############################################################################
# op_test_40
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "GivenTensorFill",
        [],
        ["label"],
        # values = [1, 2, 3],
        values = [1., 2., 3.], # value must be float type!
        shape=[3],
    )

workspace.RunOperatorOnce(op)
Output = workspace.FetchBlob("label")
print("Output:\n", Output)
"""


###############################################################################
# op_test_41
"""
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "RFFT",
        ["X_real"],
        ["Y_real", "Y_imag"],
    )

workspace.FeedBlob(
    "X_real",
    np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    device_option=device_opt,
)
print("X_real:\n", workspace.FetchBlob("X_real"))

workspace.RunOperatorOnce(op)
print("Y_real:\n", workspace.FetchBlob("Y_real"))
print("Y_imag:\n", workspace.FetchBlob("Y_imag"))
"""


###############################################################################
# op_test_42
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "SumElements",
        ["X"],
        ["reduced_"],
    )

workspace.FeedBlob(
    "X",
    np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 13).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
# print("reduced_:\n", workspace.FetchBlob("reduced_"))
reduced = workspace.FetchBlob("reduced_")
print(type(reduced))
print(reduced.shape)
print("reduced_:\n", reduced)
'''


###############################################################################
# op_test_43
"""
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Sum",
        ["X", "Y"],
        ["sum"],
    )

workspace.FeedBlob(
    "X",
    np.asarray(1.0).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.FeedBlob(
    "Y",
    np.asarray(2.0).astype(np.float32),
    device_option=device_opt,
)
print("Y:\n", workspace.FetchBlob("Y"))

workspace.RunOperatorOnce(op)
print("sum:\n", workspace.FetchBlob("sum"))
"""


###############################################################################
# net_test_11
"""
model = model_helper.ModelHelper(name='rfft_gradient_test')
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=2)
with core.DeviceScope(device_opt):
    # add input & model
    X_re = model.net.GivenTensorFill(
        [],
        ["X_re"],
        # values = np.arange(1, 25).reshape(3, 8).astype(np.float32),
        # shape=[3, 8],
        # values = np.arange(1, 7).reshape(3, 2).astype(np.float32),
        # shape=[3, 2],
        values = np.arange(1, 13).reshape(3, 4).astype(np.float32),
        shape=[3, 4],
        # values = np.arange(1, 3073).reshape(3, 1024).astype(np.float32),
        # shape=[3, 1024],
    )

    # architecture-1
    # Y_re_origin, Y_im_origin = model.net.RFFT(
    #     ["X_re"],
    #     ["Y_re_origin", "Y_im_origin"],
    # )
    # Y_re, _ = model.net.Split(
    #     [Y_re_origin],
    #     ["Y_re", "sa"],
    #     split=(3, 1),
    #     axis=1,
    # )
    # Y_im, _ = model.net.Split(
    #     [Y_im_origin],
    #     ["Y_im", "sb"],
    #     split = (3, 1),
    #     axis = 1,
    # )

    # architecture-2
    Y_re, Y_im= model.net.RFFT(
        ["X_re"],
        ["Y_re", "Y_im"],
    )
    Y_re_sum = model.net.SumElements(
        [Y_re],
        ["Y_re_sum"],
    )
    Y_im_sum = model.net.SumElements(
        [Y_im],
        ["Y_im_sum"],
    )
    Y_sum = model.net.Sum(
        [Y_re_sum, Y_im_sum],
        ["Y_sum"],
    )

    # add loss
    model.AddGradientOperators([Y_sum])

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)
print(workspace.Blobs())
print("Y_re:\n {}".format(workspace.FetchBlob("Y_re")))
print("Y_im:\n {}".format(workspace.FetchBlob("Y_im")))
print("X_re_grad:\n {}".format(workspace.FetchBlob("X_re_grad")))
"""


###############################################################################
# op_test_44
'''
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "FFT",
        ["X_real", "X_imag"],
        ["Y_real", "Y_imag"],
    )

workspace.FeedBlob(
    "X_real",
    np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    device_option=device_opt,
)
print("X_real:\n", workspace.FetchBlob("X_real"))
workspace.FeedBlob(
    "X_imag",
    np.zeros((3, 4)).astype(np.float32),
    device_option=device_opt,
)
print("X_imag:\n", workspace.FetchBlob("X_imag"))

workspace.RunOperatorOnce(op)
print("Y_real:\n", workspace.FetchBlob("Y_real"))
print("Y_imag:\n", workspace.FetchBlob("Y_imag"))
'''


###############################################################################
# net_test_12
"""
model = model_helper.ModelHelper(name='fft_gradient_test')
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=2)
with core.DeviceScope(device_opt):
    # add input & model
    X_re = model.net.GivenTensorFill(
        [],
        ["X_re"],
        # values = np.arange(1, 25).reshape(3, 8).astype(np.float32),
        # shape=[3, 8],
        # values = np.arange(1, 7).reshape(3, 2).astype(np.float32),
        # shape=[3, 2],
        values = np.arange(1, 13).reshape(3, 4).astype(np.float32),
        shape=[3, 4],
        # values = np.arange(1, 3073).reshape(3, 1024).astype(np.float32),
        # shape=[3, 1024],
    )
    X_im = model.net.GivenTensorFill(
        [],
        ["X_im"],
        values = np.zeros((3, 4)).astype(np.float32),
        shape=[3, 4],
    )

    Y_re, Y_im= model.net.FFT(
        ["X_re", "X_im"],
        ["Y_re", "Y_im"],
    )
    Y_re_sum = model.net.SumElements(
        [Y_re],
        ["Y_re_sum"],
    )
    Y_im_sum = model.net.SumElements(
        [Y_im],
        ["Y_im_sum"],
    )
    Y_sum = model.net.Sum(
        [Y_re_sum, Y_im_sum],
        ["Y_sum"],
    )

    # add loss
    model.AddGradientOperators([Y_sum])

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)
print(workspace.Blobs())
print("Y_re:\n {}".format(workspace.FetchBlob("Y_re")))
print("Y_im:\n {}".format(workspace.FetchBlob("Y_im")))
print("X_re_grad:\n {}".format(workspace.FetchBlob("X_re_grad")))
print("X_im_grad:\n {}".format(workspace.FetchBlob("X_im_grad")))
"""


###############################################################################
# op_test_45
"""
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=2)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "IFFT",
        ["X_real", "X_imag"],
        ["Y_real", "Y_imag"],
    )

workspace.FeedBlob(
    "X_real",
    np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    device_option=device_opt,
)
print("X_real:\n", workspace.FetchBlob("X_real"))
workspace.FeedBlob(
    "X_imag",
    np.zeros((3, 4)).astype(np.float32),
    device_option=device_opt,
)
print("X_imag:\n", workspace.FetchBlob("X_imag"))

workspace.RunOperatorOnce(op)
print("Y_real:\n", workspace.FetchBlob("Y_real"))
print("Y_imag:\n", workspace.FetchBlob("Y_imag"))
"""


###############################################################################
# net_test_13
'''
model = model_helper.ModelHelper(name='ifft_gradient_test')
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    # add input & model
    X_re = model.net.GivenTensorFill(
        [],
        ["X_re"],
        # values = np.arange(1, 25).reshape(3, 8).astype(np.float32),
        # shape=[3, 8],
        # values = np.arange(1, 7).reshape(3, 2).astype(np.float32),
        # shape=[3, 2],
        # values = np.arange(1, 13).reshape(3, 4).astype(np.float32),
        # shape=[3, 4],
        values = np.arange(1, 3073).reshape(3, 1024).astype(np.float32),
        shape=[3, 1024],
    )
    X_im = model.net.GivenTensorFill(
        [],
        ["X_im"],
        # values = np.zeros((3, 4)).astype(np.float32),
        # shape=[3, 4],
        values = np.zeros((3, 1024)).astype(np.float32),
        shape=[3, 1024],
    )

    Y_re, Y_im= model.net.IFFT(
        ["X_re", "X_im"],
        ["Y_re", "Y_im"],
    )
    Y_re_sum = model.net.SumElements(
        [Y_re],
        ["Y_re_sum"],
    )
    model.net.ZeroGradient([Y_im], [])
    model.AddGradientOperators([Y_re_sum])
    # Y_im_sum = model.net.SumElements(
    #     [Y_im],
    #     ["Y_im_sum"],
    # )
    # Y_sum = model.net.Sum(
    #     [Y_re_sum, Y_im_sum],
    #     ["Y_sum"],
    # )

    # # add loss
    # model.AddGradientOperators([Y_sum])

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)
print(workspace.Blobs())
print("X_re:\n {}".format(workspace.FetchBlob("X_re")))
print("X_im:\n {}".format(workspace.FetchBlob("X_im")))
print("Y_re:\n {}".format(workspace.FetchBlob("Y_re")))
print("Y_im:\n {}".format(workspace.FetchBlob("Y_im")))
print("X_re_grad:\n {}".format(workspace.FetchBlob("X_re_grad")))
print("X_im_grad:\n {}".format(workspace.FetchBlob("X_im_grad")))
'''


###############################################################################
# op_test_46
"""
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=2)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "ZerosLike",
        ["X"],
        ["Y"],
    )

workspace.FeedBlob(
    "X",
    np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
"""


###############################################################################
# op_test_47
'''
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=2)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Sign",
        ["X"],
        ["Sign"],
    )

workspace.FeedBlob(
    "X",
    # np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    np.random.randint(10, size=(3,4)).astype(np.float32) - 5.0,
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("Sign:\n", workspace.FetchBlob("Sign"))
'''


###############################################################################
# op_test_48
'''
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=1)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Abs",
        ["X"],
        ["abs"],
    )

workspace.FeedBlob(
    "X",
    # np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    np.random.randint(10, size=(3,4)).astype(np.float32) - 5.0,
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("abs:\n", workspace.FetchBlob("abs"))
'''


###############################################################################
# op_test_48
'''
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=1)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Abs",
        ["X"],
        ["abs"],
    )

workspace.FeedBlob(
    "X",
    # np.arange(1, 13).reshape(3,4).astype(np.float32),
    # np.arange(1, 204801).reshape(200,1024).astype(np.float32),
    np.random.randint(10, size=(3,4)).astype(np.float32) - 5.0,
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("abs:\n", workspace.FetchBlob("abs"))
'''


###############################################################################
# op_test_49
'''
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "SoftmaxWithLoss",
        ["X", "label"],
        ["softmax", "loss"],
    )

workspace.FeedBlob(
    "X",
    np.asarray([[1,2,3,4,5,6,7,8,9,9]]).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))
workspace.FeedBlob(
    "label",
    np.asarray([1]).astype(np.int32),
    device_option=device_opt,
)
print("label:\n", workspace.FetchBlob("label"))

workspace.RunOperatorOnce(op)
print("softmax:\n", workspace.FetchBlob("softmax"))
print("loss:\n", workspace.FetchBlob("loss"))
'''


###############################################################################
# op_test_50
'''
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "SoftmaxWithLoss",
        ["X", "label"],
        ["softmax", "loss"],
    )

workspace.FeedBlob(
    "X",
    np.asarray([[1,2,3,4,5,6,7,8,9,9],
                [1,2,3,4,5,6,7,8,9,9],
                ]).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))
workspace.FeedBlob(
    "label",
    # np.asarray([0, 0]).reshape(2, 1).astype(np.int32),
    np.asarray([0, 1]).astype(np.int32),
    device_option=device_opt,
)
print("label:\n", workspace.FetchBlob("label"))

workspace.RunOperatorOnce(op)
print("softmax:\n", workspace.FetchBlob("softmax"))
print("loss:\n", workspace.FetchBlob("loss"))
'''


###############################################################################
# net_test_14
'''
model = model_helper.ModelHelper(name='norm_test')
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    X = model.net.GivenTensorFill(
        [],
        ["X"],
        # values=np.arange(1, 25).reshape(4,6).astype(np.float32),
        values=np.ones((4, 6)).astype(np.float32),
        shape=[4, 6],
    )

    norm = model.net.Normalize(
        [X],
        ['X_norm'],
        # axis=0,
        axis=1,
    )

    norm_sum = model.net.SumElements(
        [norm],
        ["norm_sum"],
    )

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)
print(workspace.Blobs())
print("X:\n {}".format(workspace.FetchBlob("X")))
print("X_norm:\n {}".format(workspace.FetchBlob("X_norm")))
print("norm_sum:\n {}".format(workspace.FetchBlob("norm_sum")))
'''


###############################################################################
# net_test_15
'''
model = model_helper.ModelHelper(name='div_test')
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    X = model.net.GivenTensorFill(
        [],
        ["X"],
        # values=np.asarray(10).astype(np.float32),
        # shape=(),
        values=np.arange(1, 13).reshape(3, 4).astype(np.float32),
        shape=(3, 4),
    )

    factor = model.net.GivenTensorFill(
        [],
        ["factor"],
        values=2 * np.ones((3, 4)).astype(np.float32),
        shape=(3, 4),
    )

    Y = model.net.Div(
        [X, factor],
        ["Y"],
        # broadcast=1,
        # axis=0,
    )

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)
print(workspace.Blobs())
print("X:\n {}".format(workspace.FetchBlob("X")))
print("factor:\n {}".format(workspace.FetchBlob("factor")))
print("Y:\n {}".format(workspace.FetchBlob("Y")))
'''


###############################################################################
# op_test_51
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "SquaredL2Distance",
        ["X", "Y"],
        ["dis"],
    )

workspace.FeedBlob(
    "X",
    -1 * np.arange(1, 25).reshape(4,6).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))
workspace.FeedBlob(
    "Y",
    np.zeros((4,6)).astype(np.float32),
    device_option=device_opt,
)
print("Y:\n", workspace.FetchBlob("Y"))

workspace.RunOperatorOnce(op)
print("dis:\n", workspace.FetchBlob("dis"))
'''


###############################################################################
# net_test_16
'''
model = model_helper.ModelHelper(name='confusion_test')
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=3)
with core.DeviceScope(device_opt):
    pred = model.net.GivenTensorFill(
        [],
        ["pred"],
        values=np.arange(1, 25).reshape(4, 6).astype(np.float32),
        shape=(4, 6),
    )
    left, right = model.net.Split(
        [pred],
        ['left', 'right'],
        split=(2, 2),
        axis=0,
    )
    diff = model.net.Sub(
        [left, right],
        ['diff'],
        broadcast=1,
        axis=0,
    )
    diff_abs = model.net.Abs(diff, 'diff_abs')
    diff_square = model.net.Mul(
        [diff_abs, diff_abs],
        ['diff_square'],
        broadcast=1,
        axis=0,
    )
    diff_square_sum = model.net.ReduceBackSum(
        [diff_square],
        ['diff_square_sum'],
    )
    diff_sqrt = model.net.Sqrt(
        [diff_square_sum],
        ['diff_sqrt'],
    )
    diff_norm2 = model.net.ReduceBackSum(
        [diff_sqrt],
        ['diff_norm2'],
    )
    BATCHSIZE = model.net.GivenTensorFill(
        [],
        ['BATCHSIZE'],
        values=np.asarray(4).astype(np.float32),
        shape=(),
    )
    pc_reg = model.net.Div(
        [diff_norm2, BATCHSIZE],
        ['pc_reg'],
        # broadcast=1,
        # axis=0,
    )

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)
print(workspace.Blobs())
print("pred:\n {}".format(workspace.FetchBlob("pred")))
print("pc_reg:\n {}".format(workspace.FetchBlob("pc_reg")))
'''


###############################################################################
# op_test_52
'''
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "Pow",
        ["X", "exp"],
        ["Y"],
        broadcast=1,
    )

workspace.FeedBlob(
    "X",
    np.asarray(2).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))
workspace.FeedBlob(
    "exp",
    2* np.ones((4,6)).astype(np.float32),
    device_option=device_opt,
)
print("exp:\n", workspace.FetchBlob("exp"))

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
'''


###############################################################################
# op_test_53
'''
print("="*100)
print("Running caffe2 operator {}".format("TopK"))
workspace.ResetWorkspace()
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "TopK",
        ["X"],
        ["values", "indices", "flat_indices"],
        k = 3,
    )

workspace.FeedBlob(
    "X",
    # np.asarray(2).astype(np.float32),
    np.random.randint(100, size=(4, 5)).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("values:\n", workspace.FetchBlob("values"))
print("indices:\n", workspace.FetchBlob("indices"))
print("flat_indices:\n", workspace.FetchBlob("flat_indices"))
'''


###############################################################################
# op_test_54
'''
print("="*100)
print("Running caffe2 operator {}".format("TopKGradHook"))
workspace.ResetWorkspace()
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
with core.DeviceScope(device_opt):
    op = core.CreateOperator(
        "TopKGradHook",
        ["X"],
        ["Y"],
        k = 3,
    )

workspace.FeedBlob(
    "X",
    # np.asarray(2).astype(np.float32),
    np.random.randint(100, size=(4, 5)).astype(np.float32),
    device_option=device_opt,
)
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
'''


###############################################################################
# net_test_17
model = model_helper.ModelHelper(name='topk_grad_hook_grad_test')
# device_opt = core.DeviceOption(device_type=caffe2_pb2.CPU)
device_opt = core.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=0)
with core.DeviceScope(device_opt):
    # add pre-set grads
    Y_grad = model.param_init_net.GivenTensorFill(
        [],
        ["Y_grad"],
        # values = np.random.randint(10, size=(4, 5)).astype(np.float32) - 5,
        # shape=[4, 5],

        values = np.random.randint(10, size=(2, 4, 2, 3)).astype(np.float32) - 5,
        shape=[2, 4, 2, 3],

        # TopK (k == 20)
        # values = np.random.randint(100, size=(32, 20, 28, 28)).astype(np.float32),
        # shape=[32, 20, 28, 28],

        # values = np.random.randint(100, size=(32, 1024, 28, 28)).astype(np.float32),
        # shape=[32, 1024, 28, 28],

        # values = 3 * np.ones((2, 4, 2, 3)).astype(np.float32),
        # shape=[2, 4, 2, 3],

        # values = np.arange(1, 49).reshape(2, 4, 2, 3).astype(np.float32),
        # shape=[2, 4, 2, 3],

        # TopK (k == 2)
        # values = np.random.randint(10, size=(2, 2, 2, 3)).astype(np.float32),
        # shape=[2, 2, 2, 3],
    )

    # add input & model
    X = model.net.GivenTensorFill(
        [],
        ["X"],
        # values = np.arange(1, 21).reshape(4, 5).astype(np.float32),
        # shape=[4, 5],

        # values = np.arange(1, 25).reshape(2, 2, 2, 3).astype(np.float32),
        # values = 2*np.ones((2, 2, 2, 3)).astype(np.float32),

        values = 3*np.ones((2, 4, 2, 3)).astype(np.float32),
        shape=[2, 4, 2, 3],

        # values = 3*np.ones((32, 1024, 28, 28)).astype(np.float32),
        # shape=[32, 1024, 28, 28],

        # values = 2*np.ones((2, 2, 2, 3)).astype(np.float32),
        # values = np.random.randint(30, size=(3,2,1,3)).astype(np.float32),
        # shape=[3, 2, 1, 3],
    )

    Y = model.net.TopKGradHook(
        ["X"],
        ["Y"],
        # k = 20,
        k = 2,
    )

    # Y, indices, flat_indices = model.net.TopK(
    #     ["X"],
    #     # ["values", "indices", "flat_indices"],
    #     ["Y", "indices", "flat_indices"],
    #     k = 20,
    # )

    # model.AddGradientOperators([Y])
    model.AddGradientOperators({
        Y : Y_grad,
    })


workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)
workspace.RunNet(model.net)

# time_beg = time.time()
# for i in range(50):
#     print("i: {}".format(i))
#     workspace.RunNet(model.net)
# time_end = time.time()
# print("total running time: {:.4f}s".format(time_end - time_beg))

print("X:\n {}".format(workspace.FetchBlob("X")))
print("Y:\n {}".format(workspace.FetchBlob("Y")))
# print("Y_sum:\n {}".format(workspace.FetchBlob("Y_sum")))
# print("Y_sum_autogen_grad:\n {}".format(workspace.FetchBlob("Y_sum_autogen_grad")))
print("Y_grad:\n {}".format(workspace.FetchBlob("Y_grad")))
print("X_grad:\n {}".format(workspace.FetchBlob("X_grad")))
print("*"*20)
print("all blobs")
for blob in workspace.Blobs():
    print(blob)
print("*"*20)






