from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import os
import cv2
import lmdb
import sys

from caffe2.python import workspace, model_helper, core, brew
from caffe2.proto import caffe2_pb2



# define vgg-type model structure
def meta_conv(
    model,
    inputs,
    dim_in=64,
    dim_out=64,
    kernel=3,
    pad=1,
    stride=1,
    module_seq=None,
    conv_seq=None,
):
    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        'conv{}_{}'.format(module_seq, conv_seq),
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        stride=stride,
        pad=pad
    )

    # ReLU layer
    relu = brew.relu(
        model,
        conv,
        conv # in-place
    )
    return relu


def conv_module_1(model, inputs, dim_in, dim_out, module_seq):
    m_conv1 = meta_conv(
        model,
        inputs,
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=3,
        pad=1,
        stride=1,
        module_seq=module_seq,
        conv_seq='1'
    )
    m_conv2 = meta_conv(
        model,
        m_conv1,
        dim_in=dim_out,
        dim_out=dim_out,
        kernel=3,
        pad=1,
        stride=1,
        module_seq=module_seq,
        conv_seq='2'
    )
    pool = brew.max_pool(
        model,
        m_conv2,
        'pool{}'.format(module_seq),
        kernel=2,
        stride=2
    )
    return pool


def conv_module_2(model, inputs, dim_in, dim_out, module_seq):
    m_conv1 = meta_conv(
        model,
        inputs,
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=3,
        pad=1,
        stride=1,
        module_seq=module_seq,
        conv_seq='1'
    )
    m_conv2 = meta_conv(
        model,
        m_conv1,
        dim_in=dim_out,
        dim_out=dim_out,
        kernel=3,
        pad=1,
        stride=1,
        module_seq=module_seq,
        conv_seq='2'
    )
    m_conv3 = meta_conv(
        model,
        m_conv2,
        dim_in=dim_out,
        dim_out=dim_out,
        kernel=3,
        pad=1,
        stride=1,
        module_seq=module_seq,
        conv_seq='3'
    )
    pool = brew.max_pool(
        model,
        m_conv3,
        'pool{}'.format(module_seq),
        kernel=2,
        stride=2
    )
    return pool


def fc_module(model, inputs, dim_in, dim_out, module_seq, dropout, is_test):
    fc = brew.fc(
        model,
        inputs,
        'fc{}'.format(module_seq),
        dim_in=dim_in,
        dim_out=dim_out
    )
    relu = brew.relu(
        model,
        fc,
        fc
    )
    dropout = brew.dropout(
        model,
        relu,
        'pool{}'.format(module_seq),
        is_test=is_test,
        ratio=dropout
    )
    return dropout


def add_vgg16_finetune(model, data, num_class=2, is_test=False):
    '''
    construct vgg16 net for finetune
    default from 'data' to 'pool5' stay unchanged
    '''
    conv1 = conv_module_1(model, data, dim_in=3, dim_out=64, module_seq='1')
    conv2 = conv_module_1(model, conv1, dim_in=64, dim_out=128, module_seq='2')
    conv3 = conv_module_2(model, conv2, dim_in=128, dim_out=256, module_seq='3')
    conv4 = conv_module_2(model, conv3, dim_in=256, dim_out=512, module_seq='4')
    conv5 = conv_module_2(model, conv4, dim_in=512, dim_out=512, module_seq='5')  # blob 'pool5'
    fc1 = fc_module(model, conv5, dim_in=512*7*7, dim_out=4096, module_seq='6', dropout=0.5, is_test=is_test)
    fc2 = fc_module(model, fc1, dim_in=4096, dim_out=4096, module_seq='7', dropout=0.5, is_test=is_test)
    # finetune part
    finetune_fc = brew.fc(model, fc2, 'finetune_fc', dim_in=4096, dim_out=num_class)
    softmax = brew.softmax(model, finetune_fc, 'softmax')
    return softmax


def add_vgg16(model, data, is_test=False):
    '''
    construct vgg16 net
    '''
    conv1 = conv_module_1(model, data, dim_in=3, dim_out=64, module_seq='1')
    conv2 = conv_module_1(model, conv1, dim_in=64, dim_out=128, module_seq='2')
    conv3 = conv_module_2(model, conv2, dim_in=128, dim_out=256, module_seq='3')
    conv4 = conv_module_2(model, conv3, dim_in=256, dim_out=512, module_seq='4')
    conv5 = conv_module_2(model, conv4, dim_in=512, dim_out=512, module_seq='5')
    fc1 = fc_module(model, conv5, dim_in=512*7*7, dim_out=4096, module_seq='6', dropout=0.5, is_test=is_test)
    fc2 = fc_module(model, fc1, dim_in=4096, dim_out=4096, module_seq='7', dropout=0.5, is_test=is_test)
    fc3 = brew.fc(model, fc2, 'fc8', dim_in=4096, dim_out=1000)
    softmax = brew.softmax(model, fc3, 'softmax')
    return softmax


def add_vgg16_core(model, data, is_test=False):
    ''' construct vgg16 core for finetune, default remove last fc
    Args:
        model: model_helper instance
        data: 'data' BlobRef
        is_test: bool denotes training or testing model
    Returns:
        core_output: BlobRef of the output of the core net
        dim_out: a int32 of the dim of the core_output
    '''
    conv1 = conv_module_1(model, data, dim_in=3, dim_out=64, module_seq='1')
    conv2 = conv_module_1(model, conv1, dim_in=64, dim_out=128, module_seq='2')
    conv3 = conv_module_2(model, conv2, dim_in=128, dim_out=256, module_seq='3')
    conv4 = conv_module_2(model, conv3, dim_in=256, dim_out=512, module_seq='4')
    conv5 = conv_module_2(model, conv4, dim_in=512, dim_out=512, module_seq='5')
    fc1 = fc_module(model, conv5, dim_in=512*7*7, dim_out=4096, module_seq='6', dropout=0.5, is_test=is_test)
    fc2 = fc_module(model, fc1, dim_in=4096, dim_out=4096, module_seq='7', dropout=0.5, is_test=is_test)
    return fc2, 4096


if __name__ == '__main__':
    model = model_helper.ModelHelper('test_structure')
    data = model.net.ConstantFill([], ['data'], shape=[10, 3, 244, 244], value=1.0)
    add_vgg16_finetune(model, data)

    workspace.RunNetOnce(model.param_init_net)
    for index, blob in enumerate(workspace.Blobs()):
        print("index:{}, blob:{}".format(index, workspace.FetchBlob(blob)))

    # load param_init_net
    init_net_pb = '/home/zhibin/qzhong/caffe2/caffe2_model_zoo/vgg16/vgg16_init_net.pb'
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'rb') as f:
        init_net_proto.ParseFromString(f.read())
    workspace.RunNetOnce(core.Net(init_net_proto))
    for index, blob in enumerate(workspace.Blobs()):
        print("index:{}, blob:{}".format(index, workspace.FetchBlob(blob)))
        # print("index:{}, blob:{}".format(index, blob))





