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
    net_drawer
)
from caffe2.proto import caffe2_pb2

from model_utils import load_model, save_model, save_checkpoint
from add_resnet50_model import add_resnet50_core
from add_vgg16_model import add_vgg16_core



# simple inference model class
class InferenceModel(object):
    model_construction_function = {
        'resnet50' : add_resnet50_core,
        'vgg16' : add_vgg16_core,
    }

    def __init__(self, args):
        # task params
        self.task_name = args.task_name
        self.num_class = args.num_class
        self.label_type = args.label_type
        self.num_labels = self.num_class if self.label_type == 1 else 0

        # data params
        self.testing_lmdb_path = args.test_data
        self.db_type = args.db_type
        self.mean_per_channel = args.mean_per_channel
        self.std_per_channel = args.std_per_channel
        self.scale = args.scale
        self.crop = args.crop
        self.testing_num = args.testing_num

        # pretrained model net params
        self.pretrained_model_name = args.pretrained_model_name
        self.pretrained_model_folder = args.pretrained_model_folder
        self.init_net_pb = os.path.join(self.pretrained_model_folder,
                                        args.init_net)
        self.predict_net_pb = os.path.join(self.pretrained_model_folder,
                                           args.predict_net)

        # testing params
        self.root_folder = args.root_folder
        self.use_gpu_transform = args.use_gpu_transform
        self.gpu_id = args.gpu_id
        self.arg_scope = args.arg_scope
        self.batch_size = args.batch_size
        self.testing_iterations = args.testing_iterations


    def add_input(self, model, db_path, is_test=True):
        """
        Add an database input data
        """
        assert(is_test == True)

        db_reader = model.CreateDB(
            "test_db_reader",
            # "val_db_reader",
            db=db_path,
            db_type=self.db_type
        )
        data, label = brew.image_input(
            model,
            db_reader,
            ['data', 'label'],
            batch_size=self.batch_size,
            use_gpu_transform=self.use_gpu_transform,
            scale=self.scale,
            crop=self.crop,
            mean_per_channel=[eval(item) for item in \
                              self.mean_per_channel.split(',')],
            std_per_channel=[eval(item) for item in \
                             self.std_per_channel.split(',')],
            mirror=True,
            is_test=is_test,
            label_type=self.label_type,
            num_labels=self.num_labels,
        )


    def add_model_1(self, model, data, is_test=True):
        '''
        add no net, building param_init_net & predict_net from scratch
        '''
        # add core structure
        func = InferenceModel.model_construction_function[self.pretrained_model_name]
        core_out, core_dim = func(model, data, is_test)

        # add inference classification
        brew.fc(model, core_out, 'classifier', dim_in=core_dim,
                dim_out=self.num_class)
        softmax = brew.softmax(model, 'classifier', 'softmax')


    def add_model_2(self, model, init_net_pb, predict_net_pb):
        '''
        add param_init_net & predict_net
        '''
        load_model(model, init_net_pb, predict_net_pb)


    def load_init_params(self, init_net_pb, device_opt):
        '''
        load input params on given device
        '''
        init_net_proto = caffe2_pb2.NetDef()
        with open(init_net_pb, 'rb') as f:
            init_net_proto.ParseFromString(f.read())
            for op in init_net_proto.op:
                op.device_option.CopyFrom(device_opt)
        workspace.RunNetOnce(core.Net(init_net_proto))


    def initialize(self):
        '''
        1. do the sanity check for path
        2. initialize workspace
        '''
        print("[INFO] start initialize for inference")
        if not os.path.exists(self.root_folder):
            raise ValueError("Root folder does not exist")
        if not os.path.exists(self.pretrained_model_folder):
            raise ValueError("Pretrained model does not exist")
        if not os.path.exists(self.init_net_pb):
            raise ValueError("Pretrained init_net does not exist")
        if not os.path.exists(self.predict_net_pb):
            raise ValueError("Pretrained predict_net does not exist")

        workspace.ResetWorkspace(self.root_folder)
        print("[INFO] initialize over...")


    def prepare_testing_model(self):
        print("[INFO] start preparing testing model")
        # set device
        device_opt = caffe2_pb2.DeviceOption()
        if self.gpu_id is not None:
            device_opt.device_type = caffe2_pb2.CUDA
            device_opt.cuda_gpu_id = int(self.gpu_id)

        # build model
        with core.DeviceScope(device_opt):
            self.testing_model = model_helper.ModelHelper(
                name = '{}_{}_testing_model'.format(
                    self.task_name,
                    self.pretrained_model_name,
                ),
                arg_scope = self.arg_scope,
                init_params = False,
            )
            # add input
            self.add_input(self.testing_model, self.testing_lmdb_path)

            # add model structure
            self.add_model_1(self.testing_model, 'data')
            # self.add_model_2(self.testing_model, self.init_net_pb, self.predict_net_pb)

        # workspace running
        self.load_init_params(self.init_net_pb, device_opt)  # must be placed here
        workspace.RunNetOnce(self.testing_model.param_init_net)
        workspace.CreateNet(self.testing_model.net)
        print("[INFO] prepare testing model over...")


    def plot_history(self, results, name, time):
        plt.figure(figsize=(12, 8))
        plt.title(name)
        plt.plot(range(1, 1 + len(results)), results)
        plt.draw()
        file_path = os.path.join(self.root_folder, '{}_{}.png'.format(name, time))
        plt.savefig(file_path)


    def forward(self):
        '''
        model testing
        '''
        # init
        self.initialize()

        # build model
        self.prepare_testing_model()

        # testing
        print("[INFO] start testing task {}".format(self.task_name))
        testing_start = time.time()
        for iters in range(1, 1 + self.testing_iterations):
            iter_start = time.time()
            workspace.RunNet(self.testing_model.net)
            iter_end = time.time()

            softmax = workspace.FetchBlob('softmax')
            max_index = [np.argmax(row) for row in softmax]
            max_features = []
            for row in softmax:
                row_mean = np.mean(row)
                row_feat = [i for i, val in enumerate(row) if val > row_mean]
                if 12 in row_feat:
                    row_feat = [12]
                max_features.append(row_feat)

            label = workspace.FetchBlob('label')
            print("============================================================================")
            if self.task_name != 'feature':
                print("[TESTING] iteration:{}/{} \nmax_index:{} \nlabel: {}  \ntime:{:.3f}s".format(
                    iters, self.testing_iterations, max_index, label, iter_end - iter_start)
                )
            else:
                print("[TESTING] iteration:{}/{} \nmax_features:{} \nlabel: {}  \ntime:{:.3f}s".format(
                    iters, self.testing_iterations, max_features, label, iter_end - iter_start)
                )

        testing_end = time.time()
        print("============================================================================")
        print("[INFO] Total testing time is {:.3f}s".format(testing_end - testing_start))
        print("[INFO] Inference model for task {} over...".format(self.task_name))



def parse_args():
    # load config file form .yaml
    config_parser = argparse.ArgumentParser(
        description='MAFAT model-inference config parser',
        add_help=False,
    )
    config_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help = 'config file',
    )
    args, _ = config_parser.parse_known_args()
    with open(args.config) as f:
        config = yaml.load(f)
        config_parser.set_defaults(**config)

    # parse rest testing specific arguments
    args_parser = argparse.ArgumentParser(
        description='MAFAT model-inference cmdline parser',
        parents=[config_parser],
    )
    args_parser.add_argument("--gpu_id", type=int,
                             help="which gpu to use")
    args_parser.add_argument("--batch_size", type=int,
                             help="batch size of each iterations")
    args_parser.add_argument("--testing_iterations", type=int, default=500,
                             help="total iterations for testing")

    args = args_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_model = InferenceModel(args)
    test_model.forward()





