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
from add_vgg16_model import add_vgg16_finetune
from add_resnet50_model import add_resnet50_finetune


model_structures = {
    'vgg16' : add_vgg16_finetune,
    'resnet50' : add_resnet50_finetune,
}

# simple finetune model class
class FinetuneModel(object):
    def __init__(self, args):
        # data params
        self.dataset_name = args.dataset_name
        self.training_lmdb_path = args.train_data
        self.validation_lmdb_path = args.val_data
        self.db_type = args.db_type
        self.mean_per_channel = args.mean_per_channel
        self.std_per_channel = args.std_per_channel
        self.scale = args.scale
        self.crop = args.crop
        self.training_num = args.training_num
        self.validation_num = args.validation_num

        # pretrained model params
        self.pretrained_model_name = args.pretrained_model_name
        self.pretrained_model_dir = args.pretrained_model_dir
        self.init_net_pb = os.path.join(self.pretrained_model_dir, args.init_net)
        self.predict_net_pb = os.path.join(self.pretrained_model_dir, args.predict_net)

        # training params
        self.arg_scope = {"order" : "NCHW"}
        self.root_folder = args.root_folder
        self.models_folder = os.path.join(args.root_folder, args.models_folder)
        self.checkpoints_folder = os.path.join(args.root_folder, args.checkpoints_folder)
        self.batch_size = args.batch_size
        self.total_epochs = args.total_epochs
        self.validation_period = args.validation_period
        self.checkpoint_peroid = args.checkpoint_peroid
        self.weight_decay = args.weight_decay
        self.base_lr = args.base_lr
        self.momentum = args.momentum
        self.policy = args.policy
        self.max_iter = args.max_iter
        self.stepsize = args.stepsize
        self.gamma = args.gamma
        self.nesterov = args.nesterov
        self.use_gpu_transform = args.use_gpu_transform
        self.gpu_id = args.gpu_id


    def add_input(self, model, db_path, is_test=False):
        """
        Add an database input data
        """
        db_reader = model.CreateDB(
            "val_db_reader" if is_test else 'train_db_reader',
            # 'db_reader',
            db=db_path,
            db_type=self.db_type
        )
        data, label = brew.image_input(
            model,
            db_reader,
            ['data', 'label'],
            batch_size=self.batch_size,
            use_gpu_transform=self.use_gpu_transform,
            # use_caffe_datum=True,
            scale=self.scale,
            crop=self.crop,
            mean_per_channel=[eval(item) for item in self.mean_per_channel.split(',')],
            std_per_channel=[eval(item) for item in self.std_per_channel.split(',')],
            mirror=True,
            is_test=is_test,
        )
        data = model.StopGradient(data, data)
        return data, label


    def add_training_operators(self, model):
        ''' compute loss '''
        # compute cropss entropy between softmax & label
        xent = model.LabelCrossEntropy(['softmax', 'label'], 'xent')
        # Compute the expected loss
        loss = model.AveragedLoss(xent, "loss")
        # add gradient operators to the model
        model.AddGradientOperators([loss])

        ''' configure sgd optimization params'''
        # add weight decay
        optimizer.add_weight_decay(model, self.weight_decay)
        # training policy for imagenet
        optimizer.build_multi_precision_sgd(
            model,
            base_learning_rate = self.base_lr,
            policy = self.policy,
            #max_iter = self.max_iter,
            stepsize = self.stepsize * (self.training_num // self.batch_size),
            momentum = self.momentum,
            gamma = self.gamma,
            nesterov = self.nesterov,
        )


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


    def _load_init_params(self, init_net_pb, device_opt):
        init_net_proto = caffe2_pb2.NetDef()
        with open(init_net_pb, 'rb') as f:
            init_net_proto.ParseFromString(f.read())
        self.training_model.param_init_net.AppendNet(core.Net(init_net_proto))


    def add_model(self, model, data, is_test=False):
        # add_vgg16_finetune(model, data, is_test)
        model_structures[self.pretrained_model_name](model, data, is_test)


    def prepare_training_model(self):
        # set device
        device_opt = caffe2_pb2.DeviceOption()
        if self.gpu_id is not None:
            device_opt.device_type = caffe2_pb2.CUDA
            device_opt.cuda_gpu_id = int(self.gpu_id)

        # build model
        with core.DeviceScope(device_opt):
            self.training_model = model_helper.ModelHelper(
                name = '{}_{}_training_model'.format(
                    self.pretrained_model_name,
                    self.dataset_name
                ),
                arg_scope = self.arg_scope
            )
            # add input
            self.add_input(self.training_model, self.training_lmdb_path)
            # add model structure
            self.add_model(self.training_model, 'data')
            # add BP
            self.add_training_operators(self.training_model)
            # add accuracy
            brew.accuracy(self.training_model, ['softmax', 'label'], "accuracy")

        # workspace running
        workspace.RunNetOnce(self.training_model.param_init_net)
        self.load_init_params(self.init_net_pb, device_opt)
        workspace.CreateNet(self.training_model.net)


    def prepare_validation_model(self):
        # set device
        device_opt = caffe2_pb2.DeviceOption()
        if self.gpu_id is not None:
            device_opt.device_type = caffe2_pb2.CUDA
            device_opt.cuda_gpu_id = int(self.gpu_id)

        # build model
        with core.DeviceScope(device_opt):
            self.validation_model = model_helper.ModelHelper(
                name = '{}_{}_validation_model'.format(
                    self.pretrained_model_name,
                    self.dataset_name
                ),
                arg_scope = self.arg_scope,
                init_params = False
            )
            # add input
            self.add_input(self.validation_model, self.validation_lmdb_path, is_test=True)
            # add model structure
            self.add_model(self.validation_model, 'data', is_test=True)
            # add accuracy
            brew.accuracy(self.validation_model, ['softmax', 'label'], "accuracy")

        # workspace running
        workspace.RunNetOnce(self.validation_model.param_init_net)
        workspace.CreateNet(self.validation_model.net)


    def finetune(self):
        '''
        finetune training
        '''
        # init
        workspace.ResetWorkspace(self.root_folder)

        # build model
        self.prepare_training_model()
        self.prepare_validation_model()

        # training & validating
        training_start = time.time()
        for epoch in range(1, 1 + self.total_epochs):
            # model training
            epoch_start = time.time()
            iters_per_epoch = self.training_num // self.batch_size
            avg_acc = 0
            for iters in range(1, 1 + iters_per_epoch):
                iter_start = time.time()
                workspace.RunNet(self.training_model.net)
                iter_end = time.time()
                print("============================================================================")
                accuracy = workspace.FetchBlob('accuracy')
                loss = workspace.FetchBlob('loss')
                avg_acc += accuracy
                print("[TRAIN] epoch:{}  iteration:{}/{}  accuracy:{:.3f}  loss:{:.3f}  time:{:.3f}s".format(
                    epoch, iters, iters_per_epoch, accuracy, loss, iter_end - iter_start)
                )
            epoch_end = time.time()
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("[TRAIN] epoch {} training over, epoch time: {:.3f}s, avg_acc: {:.3f}".format(
                epoch, epoch_end - epoch_start, avg_acc / iters_per_epoch))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            # model validation
            if epoch % self.validation_period == 0:
                epoch_start = time.time()
                iters_per_epoch = self.validation_num // self.batch_size
                avg_acc = 0
                for iters in range(iters_per_epoch):
                    iter_start = time.time()
                    workspace.RunNet(self.validation_model.net)
                    iter_end = time.time()
                    print("============================================================================")
                    accuracy = workspace.FetchBlob('accuracy')
                    avg_acc += accuracy
                    loss = workspace.FetchBlob('loss')
                    print("[VALIDATION] epoch:{}  iteration:{}/{}  accuracy:{:.3f}  time:{:.3f}s".format(
                        epoch, iters, iters_per_epoch, accuracy, iter_end - iter_start)
                    )
                epoch_end = time.time()
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("[VALIDATION] epoch {} validation over, epoch time: {:.3f}s, avg_acc: {:.3f}".format(
                    epoch, epoch_end - epoch_start, avg_acc / iters_per_epoch))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # model checkpoint
            if epoch % self.checkpoint_peroid == 0:
                save_checkpoint(
                    self.training_model.GetAllParams(),
                    self.validation_model.net,
                    workspace,
                    '{}_{}'.format(self.pretrained_model_name, self.dataset_name),
                    self.checkpoints_folder
                )

        training_end = time.time()
        print("Total finetuning time is {:.3f}s".format(training_end - training_start))


def parse_args():
    # load config file form .yaml
    config_parser = argparse.ArgumentParser(
        description='model-finetune config parser',
        add_help=False
    )
    config_parser.add_argument(
        '--config',
        type=str,
        required=True,
        default='vgg16_cars196_ft_config.yaml',
        help = 'config file'
    )
    args, _ = config_parser.parse_known_args()
    with open(args.config) as f:
        config = yaml.load(f)
        config_parser.set_defaults(**config)

    # parse rest training specific arguments
    args_parser = argparse.ArgumentParser(
        description='model-finetune cmdline parser',
        parents=[config_parser]
    )
    args_parser.add_argument("--batch_size", type=int, help="the mini-batch of data")
    args_parser.add_argument("--weight_decay", type=float, help="regularization of model")
    args_parser.add_argument("--base_lr", type=float, help="base learning rate of model")
    args_parser.add_argument("--momentum", type=float, help="momentum of the gradient")
    args_parser.add_argument("--gamma", type=float, help="learning rate decay rate")
    args_parser.add_argument("--policy", type=str, help="learning rate decay policy of model")
    args_parser.add_argument("--stepsize", type=int, help="iteration numbers that learning rate decays")
    args_parser.add_argument("--max_iter", type=int, help="max iterations of whole training")
    args_parser.add_argument("--gpu_id", type=str, help="which GPU to run the model")

    args = args_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ft_model = FinetuneModel(args)
    ft_model.finetune()



    # print scripts
    # print net proto
    '''
    print(self.training_model.net.Proto())
    sys.exit(0)

    for index, op in enumerate(self.training_model.param_init_net.Proto().op):
        print("index: {}, op_output: {}, op_name: {} device_option: {}".format(
            index+1, op.output, op.type, op.device_option))
    sys.exit(0)

    for index, op in enumerate(self.training_model.net.Proto().op):
        print("index: {}, op_output: {}, op_name: {} device_option: {}".format(
            index+1, op.output, op.type, op.device_option))
    sys.exit(0)
    '''

    # print blobs
    '''
    print("All Blobs:")
    for i, blob in enumerate(workspace.Blobs()):
        print("index: {}, blob name: {}".format(i + 1, blob))
    '''

    # write svg minimal
    '''
    train_init_graph = net_drawer.GetPydotGraphMinimal(self.training_model.param_init_net, rankdir='TB')
    train_init_graph.write_svg()
    train_predict_graph = net_drawer.GetPydotGraphMinimal(self.training_model.net)
    '''

    # print params
    '''
    print("All params of validation model")
    all_params = self.validation_model.GetAllParams()
    for index, name in enumerate(all_params):
        print("index: {}, name: {}".format(index + 1, name))
    computed_params = self.validation_model.GetComputedParams()
    print("computed_params length: {}".format(len(computed_params)))
    params = self.validation_model.GetParams()
    print("params length: {}".format(len(params)))
    sys.exit(0)
    '''

    '''
    print("data sanity check")
    i = 0
    data = workspace.FetchBlob('data')
    print("data shape: {}".format(data.shape))
    image = data[i].transpose(1, 2, 0)
    image = np.asarray(image, dtype=np.uint8)
    # image = image[:, :, (2, 1, 0)]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = workspace.FetchBlob('label')
    print("label: {}".format(label))
    plt.figure(i, figsize=(12, 8))
    plt.imshow(image)
    plt.draw()
    plt.savefig('{}.png'.format(label[i]))
    sys.exit(0)
    '''

    # print eval result of mean_per_channel
    '''
    # mean_per_channel=list([eval(item) for item in args.mean_per_channel.split(',')]),
    # std_per_channel=list([eval(item) for item in args.std_per_channel.split(',')]),
    mean_per_channel = args.mean_per_channel
    std_per_channel = args.std_per_channel
    print("mean_per_channel: {}".format(mean_per_channel))
    print("mean_per_channel type: {}".format(type(mean_per_channel)))
    print("std_per_channel: {}".format(std_per_channel))
    print("std_per_channel type: {}".format(type(std_per_channel)))
    '''
