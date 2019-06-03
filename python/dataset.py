from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2

import numpy as np
import os
import glob
import cv2
import csv
import lmdb
import time
import operator
import sys
import yaml
import argparse

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



class Dataset(object):
    def __init__(self, args):
        # data path
        self.root_data_dir = args.data['root_data_dir']
        self.root_sr_data_dir = args.data['root_sr_data_dir']
        self.root_cropped_data_dir = args.data['root_cropped_data_dir']
        self.root_cropped_sr_data_dir = args.data['root_cropped_sr_data_dir']
        self.root_test_imagery_dir = args.data['root_test_imagery_dir']
        self.root_cropped_test_imagery_dir = args.data[
            'root_cropped_test_imagery_dir']
        self.root_cropped_sr_test_imagery_dir = args.data[
            'root_cropped_sr_test_imagery_dir']

        self.root_data_csv_path = args.data['root_data_csv_path']
        self.root_test_csv_path = args.data['root_test_csv_path']
        self.data_num = args.data['data_num']
        self.divide_rate = args.divide_rate
        self.testing_image_num = args.data['testing_image_num']
        self.training_image_num = int(self.data_num * self.divide_rate)
        self.validation_image_num = self.data_num - self.training_image_num

        # lmdb path
        self.root_lmdb_dir = args.lmdb['root_lmdb_dir']
        self.division_lmdb_dir = os.path.join(
            self.root_lmdb_dir, str(self.divide_rate))
        self.training_lmdb_dir = os.path.join(
            self.division_lmdb_dir, args.lmdb['training_lmdb_dir'])
        self.validation_lmdb_dir = os.path.join(
            self.division_lmdb_dir, args.lmdb['validation_lmdb_dir'])
        self.testing_lmdb_dir = os.path.join(
            self.division_lmdb_dir, args.lmdb['testing_lmdb_dir'])
        self.encoded_training_lmdb_dir = os.path.join(
            self.division_lmdb_dir, args.lmdb['encoded_training_lmdb_dir'])
        self.encoded_validation_lmdb_dir = os.path.join(
            self.division_lmdb_dir, args.lmdb['encoded_validation_lmdb_dir'])
        self.encoded_testing_lmdb_dir = os.path.join(
            self.division_lmdb_dir, args.lmdb['encoded_testing_lmdb_dir'])

        # some mapping configs
        self.field_index = args.mapping['field_index']
        self.label_value = args.mapping['label_value']
        self.task_names = args.mapping['task_names']
        self.data_format_info = self._get_data_format_info()

        # some control handles
        self.encode = args.encode
        self.use_super_resolution = args.use_super_resolution
        self.super_resolution_scale = args.super_resolution_scale


    # Some private functions
    def _get_data_format_info(self):
        ''' get the extension info of the total training & testing imagery
        returns:
            dict(str, dict(str, set(str)))
        '''
        part_format_info = {'tiff' : set(), 'tif' : set()}
        format_info = {'train' : part_format_info, 'test' : part_format_info}

        # training image format
        if not os.path.exists(self.root_data_dir):
            raise ValueError('the {} does not exists'.format(
                self.root_data_dir))
        for name in os.listdir(self.root_data_dir):
            parts = name.split('.')
            if parts[-1] == 'tiff':
                format_info['train']['tiff'].add(parts[0])
            elif parts[-1] == 'tif':
                format_info['train']['tif'].add(parts[0])

        # testing image format
        if not os.path.exists(self.root_test_imagery_dir):
            raise ValueError('the {} does not exists'.format(
                self.root_test_imagery_dir))
        for name in os.listdir(self.root_test_imagery_dir):
            parts = name.split('.')
            if parts[-1] == 'tiff':
                format_info['test']['tiff'].add(parts[0])
            elif parts[-1] == 'tif':
                format_info['test']['tif'].add(parts[0])

        return format_info


    def _crop(self, image, coordinations):
        ''' crop the image with four bounding coordinations
        Args:
            image: numpy array
            image_id: image name string
            coordinations: list of the strings denote float numbers
        Returns:
            the cropped numpy array
        '''
        for i in range(len(coordinations)):
            value = coordinations[i]
            value = float(value)
            value = round(value)
            value = int(value)
            if value < 0:
                value = 0
            coordinations[i] = value
        min_x = min(coordinations[::2])
        max_x = max(coordinations[::2])
        min_y = min(coordinations[1::2])
        max_y = max(coordinations[1::2])
        return image[min_y:max_y, min_x:max_x]


    # Some core functions for preparing a dataset
    def initialize(self):
        '''
        do the sanity check for path
        '''
        print("[INFO] start initialize for making data lmdb")
        # check data dir
        if not os.path.exists(self.root_data_dir):
            raise ValueError("root training data direcroty does not exist")
        if not os.path.exists(self.root_test_imagery_dir):
            raise ValueError("root testing data direcroty does not exist")
        if not os.path.exists(self.root_cropped_data_dir):
            raise ValueError("root cropped training data direcroty does not exist")
        if not os.path.exists(self.root_cropped_sr_data_dir):
            raise ValueError("root cropped sr training data direcroty does not exist")
        if not os.path.exists(self.root_cropped_test_imagery_dir):
            raise ValueError("root cropped test data direcroty does not exist")
        if not os.path.exists(self.root_cropped_sr_test_imagery_dir):
            raise ValueError("root cropped sr testing data direcroty does not exist")
        if not os.path.exists(self.root_data_csv_path):
            raise ValueError("root training data csv file does not exist")
        if not os.path.exists(self.root_test_csv_path):
            raise ValueError("root testing data csv file does not exist")

        # check lmdb dir
        if not os.path.exists(self.root_lmdb_dir):
            os.makedirs(self.root_lmdb_dir)
        if not os.path.exists(self.division_lmdb_dir):
            os.makedirs(self.division_lmdb_dir)
        if not os.path.exists(self.training_lmdb_dir):
            os.makedirs(self.training_lmdb_dir)
        if not os.path.exists(self.validation_lmdb_dir):
            os.makedirs(self.validation_lmdb_dir)
        if not os.path.exists(self.testing_lmdb_dir):
            os.makedirs(self.testing_lmdb_dir)
        if not os.path.exists(self.encoded_training_lmdb_dir):
            os.makedirs(self.encoded_training_lmdb_dir)
        if not os.path.exists(self.encoded_validation_lmdb_dir):
            os.makedirs(self.encoded_validation_lmdb_dir)
        if not os.path.exists(self.encoded_testing_lmdb_dir):
            os.makedirs(self.encoded_testing_lmdb_dir)

        print("[INFO] initialize over...")


    def map_imagename_with_labelvalue(self, data_rows, dataset_type):
        ''' generate readable image name and label value mapping
        Args:
            data_rows: rows readed from csv file. list of strings
            dataset_type: string type names training or validation set
        '''
        if len(data_rows) == 0:
            return

        for task in self.task_names:
            print("[INFO] start mapping {}-{}'s image name and label value".format(
                task, dataset_type))
            # define file_path
            if dataset_type == 'train':
                if self.encode:
                    file_path = os.path.join(
                        self.encoded_training_lmdb_dir,
                        '{}_{}.txt'.format(task, dataset_type)
                    )
                else:
                    file_path = os.path.join(
                        self.training_lmdb_dir,
                        '{}_{}.txt'.format(task, dataset_type)
                    )
            elif dataset_type == 'val':
                if self.encode:
                    file_path = os.path.join(
                        self.encoded_validation_lmdb_dir,
                        '{}_{}.txt'.format(task, dataset_type)
                    )
                else:
                    file_path = os.path.join(
                        self.validation_lmdb_dir,
                        '{}_{}.txt'.format(task, dataset_type)
                    )

            # write imagename - lablevalue mapping
            with open(file_path, 'w') as f:
                if task == 'feature':
                    for row in data_rows:
                        label_val = []
                        for feat in self.label_value[task].keys():
                            if feat == 'empty_feature':
                                continue
                            if (row[self.field_index[feat]] == '1'):
                                label_val.append(self.label_value[task][feat])

                        f.write(row[0]) # tag_id
                        if len(label_val) > 0:
                            f.write(' ')
                            for v in label_val[:-1]:
                                f.write('{},'.format(v))
                            f.write('{}\n'.format(label_val[-1]))
                        else:
                            f.write('\n')
                else:
                    for row in data_rows:
                        f.write(row[0])  # tag id
                        f.write(' ')
                        key = row[self.field_index[task]]
                        f.write('{}\n'.format(self.label_value[task][key]))
            print("[INFO] map {}-{}'s image name and label value over...".format(
                task, dataset_type))


    def map_labelvalue_with_labelname(self):
        '''
        generate readable label value and name mapping
        '''
        print("[INFO] start mapping label value with label name")
        for task in self.task_names:
            file_dir = os.path.join('.', 'training_data_class_words')
            file_path = os.path.join(file_dir,'{}_words.txt'.format(task))
            with open(file_path, 'w') as f:
                mapping = self.label_value[task]
                sorted_mapping = sorted(mapping.items(), key=lambda kv : kv[1])
                for key, value in sorted_mapping:
                    f.write("{} {}\n".format(value, key))
        print("[INFO] map label value with label name over...")


    def write_single_label_lmdb(self, lmdb_path, data_rows, task):
        ''' write lmdb for single label (also named multi-class) image
        classification
        Args:
            lmdb_path: string locate the lmdb path
            data_rows: rows readed from csv file, list of strings
            task: string type names a specific multi-class task e.g 'sub_class'
        Notice:
            Basically, this method is for 'training' utility, including
            training_lmdb and validation lmdb, and we should pay attention
            to the 'image_dir' variable below
            For 'testing', we have 'write_empty_label_lmdb'
            Besides, better use 'ImageInputOp' to read lmdb
        '''
        assert(lmdb_path != "")
        assert(task in self.task_names)
        if len(data_rows) == 0:
            return

        print("[INFO] start writing lmdb for {}".format(task))
        start_time = time.time()
        LMDB_MAP_SIZE = 1 << 40
        env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

        if self.use_super_resolution:
            image_dir = self.root_cropped_sr_data_dir
        else:
            image_dir = self.root_cropped_data_dir

        with env.begin(write=True) as txn:
            count = 0
            for row in data_rows:
                # create TensorProtos as lmdb record
                record = caffe2_pb2.TensorProtos()

                # image tensor
                img_tensor = record.protos.add()
                image_id = row[0] + '.png'  # tag id denotes image id
                if self.encode:
                    # write image as string
                    fstream = open(os.path.join(image_dir, image_id), 'rb')
                    img_tensor.dims.append(1)
                    img_tensor.data_type = 4 # string_data
                    img_tensor.string_data.append(fstream.read())
                    fstream.close()
                else:
                    # write image as byte data
                    img_data = cv2.imread(os.path.join(image_dir, image_id))  # BGR,HWC
                    img_tensor.dims.extend(img_data.shape)
                    img_tensor.data_type = 3  # byte_data
                    flatten_img = img_data.reshape(np.prod(img_data.shape))
                    img_tensor.byte_data = flatten_img.tobytes()

                # label tensor
                lbl_tensor = record.protos.add()
                img_tensor.dims.append(1)
                lbl_tensor.data_type = 2  # int32
                img_label_key = row[self.field_index[task]]
                img_label = self.label_value[task][img_label_key]
                lbl_tensor.int32_data.append(img_label)

                # write db record
                txn.put(
                    '{}'.format(count).encode('ascii'),
                    record.SerializeToString()
                )
                if ((count % 1000 == 0)):
                    print("[INFO]     Insert {} records".format(count))
                count += 1
        end_time = time.time()
        print("[INFO] write {} lmdb over, total time: {:.3f}s".format(
            task, end_time - start_time))


    def write_multi_label_lmdb(self, lmdb_path, data_rows, task):
        ''' write lmdb for multi label image classification
        Args:
            lmdb_path: string locate the lmdb path
            data_rows: rows readed from csv file, list of strings
            task: string type names a specific multi-label task e.g 'feature'
        Notice:
            Temporarily, we only consider 'feature' type multi-label image
            classification
            Basically, this method is for 'training' utility
            Besides, better use 'ImageInputOp' to read lmdb
        '''
        assert(lmdb_path != "")
        assert(task in self.task_names)
        if len(data_rows) == 0:
            return

        print("[INFO] start writing lmdb for {}".format(task))
        start_time = time.time()
        LMDB_MAP_SIZE = 1 << 40
        env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

        if self.use_super_resolution:
            image_dir = self.root_cropped_sr_data_dir
        else:
            image_dir = self.root_cropped_data_dir

        with env.begin(write=True) as txn:
            count = 0
            for row in data_rows:
                # create TensorProtos as lmdb record
                record = caffe2_pb2.TensorProtos()

                # image tensor
                img_tensor = record.protos.add()
                image_id = row[0] + '.png'  # tag id denotes image id
                if self.encode:
                    # write image as string
                    fstream = open(os.path.join(image_dir, image_id), 'rb')
                    img_tensor.dims.append(1)
                    img_tensor.data_type = 4 # string_data
                    img_tensor.string_data.append(fstream.read())
                    fstream.close()
                else:
                    # write image as byte data
                    img_data = cv2.imread(os.path.join(image_dir, image_id))  # BGR,HWC
                    img_tensor.dims.extend(img_data.shape)
                    img_tensor.data_type = 3  # byte_data
                    flatten_img = img_data.reshape(np.prod(img_data.shape))
                    img_tensor.byte_data = flatten_img.tobytes()

                # label tensor
                lbl_tensor = record.protos.add()
                # lbl_tensor.data_type = 2  # int32
                lbl_tensor.data_type = 1  # float
                img_label = []
                for feat in self.label_value[task].keys():
                    if feat == 'empty_feature':
                        continue
                    feat_val = int(row[self.field_index[feat]])
                    if feat_val == 1:
                        img_label.append(float(self.label_value[task][feat]))
                img_tensor.dims.append(1)

                if len(img_label) == 0:
                    img_label.append(float(len(self.label_value[task]) - 1))
                lbl_tensor.float_data.extend(img_label)

                # write db record
                txn.put(
                    '{}'.format(count).encode('ascii'),
                    record.SerializeToString()
                )
                if ((count % 1000 == 0)):
                    print("[INFO]     Insert {} records".format(count))
                count += 1
        end_time = time.time()
        print("[INFO] write {} lmdb over, total time: {:.3f}s".format(
            task, end_time - start_time))


    def write_combined_multi_label_lmdb(self, lmdb_path, data_rows, tasks):
        ''' write lmdb for multi label image classification, which combines
        'general_calss' AND 'sub_class'
        Args:
            lmdb_path: string locate the lmdb path
            data_rows: rows readed from csv file, list of strings
            tasks: list of str, names different kind of task
        Notice:
            Basically, this method is for 'training' utility
            Besides, better use 'ImageInputOp' to read lmdb
        '''
        assert(lmdb_path != "")
        assert(len(tasks) > 0)
        if len(data_rows) == 0:
            return

        print("[INFO] start writing combined lmdb for {}".format(tasks))
        start_time = time.time()
        LMDB_MAP_SIZE = 1 << 40
        env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

        if self.use_super_resolution:
            image_dir = self.root_cropped_sr_data_dir
        else:
            image_dir = self.root_cropped_data_dir

        with env.begin(write=True) as txn:
            count = 0
            for row in data_rows:
                # create TensorProtos as lmdb record
                record = caffe2_pb2.TensorProtos()

                # image tensor
                img_tensor = record.protos.add()
                image_id = row[0] + '.png'  # tag id denotes image id
                if self.encode:
                    # write image as string
                    fstream = open(os.path.join(image_dir, image_id), 'rb')
                    img_tensor.dims.append(1)
                    img_tensor.data_type = 4 # string_data
                    img_tensor.string_data.append(fstream.read())
                    fstream.close()
                else:
                    # write image as byte data
                    img_data = cv2.imread(os.path.join(image_dir, image_id))  # BGR,HWC
                    img_tensor.dims.extend(img_data.shape)
                    img_tensor.data_type = 3  # byte_data
                    flatten_img = img_data.reshape(np.prod(img_data.shape))
                    img_tensor.byte_data = flatten_img.tobytes()

                # label tensor
                lbl_tensor = record.protos.add()
                # lbl_tensor.data_type = 2  # int32
                lbl_tensor.data_type = 1  # float
                img_label = []
                for task in tasks:
                    label_key = row[self.field_index[task]]
                    label_val = self.label_value[task][label_key]
                    if task == 'sub_class':
                        label_val += 2
                    img_label.append(float(label_val))
                img_tensor.dims.append(1)
                lbl_tensor.float_data.extend(img_label)

                # write db record
                txn.put(
                    '{}'.format(count).encode('ascii'),
                    record.SerializeToString()
                )
                if ((count % 1000 == 0)):
                    print("[INFO]     Insert {} records".format(count))
                count += 1
        end_time = time.time()
        print("[INFO] write combined {} lmdb over, total time: {:.3f}s".format(
            tasks, end_time - start_time))


    def write_empty_label_lmdb(self, lmdb_path, data_rows):
        ''' write lmdb only for images without labels
        Args:
            lmdb_path: string locate the lmdb path
            data_rows: rows readed from csv file, list of strings
        Notice:
            testing lmdb record only has image tensor, no label tensor
        '''
        assert(lmdb_path != "")
        assert(len(data_rows) > 0)

        print("[INFO] start writing testing lmdb")
        start_time = time.time()
        LMDB_MAP_SIZE = 1 << 40
        env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

        if self.use_super_resolution:
            image_dir = self.root_cropped_sr_test_imagery_dir
        else:
            image_dir = self.root_cropped_test_imagery_dir

        with env.begin(write=True) as txn:
            count = 0
            for row in data_rows:
                # create TensorProtos as lmdb record
                record = caffe2_pb2.TensorProtos()

                # image tensor
                img_tensor = record.protos.add()
                image_id = row[0] + '.png'  # tag id denotes image id
                if self.encode:
                    # write image as string
                    fstream = open(os.path.join(image_dir, image_id), 'rb')
                    img_tensor.dims.append(1)
                    img_tensor.data_type = 4 # string_data
                    img_tensor.string_data.append(fstream.read())
                    fstream.close()
                else:
                    # write image as byte data
                    img_data = cv2.imread(os.path.join(image_dir, image_id))  # BGR,HWC
                    img_tensor.dims.extend(img_data.shape)
                    img_tensor.data_type = 3  # byte_data
                    flatten_img = img_data.reshape(np.prod(img_data.shape))
                    img_tensor.byte_data = flatten_img.tobytes()

                # id tensor
                id_tensor = record.protos.add()
                img_tensor.dims.append(1)
                id_tensor.data_type = 2  # int32
                id_tensor.int32_data.append(int(row[0]))

                # write db record
                txn.put(
                    '{}'.format(count).encode('ascii'),
                    record.SerializeToString()
                )
                if ((count % 1000 == 0)):
                    print("[INFO]     Insert {} records".format(count))
                count += 1
        end_time = time.time()
        print("[INFO] write testing lmdb over, total time: {:.3f}s".format(
            end_time - start_time))


    def write_training_lmdb(self, training_rows):
        '''
        write training lmdb
        '''
        for task in self.task_names:
            if task == 'feature':
                write_lmdb_function = self.write_multi_label_lmdb
            else:
                write_lmdb_function = self.write_single_label_lmdb
            write_lmdb_function(
                os.path.join(
                    self.encoded_training_lmdb_dir if self.encode else \
                    self.training_lmdb_dir,
                    '{}_training_lmdb'.format(task)
                ),
                training_rows,
                task,
            )


    def write_validation_lmdb(self, validation_rows):
        '''
        write validation lmdb
        '''
        for task in self.task_names:
            if task == 'feature':
                write_lmdb_function = self.write_multi_label_lmdb
            else:
                write_lmdb_function = self.write_single_label_lmdb
            write_lmdb_function(
                os.path.join(
                    self.encoded_validation_lmdb_dir if self.encode else \
                    self.validation_lmdb_dir,
                    '{}_validation_lmdb'.format(task)
                ),
                validation_rows,
                task,
            )


    def write_testing_lmdb(self):
        '''
        write testing lmdb
        '''
        # load test csv rows
        reader = csv.reader(open(self.root_test_csv_path, 'rU'), dialect='excel')
        testing_rows = np.array([row for i, row in enumerate(reader) if i != 0])

        #
        self.write_empty_label_lmdb(
            self.encoded_testing_lmdb_dir if self.encode else \
            self.testing_lmdb_dir,
            testing_rows,
        )


    def write_single_mafat_lmdb(self, training_rows=[], validation_rows=[]):
        '''
        write single task including combined tasks lmdb file for training
        Temporarily, only for 'general_class' + 'sub_class' combined tasks
        '''
        # load data csv rows
        if len(training_rows) == 0:
            print("[INFO]     Reshuffling the data rows")
            reader = csv.reader(open(self.root_data_csv_path, 'rU'),
                                dialect='excel')
            total_rows = np.array(
                [row for i, row in enumerate(reader) if i != 0])
            index_shuffle = np.random.permutation(self.data_num)
            total_rows = total_rows[index_shuffle]  # shuffle rows
            training_rows = total_rows[:self.training_image_num]
            validation_rows = total_rows[self.training_image_num:]

        # write lmdb
        # ugly code for temporary using
        tasks = ['general_class', 'sub_class']
        self.write_combined_multi_label_lmdb(
                os.path.join(
                    self.encoded_training_lmdb_dir if self.encode else \
                    self.training_lmdb_dir,
                    '{0[0]}_{0[1]}_training_lmdb'.format(tasks)
                ),
                training_rows,
                tasks,
            )
        self.write_combined_multi_label_lmdb(
                os.path.join(
                    self.encoded_validation_lmdb_dir if self.encode else \
                    self.validation_lmdb_dir,
                    '{0[0]}_{0[1]}_validation_lmdb'.format(tasks)
                ),
                validation_rows,
                tasks,
            )


    def write_mafat_lmdb(self):
        '''
        write all lmdb file, including all the 4 tasks:
            ['general_class','sub_class', 'feature', 'color']
        '''
        # initialize
        self.initialize()

        # load data csv rows
        reader = csv.reader(open(self.root_data_csv_path, 'rU'), dialect='excel')
        total_rows = np.array([row for i, row in enumerate(reader) if i != 0])
        index_shuffle = np.random.permutation(self.data_num)
        total_rows = total_rows[index_shuffle]
        training_rows = total_rows[:self.training_image_num]
        validation_rows = total_rows[self.training_image_num:]

        # generate readable label info of dataset
        self.map_imagename_with_labelvalue(training_rows, 'train')
        self.map_imagename_with_labelvalue(validation_rows, 'val')
        self.map_labelvalue_with_labelname()

        # write lmdb
        self.write_training_lmdb(training_rows)
        self.write_validation_lmdb(validation_rows)
        self.write_testing_lmdb()  # testing lmdb has no label

        self.write_single_mafat_lmdb(training_rows, validation_rows)


    def write_general_mafat_lmdb(self):
        '''
        for general_class sub task, we make lmdb separately for
        'large vehicle' and 'small vehicle'
        '''
        task = 'general_class'

        # load data csv rows
        reader = csv.reader(open(self.root_data_csv_path, 'rU'), dialect='excel')
        total_rows = np.array([row for i, row in enumerate(reader) if i != 0])

        # rows label shuffling
        large_vehicle_rows = [row for row in total_rows if row[self.field_index[task]] == 'large vehicle']
        small_vehicle_rows = [row for row in total_rows if row[self.field_index[task]] == 'small vehicle']
        assert(len(large_vehicle_rows) + len(small_vehicle_rows) == len(total_rows))

        # write lmdb
        self.write_single_label_lmdb(
            os.path.join(
                self.encoded_training_lmdb_dir if self.encode else \
                self.training_lmdb_dir,
                '{}_large_vehicle_training_lmdb'.format(task)
            ),
            large_vehicle_rows,
            task,
        )
        self.write_single_label_lmdb(
            os.path.join(
                self.encoded_training_lmdb_dir if self.encode else \
                self.training_lmdb_dir,
                '{}_small_vehicle_training_lmdb'.format(task)
            ),
            small_vehicle_rows,
            task,
        )


    # Some plotting functions for data statistic info
    def plot_training_imagery_statistics(self):
        '''
        plot the training image number statistics of each task
        '''
        general_class_num = {'large vehicle' : 0, 'small vehicle' : 0}
        sub_class_num = {'sedan':0, 'hatchback':0, 'minivan':0, 'van':0,
            'pickup':0, 'jeep':0, 'truck':0, 'light truck':0, 'crane truck':0,
            'bus':0, 'prime mover':0, 'dedicated agricultural vehicle':0,
            'cement mixer':0, 'tanker':0, 'minibus':0}
        feature_num = {'sunroof':0, 'luggage_carrier':0, 'open_cargo_area':0,
            'enclosed_cab':0, 'spare_wheel':0, 'wrecked':0, 'flatbed':0,
            'ladder':0, 'enclosed_box':0, 'soft_shell_box':0,
            'harnessed_to_a_cart':0, 'ac_vents':0}
        color_num = {'yellow':0, 'red':0, 'blue':0, 'black':0,
            'silver/grey':0, 'white':0, 'green':0, 'other':0}

        # count
        start_time = time.time()
        reader = csv.reader(open(self.root_data_csv_path, 'rU'),
                            dialect='excel')
        for i, row in enumerate(reader):
            if i == 0:
                continue

            general_key = row[self.field_index['general_class']]
            general_class_num[general_key] += 1

            sub_key = row[self.field_index['sub_class']]
            sub_class_num[sub_key] += 1

            color_key = row[self.field_index['color']]
            color_num[color_key] += 1

            for feat_key in feature_num.keys():
                if feat_key == 'empty_feature':
                    continue
                if row[self.field_index[feat_key]] == '-1':
                    feature_num[feat_key] += 1

        # plot
        statistics = [general_class_num, sub_class_num, feature_num, color_num]
        stat_name = {0:'general_num', 1:'sub_num',
                     2:'feature_num', 3:'color_num'}
        for i in range(len(statistics)):
            s_dict = statistics[i]
            plt.figure(i, figsize=(12, 8))
            plt.title(stat_name[i])
            plt.barh(range(len(s_dict)), s_dict.values(),
                    tick_label=s_dict.keys())
            plt.draw()
            plt.savefig('./training_data_statistics/{}.png'.format(stat_name[i]))
        end_time = time.time()
        print("plot_statistics over, total time is {:.3f}s".format(
            end_time - start_time))


    def plot_cropped_training_imagery_statistics(self):
        '''
        plot the width & height distribution of training imagery
        '''
        reader = csv.reader(open(self.root_data_csv_path, 'rU'), dialect='excel')
        widths = []
        heights = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            coordinations[i] = value
            coordinations = [row[self.field_index['p1_x']],
                             row[self.field_index['p1_y']],
                             row[self.field_index['p2_x']],
                             row[self.field_index['p2_y']],
                             row[self.field_index['p3_x']],
                             row[self.field_index['p3_y']],
                             row[self.field_index['p4_x']],
                             row[self.field_index['p4_y']]]
            coordinations = [float(value) for value in coordinations]
            coordinations = [int(round(value)) for value in coordinations]
            coordinations = [0 if value < 0 else v for value in coordinations]
            min_x = min(coordinations[::2])
            max_x = max(coordinations[::2])
            min_y = min(coordinations[1::2])
            max_y = max(coordinations[1::2])
            widths.append(max_x - min_x)
            heights.append(max_y - min_y)

        # get number
        width_number_dict = {}
        for w in widths:
            if w in width_number_dict:
                width_number_dict[w] += 1
            else:
                width_number_dict[w] = 1
        sorted_width_keys = width_number_dict.keys()
        sorted_width_keys.sort()
        sorted_width_values = [width_number_dict[k] for k in sorted_width_keys]

        height_number_dict = {}
        for h in heights:
            if h in height_number_dict:
                height_number_dict[h] += 1
            else:
                height_number_dict[h] = 1
        sorted_height_keys = height_number_dict.keys()
        sorted_height_keys.sort()
        sorted_height_values = [height_number_dict[k] for k in sorted_height_keys]

        # plot
        plt.figure(0, figsize=(12, 8))
        plt.title('croped width')
        plt.plot(sorted_width_keys, sorted_width_values)
        plt.draw()
        plt.savefig('./training_data_statistics/cropped_width.png')

        plt.figure(1, figsize=(12, 8))
        plt.title('croped height')
        plt.plot(sorted_height_keys, sorted_height_values)
        plt.draw()
        plt.savefig('./training_data_statistics/cropped_height.png')
        print("[INFO] plot cropped image's width & height info over...")


    # Some utility functions for pre or post processing
    def make_cropped_training_imagery(self):
        '''
        crop out the training image named with 'tag_id.png'
        '''
        print("[INFO] start crop training imagery...")
        start_time = time.time()
        reader = csv.reader(open(self.root_data_csv_path, 'rU'), dialect='excel')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if i % 100 == 0:
                print("[INFO]     {} images have cropped".format(i))
            tag_id = row[0]
            image_id = row[1]

            # read
            if image_id in self.data_format_info['train']['tif']:
                image_id += '.tif'
            elif image_id in self.data_format_info['train']['tiff']:
                image_id += '.tiff'
            else:
                image_id += '.jpg'
            image = cv2.imread(os.path.join(self.root_data_dir, image_id))
            cropped_image = self._crop(
                image,
                [row[self.field_index['p1_x']],
                 row[self.field_index['p1_y']],
                 row[self.field_index['p2_x']],
                 row[self.field_index['p2_y']],
                 row[self.field_index['p3_x']],
                 row[self.field_index['p3_y']],
                 row[self.field_index['p4_x']],
                 row[self.field_index['p4_y']],
                 ]
            )
            # write
            # cropped_image_id = tag_id + '.jpg'
            cropped_image_id = tag_id + '.png'
            if not os.path.exists(self.root_cropped_data_dir):
                raise ValueError('root_cropped_data_dir not exists')
            cv2.imwrite(
                os.path.join(self.root_cropped_data_dir, cropped_image_id),
                cropped_image
            )
        end_time = time.time()
        print("[INFO] Crop training imagery over. Total time is {:.3f}s".format(
            end_time - start_time))


    def make_cropped_testing_imagery(self):
        '''
        crop out the testing imagery named with 'tag_id.png'
        '''
        print("[INFO] start crop testing image...")
        start_time = time.time()
        reader = csv.reader(open(self.root_test_csv_path, 'rU'), dialect='excel')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if i % 100 == 0:
                print("[INFO]     {} images have cropped".format(i))
            tag_id = row[0]
            image_id = row[1]

            # read
            if image_id in self.data_format_info['test']['tif']:
                image_id += '.tif'
            elif image_id in self.data_format_info['test']['tiff']:
                image_id += '.tiff'
            else:
                image_id += '.jpg'
            image = cv2.imread(os.path.join(self.root_test_imagery_dir, image_id))
            cropped_image = self._crop(
                image,
                [row[self.field_index['p1_x']],
                 row[self.field_index['p1_y']],
                 row[self.field_index['p2_x']],
                 row[self.field_index['p2_y']],
                 row[self.field_index['p3_x']],
                 row[self.field_index['p3_y']],
                 row[self.field_index['p4_x']],
                 row[self.field_index['p4_y']],
                 ]
            )
            # write
            # cropped_image_id = tag_id + '.jpg'
            cropped_image_id = tag_id + '.png'
            if not os.path.exists(self.root_cropped_test_imagery_dir):
                raise ValueError('root_cropped_test_imagery_dir not exists')
            cv2.imwrite(
                os.path.join(self.root_cropped_test_imagery_dir, cropped_image_id),
                cropped_image
            )
        end_time = time.time()
        print("[INFO] Crop testing imagery over. Total time is {:.3f}s".format(
            end_time - start_time))


def parse_args():
    # load config file form .yaml
    config_parser = argparse.ArgumentParser(
        description='config parser for preparing dataset',
        add_help=False
    )
    config_parser.add_argument(
        '--config',
        type=str,
        # required=True,
        default='mafat_dataset.yaml',
        help = 'config file'
    )
    args, _ = config_parser.parse_known_args()
    with open(args.config) as f:
        config = yaml.load(f)
        config_parser.set_defaults(**config)

    # parse rest arguments
    args_parser = argparse.ArgumentParser(
        description='cmdline parser for preparing dataset',
        parents=[config_parser]
    )
    args_parser.add_argument("--divide_rate", type=float,
                             help="training image proportion of the total image")
    args_parser.add_argument("--super_resolution_scale", type=int, default=4,
                             help="how numch to enhance the training images" \
                             "resolution")
    args_parser.add_argument("--use_super_resolution",
                             dest="use_super_resolution", action="store_true",
                             help="enhance the resolution of the training image")
    args_parser.add_argument("--encode", dest="encode", action="store_true",
                             help="write image file stream as lmdb record")

    args = args_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test = Dataset(args)
    # test.make_cropped_training_imagery()
    # test.make_cropped_testing_imagery()
    test.write_mafat_lmdb()
    # test.write_single_mafat_lmdb()
    # test.write_general_mafat_lmdb()









