import numpy as np
import random
import math
import time
import os
from datagenerator_individual import DataGenerator_Individual

class DataRegulator:
    # seq_hop and seq_hope must be a factor of 5
    def __init__(self, split, label_dir, feat_dir, Ncat=14, seq_len=300, seq_hop=300, shuffle=False, iseval=False):

        self._shuffle = shuffle
        self._pointer = 0

        self._iseval = iseval
        self._Ncat = Ncat

        self._label_dir = label_dir
        self._feat_dir = feat_dir
        self._splits = np.array(split)
        self._filename_list = list()
        self.get_file_list()

        self._gen_list = list()
        for i in np.arange(len(self._filename_list)):
            gen = DataGenerator_Individual(self._label_dir, self._feat_dir, self._filename_list[i],
                                           self._Ncat, seq_len, seq_hop, iseval)
            self._gen_list.append(gen)

        self._data_size = len(self._filename_list)
        self._data_index = np.arange(len(self._filename_list))
        self._all_label_sed_2d = None
        self._all_label_doa_2d = None

    def get_file_list(self):
        for filename in os.listdir(self._feat_dir):
            if self._iseval is False:
                if int(filename[4]) in self._splits:  # check which split the file belongs to
                    self._filename_list.append(filename)
            else:
                self._filename_list.append(filename)

        # test small here

    def load_data(self):
        for gen in self._gen_list:
            gen.load_data()
        self.prepare_all_label_2d()

    def prepare_all_label_2d(self):
        N = len(self._gen_list[0]._y_sed)
        self._all_label_sed_2d = np.ndarray([self._data_size * N, self._Ncat]).astype(np.float32)
        self._all_label_doa_2d = np.ndarray([self._data_size * N, self._Ncat*3]).astype(np.float32)
        for i in range(self._data_size):
            self._all_label_sed_2d[i * N: (i + 1) * N] = self._gen_list[i]._y_sed
            self._all_label_doa_2d[i * N: (i + 1) * N] = self._gen_list[i]._y_doa

    def get_in_shape(self):
        return self._gen_list[0].get_feat_shape()

    def get_out_shape_sed(self):
        return self._gen_list[0].get_label_shape_sed()

    def get_out_shape_doa(self):
        return self._gen_list[0].get_label_shape_doa()

    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(self._data_index))
        self._data_index = self._data_index[idx]

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self._pointer = 0
        if self._shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self._data_index[self._pointer: self._pointer + batch_size]
        # update pointer
        self._pointer += batch_size

        x_mel = np.ndarray(np.append([batch_size], self._gen_list[0].get_feat_shape()))
        y_sed = np.ndarray(np.append([batch_size], self._gen_list[0].get_label_shape_sed()))
        y_doa = np.ndarray(np.append([batch_size], self._gen_list[0].get_label_shape_doa()))

        for i in range(len(data_index)):
            x_mel[i],  y_sed[i], y_doa[i] = self._gen_list[data_index[i]].get_random_sample()

        # Get next batch of image (path) and labels
        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_doa.astype(np.float32)

        return x_mel, y_sed, y_doa

    def get_rest_batch(self):
        data_index = self._data_index[self._pointer: len(self._data_index)]
        actual_len = len(self._data_index) - self._pointer
        # update pointer
        self._pointer = len(self._data_index)

        x_mel = np.ndarray(np.append([actual_len], self._gen_list[0].get_feat_shape()))
        y_sed = np.ndarray(np.append([actual_len], self._gen_list[0].get_label_shape_sed()))
        y_doa = np.ndarray(np.append([actual_len], self._gen_list[0].get_label_shape_doa()))

        for i in range(len(data_index)):
            x_mel[i], y_sed[i], y_doa[i] = self._gen_list[data_index[i]].get_random_sample()

        # Get next batch of image (path) and labels
        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_doa.astype(np.float32)

        return actual_len, x_mel, y_sed, y_doa

    def next_batch_whole(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self._data_index[self._pointer: self._pointer + batch_size]

        # update pointer
        self._pointer += batch_size
        N = self._gen_list[0]._data_size
        x_mel = np.ndarray(np.append([batch_size*N], self._gen_list[0].get_feat_shape()))
        y_sed = np.ndarray(np.append([batch_size*N], self._gen_list[0].get_label_shape_sed()))
        y_doa = np.ndarray(np.append([batch_size*N], self._gen_list[0].get_label_shape_doa()))

        for i in range(len(data_index)):
            x_mel[i*N : (i+1)*N], y_sed[i*N : (i+1)*N], y_doa[i*N : (i+1)*N] = \
                self._gen_list[data_index[i]].get_all_samples()

        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_doa.astype(np.float32)

        return x_mel, y_sed, y_doa

    def rest_batch_whole(self):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self._data_index[self._pointer: len(self._data_index)]
        actual_len = len(self._data_index) - self._pointer

        # update pointer
        self._pointer = len(self._data_index)

        N = self._gen_list[0]._data_size
        x_mel = np.ndarray(np.append([actual_len*N], self._gen_list[0].get_feat_shape()))
        y_sed = np.ndarray(np.append([actual_len*N], self._gen_list[0].get_label_shape_sed()))
        y_doa = np.ndarray(np.append([actual_len*N], self._gen_list[0].get_label_shape_doa()))

        for i in range(len(data_index)):
            x_mel[i*N : (i+1)*N], y_sed[i*N : (i+1)*N], y_doa[i*N : (i+1)*N] = \
                self._gen_list[data_index[i]].get_all_samples()

        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_doa.astype(np.float32)

        return actual_len*N, x_mel, y_sed, y_doa

    def all_label_sed_2d(self):
        return self._all_label_sed_2d

    def all_label_doa_2d(self):
        return self._all_label_doa_2d
