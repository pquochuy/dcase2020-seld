import numpy as np
import random
import math
import time
import os

class DataGenerator_Individual:
    def __init__(self, label_dir, feat_dir, filename, Ncat=14, seq_len=300, seq_hop=300, iseval=False):
        self._feat2label_factor = 5

        self._X_mel = None
        self._y_sed = None
        self._y_doa = None

        self._label_dir = label_dir
        self._feat_dir = feat_dir
        self._filename = filename

        self._iseval = iseval
        self._Ncat = Ncat

        self._data_index = None
        self._data_size = 0
        self._seq_len = seq_len
        self._seq_hop = seq_hop
        self._ndim = 64

        self._feat_shape = None
        self._label_shape_sed = None
        self._label_shape_doa = None

        self._pointer = 0

    def load_data(self):
        print(os.path.join(self._feat_dir, self._filename))
        X = np.load(os.path.join(self._feat_dir, self._filename))
        self._X_mel = np.reshape(X, [X.shape[0], self._ndim, X.shape[-1]//self._ndim])
        if self._iseval is False:
            print(os.path.join(self._label_dir, self._filename))
            label = np.load(os.path.join(self._label_dir, self._filename))
        else:
            # dummy label
            label = np.zeros([self._X_mel.shape[0]//self._feat2label_factor, self._Ncat*4])
        self._y_sed = label[:, :self._Ncat]
        self._y_doa = label[:, self._Ncat:]

        self._data_index = np.arange(0, len(self._X_mel) - self._seq_len + 1, self._seq_hop)
        self._data_size = len(self._data_index)
        self._feat_shape = self.get_feat_shape()
        self._label_shape_sed = self.get_label_shape_sed()
        self._label_shape_doa = self.get_label_shape_doa()

    def get_feat_shape(self):
        return np.append([self._seq_len], self._X_mel.shape[1:])

    def get_label_shape_sed(self):
        return np.append([self._seq_len//self._feat2label_factor], self._y_sed.shape[1:])

    def get_label_shape_doa(self):
        return np.append([self._seq_len//self._feat2label_factor], self._y_doa.shape[1:])

    def get_random_sample(self,N=1):
        if (N > len(self._data_index)):
            N = len(self._data_index)
        x_mel = np.ndarray(np.append([N], self._feat_shape))
        y_sed = np.ndarray(np.append([N], self._label_shape_sed))
        y_doa = np.ndarray(np.append([N], self._label_shape_doa))

        data_index = np.random.choice(self._data_index, N, replace=False)
        for i in range(len(data_index)):
            x_mel[i] = self._X_mel[data_index[i]: data_index[i] + self._seq_len]
            y_sed[i] = self._y_sed[data_index[i]//self._feat2label_factor:
                                   data_index[i]//self._feat2label_factor + self._seq_len//self._feat2label_factor]
            y_doa[i] = self._y_doa[data_index[i]//self._feat2label_factor:
                                   data_index[i]//self._feat2label_factor + self._seq_len//self._feat2label_factor]

        # Get next batch of image (path) and labels
        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_doa.astype(np.float32)

        return x_mel, y_sed, y_doa

    def get_all_samples(self):
        N = len(self._data_index)
        x_mel = np.ndarray(np.append([N], self._feat_shape))
        y_sed = np.ndarray(np.append([N], self._label_shape_sed))
        y_doa = np.ndarray(np.append([N], self._label_shape_doa))

        data_index = self._data_index
        for i in range(len(data_index)):
            x_mel[i] = self._X_mel[data_index[i]: data_index[i] + self._seq_len]
            y_sed[i] = self._y_sed[data_index[i] // self._feat2label_factor:
                                   data_index[i] // self._feat2label_factor + self._seq_len // self._feat2label_factor]
            y_doa[i] = self._y_doa[data_index[i] // self._feat2label_factor:
                                   data_index[i] // self._feat2label_factor + self._seq_len // self._feat2label_factor]

        # Get next batch of image (path) and labels
        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_doa.astype(np.float32)

        return x_mel, y_sed, y_doa

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self._data_index[self._pointer:self._pointer + batch_size]

        # update pointer
        self._pointer += batch_size

        x_mel = np.ndarray(np.append([batch_size, self._seq_len], self._feat_shape[1:]))
        y_sed = np.ndarray([batch_size, self._seq_len//self._feat2label_factor, self._Ncat])
        y_sel = np.ndarray([batch_size, self._seq_len//self._feat2label_factor, self._Ncat*3])

        for i in range(len(data_index)):
            x_mel[i] = self._X_mel[data_index[i]: data_index[i] + self._seq_len]
            y_sed[i] = self._y_sed[data_index[i]//self._feat2label_factor:
                                   data_index[i]//self._feat2label_factor + self._seq_len//self._feat2label_factor]
            y_sel[i] = self._y_doa[data_index[i]//self._feat2label_factor:
                                   data_index[i]//self._feat2label_factor + self._seq_len//self._feat2label_factor]

        # Get next batch of image (path) and labels
        x_mel.astype(np.float32)
        y_sed.astype(np.float32)
        y_sel.astype(np.float32)

        return x_mel, y_sed, y_sel

    def rest_batch(self):

        data_index = self._data_index[self._pointer: len(self._data_index)]
        actual_len = len(self._data_index) - self._pointer

        # update pointer
        self._pointer = len(self._data_index)

        x_mel = np.ndarray(np.append([actual_len, self._seq_len], self._feat_shape[1:]))
        y_sed = np.ndarray([actual_len, self._seq_len // self._feat2label_factor, self._Ncat])
        y_sel = np.ndarray([actual_len, self._seq_len // self._feat2label_factor, self._Ncat * 3])

        for i in range(len(data_index)):
            x_mel[i] = self._X_mel[data_index[i]: data_index[i] + self._seq_len]
            y_sed[i] = self._y_sed[data_index[i] // self._feat2label_factor:
                                   data_index[i] // self._feat2label_factor + self._seq_len // self._feat2label_factor]
            y_sel[i] = self._y_doa[data_index[i] // self._feat2label_factor:
                                   data_index[i] // self._feat2label_factor + self._seq_len // self._feat2label_factor]

            # Get next batch of image (path) and labels
            x_mel.astype(np.float32)
            y_sed.astype(np.float32)
            y_sel.astype(np.float32)

            return actual_len, x_mel, y_sed, y_sel