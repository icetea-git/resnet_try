from os.path import join

import nrrd
import numpy as np

from rand_functions.helper_functions import gen_file_list, random_shuffle


class BatchDataGen:
    def __init__(self, file_path, mask_path, batch_size):
        self.file_path = file_path
        self.mask_path = mask_path
        self.batch_size = batch_size

    def gen_batch(self):
        file_list = gen_file_list(self.file_path)
        mask_list = gen_file_list(self.mask_path)

        while True:
            file_list, mask_list = random_shuffle(file_list, mask_list)
            for (file, mask) in zip(file_list, mask_list):

                X = load_data(file, self.file_path)
                Y = load_data(mask, self.mask_path)
                X, Y = random_shuffle(X, Y)
                X = np.array(X)
                Y = np.array(Y)
                for i in range(X.shape[0] // self.batch_size):
                    x_batch = X[i * self.batch_size:(i + 1) * self.batch_size, ...]
                    y_batch = Y[i * self.batch_size:(i + 1) * self.batch_size, ...]

                    yield x_batch, y_batch



def load_data(data, data_path):

    data, header = nrrd.read(join(data_path, data), index_order='C')
    data = (data - np.min(data)) / np.ptp(data)
    data = data[:, :, :, np.newaxis]

    return data
