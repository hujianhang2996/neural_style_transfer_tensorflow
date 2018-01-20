import pickle
import numpy as np


def read(file, TRAIN=True):
    if TRAIN:
        data = []
        label = []
        for i in range(5):
            with open(file + '/data_batch_' + str(i + 1), 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
            data.append(np.reshape(batch[b'data'], [10000, 3, 32, 32]))
            label.append(np.array(batch[b'labels']))

        batch_data = np.concatenate(
            (data[0], data[1], data[2], data[3], data[4]), axis=0)
        batch_data = np.transpose(batch_data, (0, 2, 3, 1))
        batch_label = np.concatenate(
            (label[0], label[1], label[2], label[3], label[4]), axis=0)

    else:
        with open(file + '/test_batch', 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        batch_data = np.reshape(batch[b'data'], [10000, 3, 32, 32])
        batch_data = np.transpose(batch_data, (0, 2, 3, 1))
        batch_label = np.array(batch[b'labels'])

    return batch_data, batch_label
