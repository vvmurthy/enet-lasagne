import numpy as np
import os
import scipy.misc as misc


class CamVid:
    def __init__(self, dir=os.getcwd() + '/CamVid/', **kwargs):
        self.dir = dir

        # sets unchangeable params
        self.nc = 3
        self.lab_ln = 12
        self.h = 360
        self.w = 480

        self.images_in_mem = kwargs.get('images_in_mem', 30)

        # Initialize images in dataset
        self.X_files_train, \
        self.y_files_train, \
        self.X_files_val, \
        self.y_files_val, \
        self.X_files_test, \
        self.y_files_test = self.init_dataset()

    def preprocess_im(self, filename):
        im = misc.imread(filename)
        im.astype(np.float32)

        # Swap to n_channels x image_size x image_size
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        return im

    def preprocess_annot(self, filename, h, w):

        # Load annotation
        annot = misc.imread(filename)

        if h != 360 and w != 480:
            for n in range(0, self.lab_ln):
                annot = misc.imresize(annot, (h, w), interp='nearest')

        annot.astype(np.float32)

        annot_im = np.zeros((self.lab_ln, h, w)).astype(np.float32)

        for n in range(0, self.lab_ln):
            annot_im[n, (annot == n)] = 1

        return annot_im

    # Loads files after minibatch iteration
    def load_files(self, X_files, y_files, num_samples):
        X = np.zeros((num_samples, self.nc, 360, 480)).astype(np.float32)
        y = np.zeros((num_samples, self.lab_ln, 360, 480)).astype(np.float32)
        y_small = np.zeros((num_samples, self.lab_ln, 45, 60)).astype(np.float32)
        if num_samples == 1:
            return np.expand_dims(self.preprocess_im(X_files), axis=0), \
                   np.expand_dims(self.preprocess_annot(y_files, 360, 480), axis=0), \
                   np.expand_dims(self.preprocess_annot(y_files, 45, 60), axis=0)
        else:
            for im in range(0, num_samples):
                X[im, :, :, :] = self.preprocess_im(X_files[im])
                y[im, :, :, :] = self.preprocess_annot(y_files[im], 360, 480)
                y_small[im, :, :, :] = self.preprocess_annot(y_files[im], 45, 60)
        return X, y, y_small

    def init_dataset(self):

        print("Loading Dataset...")

        # Get all training images and corresponding labels
        base = self.dir
        mode = "test"
        files = os.listdir(base + mode)
        X_files_test = []
        y_files_test = []

        for fl in files:

            # Load image
            X_files_test.append(base + mode + '/' + fl)
            y_files_test.append(base + mode + 'annot/' + fl)

        mode = "val"
        files = os.listdir(base + mode)
        X_files_val = []
        y_files_val = []

        for fl in files:
            # Load image
            X_files_val.append(base + mode + '/' + fl)
            y_files_val.append(base + mode + 'annot/' + fl)

        mode = "train"
        files = os.listdir(base + mode)
        X_files_train = []
        y_files_train = []

        for fl in files:
            # Load image
            X_files_train.append(base + mode + '/' + fl)
            y_files_train.append(base + mode + 'annot/' + fl)

        print("...Done")

        return np.array(X_files_train), np.array(y_files_train), np.array(X_files_val), np.array(y_files_val)\
            , np.array(X_files_test), np.array(y_files_test)
