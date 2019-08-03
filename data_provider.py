# Author: Soheil Khorram
# License: Simplified BSD

import numpy as np
import scipy.io
import os


class DataProvider(object):

    def __init__(self):
        """
        Initializes a data provider that is able to provide data for training
        """
        self.Xtr = None
        self.Xde = None
        self.Xte = None
        self.Ytr = None
        self.Yde = None
        self.Yte = None
        self.filestr = None
        self.filesde = None
        self.fileste = None

    def copy_from(self, data):
        """
        One level copy of the DataProvider.
        """
        self.Xtr = data.Xtr
        self.Xde = data.Xde
        self.Xte = data.Xte
        self.Ytr = data.Ytr
        self.Yde = data.Yde
        self.Yte = data.Yte
        self.filestr = data.filestr
        self.filesde = data.filesde
        self.fileste = data.fileste

    def load(self, prm):
        """
        Loads all data samples to in the memory
        For big datasets you can not load everything in the memory
        """
        files = []
        for fi in range(prm.inp_num):
            files.append(prm.dataset_path + '/noisy' + str(fi + 1) + '.mat')
        self.Xtr = np.zeros([int(round(len(files)/2)), prm.inp_len, prm.inp_dim])
        self.filestr = ['' for i in range(int(round(len(files)/2)))]
        self.Xde = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
        self.filesde = ['' for i in range(int(round(len(files)/4)))]
        self.Xte = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
        self.fileste = ['' for i in range(int(round(len(files)/4)))]

        for i in range(int(round(len(files)/4))):
            x = scipy.io.loadmat(files[i*4])['data']
            self.Xtr[i * 2, :, :] = x.T
            self.filestr[i*2] = files[i*4]
            x = scipy.io.loadmat(files[i*4+1])['data']
            self.Xtr[i * 2 + 1, :, :] = x.T
            self.filestr[i*2+1] = files[i*4+1]

            x = scipy.io.loadmat(files[i*4+2])['data']
            self.Xde[i, :, :] = x.T
            self.filesde[i] = files[i*4+2]

            x = scipy.io.loadmat(files[i*4+3])['data']
            self.Xte[i, :, :] = x.T
            self.fileste[i] = files[i*4+3]

        files = []
        for fi in range(prm.inp_num):
            files.append(prm.dataset_path + '/clean' + str(fi + 1) + '.mat')
        self.Ytr = np.zeros([int(round(len(files)/2)), prm.inp_len, prm.inp_dim])
        self.Yde = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
        self.Yte = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
        for i in range(int(round(len(files)/4))):
            y = scipy.io.loadmat(files[i*4])['data']
            self.Ytr[i*2, :, :] = y.T
            y = scipy.io.loadmat(files[i*4+1])['data']
            self.Ytr[i*2 + 1, :, :] = y.T
            y = scipy.io.loadmat(files[i*4+2])['data']
            self.Yde[i, :, :] = y.T
            y = scipy.io.loadmat(files[i*4+3])['data']
            self.Yte[i, :, :] = y.T

    def write_feats(self, prm):
        DataProvider.write_feats_one_set(self.Ytr, prm.filestr, prm.out_feat_path + '/tr/Predict')
        DataProvider.write_feats_one_set(self.Yde, prm.filesde, prm.out_feat_path + '/de/Predict')
        DataProvider.write_feats_one_set(self.Yte, prm.fileste, prm.out_feat_path + '/te/Predict')

    @staticmethod
    def write_feats_one_set(X, filestr, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if X.shape[0] != len(filestr):
            raise Exception('Error in writing: X.shape[0] != len(filestr)')
        for i in range(X.shape[0]):
            tempind = filestr[i].rfind('/')
            temp = filestr[i][tempind+1:]
            temp.replace('noisy', 'predicted')
            temp.replace('clean', 'predicted')
            # print(str(i))
            DataProvider.write_one_feature_vector(X[i, :, :], dir + '/' + temp)

    @staticmethod
    def write_one_feature_vector(x, path):
        data = {}
        data['data'] = x
        scipy.io.savemat(path, data)
