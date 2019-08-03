# Author: Soheil Khorram
# License: Simplified BSD

import numpy as np
from logger import Logger

class Metric(object):

    @staticmethod
    def calc_rmse_one(y_true, y_pred):
        """
        Calculates root mean square error between two arrays
        This function removes the zero padded samples in its calculation
        """
        mask = (np.sum(y_true ** 2, 1) > 1e-10)
        y_true_mask = y_true[mask, :]
        y_pred_mask = y_pred[mask, :]
        rmse = np.sqrt(np.mean((y_true_mask - y_pred_mask) ** 2))
        return rmse

    @staticmethod
    def calc_rmse_mean_std(y_true_list, y_pred_list):
        """
        Calculates mean and std of the rmse values between y_true_list and y_pred_list
        """
        rmses = np.array(
            [Metric.calc_rmse_one(y_true_list[i, :, :], y_pred_list[i, :, :]) 
                for i in range(y_true_list.shape[0])]
            )
        return (np.mean(rmses), np.std(rmses))

    @staticmethod
    def calc_ecm(x1_inp, x2_inp, elimination_factor=0.1):
        """
        Calculates ECM between two arrays
        This function removes the zero padded samples in its calculation
        """
        mask = (np.sum(x1_inp ** 2, 1) > 1e-10)
        x1 = x1_inp[mask, :]
        x2 = x2_inp[mask, :]
        frame_num = x1.shape[0]
        FS = 8000
        FRAME_SHIFT = 16
        BLOCK_IN_MS = 128
        blockInSample = (BLOCK_IN_MS * FS) / 1000
        blockSize = int(blockInSample // FRAME_SHIFT)
        numberOfBlocks = int(frame_num // blockSize - 1)
        corr2Mat = np.zeros((numberOfBlocks))
        currentIndex = 0
        for i in range(numberOfBlocks):
            corr2Mat[i] = (
                Metric.corr2(
                    x1[currentIndex: currentIndex + blockSize, :],
                    x2[currentIndex: currentIndex + blockSize, :],
                )
                ) ** 2
            currentIndex = currentIndex + blockSize
        corr2Mat = corr2Mat[np.invert(np.isnan(corr2Mat))]
        itemsExclude = int(np.round(elimination_factor * numberOfBlocks))
        sortedCorr = np.array(sorted(corr2Mat))
        sortedCorr = sortedCorr[itemsExclude:]
        if len(sortedCorr) == 0:
            return 0
        sortedCorr = sortedCorr[np.invert(np.isnan(sortedCorr))]
        finalCorr = np.mean(sortedCorr)
        return finalCorr

    @staticmethod
    def mean2(x):
        """Used in corr2 method"""
        y = np.sum(x) / np.size(x)
        return y

    @staticmethod
    def corr2(a, b):
        """Used in calc_ecm method"""
        a = a - Metric.mean2(a)
        b = b - Metric.mean2(b)
        r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
        return r

    @staticmethod
    def calc_ecm_mean_std(y_true, y_pred, elimination_factor=0.1):
        ecms = np.array(
            [Metric.calc_ecm(y_true[i, :, :], y_pred[i, :, :], elimination_factor) for i in range(y_true.shape[0])]
            )
        return np.mean(ecms), np.std(ecms)

    @staticmethod
    def report_all_metrics(data):
        selection_metric = 0
        sets = ['tr', 'de', 'te']
        X = data.Xtr, data.Xde, data.Xte
        Y = data.Ytr, data.Yde, data.Yte
        for i in range(3):
            mu, sigma = Metric.calc_rmse_mean_std(X[i], Y[i])
            Logger.write_log('    rmse_{}: {} +- {}'.format(sets[i], mu, sigma))
        for i in range(3):
            mu, sigma = Metric.calc_ecm_mean_std(X[i], Y[i], 0.1)
            Logger.write_log('    ecm_{}: {} +- {}'.format(sets[i], mu, sigma))
            if i == 1:
                selection_metric = mu
        for i in range(3):
            mu, sigma = Metric.calc_ecm_mean_std(X[i], Y[i], 0)
            Logger.write_log('    ecm0_{}: {} +- {}'.format(sets[i], mu, sigma))
        return selection_metric
