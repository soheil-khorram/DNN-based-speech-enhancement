# Author: Soheil Khorram
# License: Simplified BSD

from logger import Logger
from data_provider import DataProvider
from metric import Metric
import copy
from keras.models import load_model


class Network(object):
    def __init__(self):
        self.net = None

    def construct(self):
        """
        Constructs the network graph and stores it in self.net
        """
        pass

    def train(self, data, prm):
        best_corr_de = -1e100
        best_net = None
        patient = 0
        for it in range(prm.max_epochs_num):
            Logger.write_log('iter = ' + str(it))
            self.net.fit(x=data.Xtr,
                    y=data.Ytr,
                    batch_size=prm.batch_size,
                    epochs=1,
                    verbose=1,
                    shuffle=True)
            data_pred = self.evaluate(data, prm)
            corr_de = Metric.report_all_metrics(data_pred)
            if corr_de > best_corr_de:
                Logger.write_log('nice, model gets better ...')
                Logger.write_log('###############################')
                best_corr_de = corr_de
                best_net = copy.deepcopy(self.net)
                patient = 0
            else:
                Logger.write_log('oops, model gets worse ...')
                Logger.write_log('###############################')
                patient += 1
            if patient >= prm.max_patience:
                break
        self.net = best_net

    def evaluate(self, data, prm):
        data_pred = DataProvider()
        data_pred.copy_from(data)
        data_pred.Xtr = self.net.predict(data_pred.Xtr, batch_size=prm.batch_size, verbose=1)
        data_pred.Xde = self.net.predict(data_pred.Xde, batch_size=prm.batch_size, verbose=1)
        data_pred.Xte = self.net.predict(data_pred.Xte, batch_size=prm.batch_size, verbose=1)
        return data_pred

    def save(self, path):
        self.net.save(path)

    def load(self, path):
        self.net = load_model(path)
