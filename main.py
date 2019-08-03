# Author: Soheil Khorram
# License: Simplified BSD

from conv_network import ConvNetwork
from dilated_network import DilatedNetwork
from data_provider import DataProvider
from metric import Metric
from numpy.random import seed
import os
import argparse
from logger import Logger
import json


def set_parameters(net):
    parser = argparse.ArgumentParser(description='neural net training script')
    parser.add_argument('-dataset-path', default='', type=str)
    parser.add_argument('-result-path', default='', type=str)
    parser.add_argument('-inp-num', default=6300, type=int)
    parser.add_argument('-inp-len', default=2500, type=int)
    parser.add_argument('-inp-dim', default=22, type=int)
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-max-epochs_num', default=1000, type=int)
    parser.add_argument('-max-patience', default=5, type=int)
    parser.add_argument('-step-size', default=0.0001, type=float)
    parser.add_argument('-exp-name', default='exp', type=str)
    net.load_parameters(parser)
    prm = parser.parse_args()
    prm.model = net.get_name()
    prm.exp_path = prm.result_path + '/' + prm.model + '/' + prm.exp_name
    prm.model_path = prm.exp_path + '/net.h5'
    prm.out_feat_path = prm.exp_path + '/out_feats'
    return prm


seed(1)
MODEL = os.environ["MODEL"]
if MODEL == 'conv':
    net = ConvNetwork()
elif MODEL == 'dilated':
    net = DilatedNetwork()
else:
    raise Exception('MODEL global variable must be conv or dilated, but it is {}'.format(MODEL))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
prm = set_parameters(net)
if os.path.isfile(prm.model_path):
    print('You have conducted this experiment before ...')
    exit(0)
Logger.set_path(prm.exp_path + '/log.txt')
Logger.write_log('MODEL = ' + MODEL)
Logger.write_log('Parameters: \n' + json.dumps(vars(prm), indent=4))
data = DataProvider()
data.load(prm)
Metric.report_all_metrics(data)
net.train(data, prm)
net.save(prm.model_path)
data_pred = net.evaluate(data, prm)
Metric.report_all_metrics(data_pred)
data_pred.write_feats(prm)
