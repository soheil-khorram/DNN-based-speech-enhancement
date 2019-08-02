from keras.layers import Input, Conv1D, Add, MaxPooling1D, UpSampling1D, multiply, add, Activation, Lambda
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import scipy.io
from numpy.random import seed
import os
import argparse
from logger import Logger
import copy
import json


def dilated_conv(prm, x, dilation_rate):
    return Conv1D(filters=prm.kernel_num,
                  kernel_size=prm.kernel_size,
                  strides=1,
                  padding=prm.padding,
                  data_format='channels_last',
                  dilation_rate=dilation_rate,
                  activation=prm.activation,
                  use_bias=True,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=l2(prm.l2),
                  bias_regularizer=l2(prm.l2),
                  activity_regularizer=None)(x)


def last_conv(prm, x):
    return Conv1D(filters=prm.inp_dim,
                  kernel_size=prm.kernel_size,
                  strides=1,
                  padding=prm.padding,
                  data_format='channels_last',
                  dilation_rate=1,
                  activation='linear',
                  use_bias=True,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=l2(prm.l2),
                  bias_regularizer=l2(prm.l2),
                  activity_regularizer=None)(x)


def construct_dilated(prm):
    inp = Input(shape=(prm.inp_len, prm.inp_dim))
    x = inp
    x = dilated_conv(prm, x, 1)
    for i in range(prm.feature_layer_num - 1):
        y = dilated_conv(prm, x, 1)
        if prm.skip == 1:
            x = Add()([x, y])
        else:
            x = y
    for i in range(prm.dilated_layer_num):
        y = dilated_conv(prm, x, 2 ** (i + 1))
        if prm.skip == 1:
            x = Add()([x, y])
        else:
            x = y
    x = last_conv(prm, x)
    outp = None
    if prm.bridge == 'nothing':
        outp = x
    elif prm.bridge == 'add':
        outp = add([x, inp])
    elif prm.bridge == 'mul':
        outp = multiply([x, inp])
    else:
        print('ERROR in bridge param')
    if prm.apply_end_relu == 1:
        outp = Activation('relu')(outp)
    net = Model(inputs=inp, outputs=outp)
    net.compile(optimizer=Adam(lr=prm.step_size),
                loss='mean_squared_error')
    return net


def load_parameters_dilated(parser):
    parser.add_argument('-skip', default=0, type=int)
    parser.add_argument('-dilated-layer-num', default=3, type=int)
    parser.add_argument('-feature-layer-num', default=5, type=int)
    parser.add_argument('-activation', default='tanh', type=str)
    parser.add_argument('-kernel-num', default=50, type=int)
    parser.add_argument('-kernel-size', default=5, type=int)
    parser.add_argument('-l2', default=0.0, type=float)
    parser.add_argument('-apply-end-relu', default=1, type=int)
    parser.add_argument('-bridge', default='nothing', type=str)
    parser.add_argument('-padding', default='same', type=str)


def construct_conv(prm):
    inp = Input(shape=(prm.inp_len, prm.inp_dim))
    x = inp
    for i in range(prm.layer_num):
        activation = 'tanh'
        kernel_num = prm.kernel_num
        if i == (prm.layer_num - 1):
            kernel_num = prm.inp_dim
            activation = None
        x = Conv1D(filters=kernel_num,
                   kernel_size=prm.kernel_size,
                   strides=1,
                   padding=prm.padding,
                   data_format='channels_last',
                   dilation_rate=1,
                   activation=activation,
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l2(prm.l2),
                   bias_regularizer=regularizers.l2(prm.l2),
                   activity_regularizer=None)(x)
    outp = None
    if prm.bridge == 'nothing':
        outp = x
    elif prm.bridge == 'add':
        outp = add([x, inp])
    elif prm.bridge == 'mul':
        outp = multiply([x, inp])
    else:
        print('ERROR in bridge param')
    if prm.apply_end_relu == 1:
        outp = Activation('relu')(outp)
    net = Model(inputs=inp, outputs=outp)
    net.compile(optimizer=Adam(lr=prm.step_size),
                loss='mean_squared_error')
    return net


def load_parameters_conv(parser):
    parser.add_argument('-layer-num', default=5, type=int)
    parser.add_argument('-kernel-size', default=7, type=int)
    parser.add_argument('-kernel-num', default=64, type=int)
    parser.add_argument('-l2', default=0.0, type=float)
    parser.add_argument('-apply-end-relu', default=1, type=int)
    parser.add_argument('-bridge', default='mul', type=str)
    parser.add_argument('-padding', default='causal', type=str)


def train(net, prm, Xtr, Ytr, Xde, Yde, Xte, Yte):
    best_corr_de = -1e100
    best_net = None
    patient = 0
    for it in range(prm.max_epochs_num):
        Logger.write_log('iter = ' + str(it))
        net.fit(x=Xtr,
                y=Ytr,
                batch_size=prm.batch_size,
                epochs=1,
                verbose=1,
                shuffle=True)
        [pred_Yde, pred_Yte, rmse_de, rmse_te, corr_de, corr_te, corr0_de, corr0_te] =\
            evaluate_report_all(net, Xtr, Xde, Xte, Ytr, Yde, Yte)
        if corr_de > best_corr_de:
            Logger.write_log('nice, model gets better ...')
            Logger.write_log('###############################')
            best_corr_de = corr_de
            best_net = copy.deepcopy(net)
            patient = 0
        else:
            Logger.write_log('oops, model gets worse ...')
            Logger.write_log('###############################')
            patient += 1
        if patient >= prm.max_patience:
            break
    return best_net


def evaluate(net, inputs):
    return net.predict(inputs, batch_size=prm.batch_size, verbose=1)


def save(net, path):
    net.save(path)


def load(path):
    return load_model(path)


def load_data(feat_path, prm):
    files = []
    for fi in range(prm.inp_num):
        files.append(feat_path + '/noisy' + str(fi + 1) + '.mat')
    Xtr = np.zeros([int(round(len(files)/2)), prm.inp_len, prm.inp_dim])
    filestr = ['' for i in range(int(round(len(files)/2)))]
    Xde = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
    filesde = ['' for i in range(int(round(len(files)/4)))]
    Xte = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
    fileste = ['' for i in range(int(round(len(files)/4)))]

    for i in range(int(round(len(files)/4))):
        x = scipy.io.loadmat(files[i*4])['data']
        Xtr[i * 2, :, :] = x.T
        filestr[i*2] = files[i*4]
        x = scipy.io.loadmat(files[i*4+1])['data']
        Xtr[i * 2 + 1, :, :] = x.T
        filestr[i*2+1] = files[i*4+1]

        x = scipy.io.loadmat(files[i*4+2])['data']
        Xde[i, :, :] = x.T
        filesde[i] = files[i*4+2]

        x = scipy.io.loadmat(files[i*4+3])['data']
        Xte[i, :, :] = x.T
        fileste[i] = files[i*4+3]

    files = []
    for fi in range(prm.inp_num):
        files.append(feat_path + '/clean' + str(fi + 1) + '.mat')
    Ytr = np.zeros([int(round(len(files)/2)), prm.inp_len, prm.inp_dim])
    Yde = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
    Yte = np.zeros([int(round(len(files)/4)), prm.inp_len, prm.inp_dim])
    for i in range(int(round(len(files)/4))):
        y = scipy.io.loadmat(files[i*4])['data']
        Ytr[i*2, :, :] = y.T
        y = scipy.io.loadmat(files[i*4+1])['data']
        Ytr[i*2 + 1, :, :] = y.T
        y = scipy.io.loadmat(files[i*4+2])['data']
        Yde[i, :, :] = y.T
        y = scipy.io.loadmat(files[i*4+3])['data']
        Yte[i, :, :] = y.T
    return [Xtr, Xde, Xte, Ytr, Yde, Yte, filestr, filesde, fileste]


def write_out_one_feat(x, path):
    data = {}
    data['data'] = x
    scipy.io.savemat(path, data)


def write_out_feats(X, filestr, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if X.shape[0] != len(filestr):
        print('ERRRROOOORRRRR: X.shape[0] != len(filestr)')
        exit(0)
    for i in range(X.shape[0]):
        tempind = filestr[i].rfind('/')
        temp = filestr[i][tempind+1:]
        temp.replace('noisy', 'predicted')
        temp.replace('clean', 'predicted')
        # print(str(i))
        write_out_one_feat(X[i, :, :], dir + '/' + temp)


def calc_rmse_one(y_true, y_pred):
    mask = (np.sum(y_true ** 2, 1) > 1e-10)
    y_true_mask = y_true[mask, :]
    y_pred_mask = y_pred[mask, :]
    rmse = np.sqrt(np.mean((y_true_mask - y_pred_mask) ** 2))
    return rmse


def calc_rmse_all(y_true, y_pred):
    return np.array([calc_rmse_one(y_true[i, :, :], y_pred[i, :, :]) for i in range(y_true.shape[0])])


def evaluate_report_all(net, Xtr, Xde, Xte, Ytr, Yde, Yte):
    if net is not None:
        # pred_Ytr = evaluate(net, Xtr)
        pred_Yde = evaluate(net, Xde)
        pred_Yte = evaluate(net, Xte)
    else:
        # pred_Ytr = Xtr
        pred_Yde = Xde
        pred_Yte = Xte
    # rmses_tr = calc_rmse_all(Ytr, pred_Ytr)
    rmses_de = calc_rmse_all(Yde, pred_Yde)
    rmses_te = calc_rmse_all(Yte, pred_Yte)
    # Logger.write_log('    rmse_tr: ' + str(np.mean(rmses_tr)) + ' +- ' + str(np.std(rmses_tr)))
    Logger.write_log('    rmse_de: ' + str(np.mean(rmses_de)) + ' +- ' + str(np.std(rmses_de)))
    Logger.write_log('    rmse_te: ' + str(np.mean(rmses_te)) + ' +- ' + str(np.std(rmses_te)))
    # corrs_tr = elect_corr_all(Ytr, pred_Ytr, 0.1)
    corrs_de = elect_corr_all(Yde, pred_Yde, 0.1)
    corrs_te = elect_corr_all(Yte, pred_Yte, 0.1)
    # Logger.write_log('    corr_tr: ' + str(np.mean(corrs_tr)) + ' +- ' + str(np.std(corrs_tr)))
    Logger.write_log('    corr_de: ' + str(np.mean(corrs_de)) + ' +- ' + str(np.std(corrs_de)))
    Logger.write_log('    corr_te: ' + str(np.mean(corrs_te)) + ' +- ' + str(np.std(corrs_te)))
    # corrs0_tr = elect_corr_all(Ytr, pred_Ytr, 0)
    corrs0_de = elect_corr_all(Yde, pred_Yde, 0)
    corrs0_te = elect_corr_all(Yte, pred_Yte, 0)
    # Logger.write_log('    corr0_tr: ' + str(np.mean(corrs0_tr)) + ' +- ' + str(np.std(corrs0_tr)))
    Logger.write_log('    corr0_de: ' + str(np.mean(corrs0_de)) + ' +- ' + str(np.std(corrs0_de)))
    Logger.write_log('    corr0_te: ' + str(np.mean(corrs0_te)) + ' +- ' + str(np.std(corrs0_te)))
    return [pred_Yde, pred_Yte,
            np.mean(rmses_de), np.mean(rmses_te),
            np.mean(corrs_de), np.mean(corrs_te),
            np.mean(corrs0_de), np.mean(corrs0_te)]


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    return r


def elect_corr_one(x1_inp, x2_inp, elimination_factor=0.1):
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
            corr2(
                x1[currentIndex: currentIndex + blockSize, :],
                x2[currentIndex: currentIndex + blockSize, :],
            )
            ) ** 2
        currentIndex = currentIndex + blockSize
    # print(numberOfBlocks)
    corr2Mat = corr2Mat[np.invert(np.isnan(corr2Mat))]
    itemsExclude = int(np.round(elimination_factor * numberOfBlocks))
    sortedCorr = np.array(sorted(corr2Mat))
    sortedCorr = sortedCorr[itemsExclude:]
    if len(sortedCorr) == 0:
        return 0
    sortedCorr = sortedCorr[np.invert(np.isnan(sortedCorr))]
    finalCorr = np.mean(sortedCorr)
    return finalCorr


def elect_corr_all(y_true, y_pred, elimination_factor=0.1):
    return np.array(
        [elect_corr_one(y_true[i, :, :], y_pred[i, :, :], elimination_factor) for i in range(y_true.shape[0])]
        )


MODEL = os.environ["MODEL"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser(description='neural net training script')
parser.add_argument('-data-dir', default='/home/khorram/Desktop/Mamun/data', type=str)
parser.add_argument('-dataset', default='Car_Noise_003/Features_car_n5', type=str)
parser.add_argument('-res-dir', default='/home/khorram/Desktop/Mamun/results', type=str)
parser.add_argument('-inp-num', default=6300, type=int)
parser.add_argument('-inp-len', default=2500, type=int)
parser.add_argument('-inp-dim', default=22, type=int)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-max-epochs_num', default=1000, type=int)
parser.add_argument('-max-patience', default=5, type=int)
parser.add_argument('-step-size', default=0.0001, type=float)
parser.add_argument('-exp-name', default='exp', type=str)
if MODEL == 'conv':
    load_parameters_conv(parser)
elif MODEL == 'dilated':
    load_parameters_dilated(parser)
else:
    print('ERRORR: unknown model.')
    exit()
prm = parser.parse_args()
FEAT_PATH = prm.data_dir + '/' + prm.dataset
EXP_PATH = prm.res_dir + '/' + prm.dataset + '/' + MODEL + '/' + prm.exp_name
MODEL_PATH = EXP_PATH + '/net.h5'
if os.path.isfile(MODEL_PATH):
    print('You have finished this experiment before ...')
    exit(0)
OUT_FEAT_PATH = EXP_PATH + '/out_feats'
Logger.set_path(EXP_PATH + '/log.txt')
Logger.write_log('MODEL = ' + MODEL)
Logger.write_log('Parameters: \n' + json.dumps(vars(prm), indent=4))
seed(1)
[Xtr, Xde, Xte, Ytr, Yde, Yte, filestr, filesde, fileste] = load_data(FEAT_PATH, prm)
[pred_Yde, pred_Yte, rmse_de, rmse_te, corr_de, corr_te, corr0_de, corr0_te] = \
    evaluate_report_all(None, Xtr, Xde, Xte, Ytr, Yde, Yte)
net = None
if MODEL == 'conv':
    net = construct_conv(prm)
elif MODEL == 'dilated':
    net = construct_dilated(prm)
else:
    print('ERRORR: unknown model.')
    exit()
net = train(net, prm, Xtr, Ytr, Xde, Yde, Xte, Yte)
save(net, MODEL_PATH)
[pred_Yde, pred_Yte, rmse_de, rmse_te, corr_de, corr_te, corr0_de, corr0_te] = \
    evaluate_report_all(net, Xtr, Xde, Xte, Ytr, Yde, Yte)

write_out_feats(pred_Yte, fileste, OUT_FEAT_PATH + '/te/Predict')
write_out_feats(Yte, fileste, OUT_FEAT_PATH + '/te/Clean')
write_out_feats(Xte, fileste, OUT_FEAT_PATH + '/te/Noisy')
write_out_feats(pred_Yde, filesde, OUT_FEAT_PATH + '/de/Predict')
write_out_feats(Yde, filesde, OUT_FEAT_PATH + '/de/Clean')
write_out_feats(Xde, filesde, OUT_FEAT_PATH + '/de/Noisy')
