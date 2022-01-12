"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# Copyright (c) 2021 Kaen Chan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from evaluation.pyeer.eer_info import get_eer_stats
from mc_dropout.network_mcdropout import NetworkMCDropout
from utils import utils
from dataset.dataset_np_ipc import Dataset

import os
import argparse
import sys
import numpy as np
from scipy import misc
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
import _pickle as cPickle
from utils.utils import KFold


def calculate_eer(embeddings1, embeddings2, actual_issame, compare_func, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    dist = compare_func(embeddings1, embeddings2)
    gscores_a = dist[actual_issame == 1]
    iscores_a = dist[actual_issame == 0]
    stats_a = get_eer_stats(gscores_a, iscores_a)
    return stats_a


def evaluate(embeddings, actual_issame, compare_func, nrof_folds=10, keep_idxes=None):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual_issame = np.asarray(actual_issame)
    if keep_idxes is not None:
        embeddings1 = embeddings1[keep_idxes]
        embeddings2 = embeddings2[keep_idxes]
        actual_issame = actual_issame[keep_idxes]
    return calculate_eer(embeddings1, embeddings2,
                         actual_issame, compare_func, nrof_folds=nrof_folds)


def load_bin(path, image_size):
  print(path, image_size)
  with open(path, 'rb') as f:
      if 'lfw_all' in path:
          bins, issame_list = pickle.load(f)
      else:
          bins, issame_list = pickle.load(f, encoding='latin1')
  data_list = []
  for flip in [0]:
    data = nd.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
    data_list.append(data)
  print(len(bins))
  for i in range(len(issame_list)*2):
    _bin = bins[i]
    # print(type(_bin))
    img = mx.image.imdecode(_bin)
    # img = nd.transpose(img, axes=(2, 0, 1))
    for flip in [0]:
      if flip==1:
        img = mx.ndarray.flip(data=img, axis=2)
      data_list[flip][i][:] = img
    # if i%5000==0:
    #   print('loading bin', i)
  print(data_list[0].shape)
  return (data_list, issame_list)


def extract_features(images_preprocessed, issame_list, extract_feature_func, batch_size, name='', result_dir='',
                re_extract_feature=True):
    print('testing verification..')
    if name:
        save_name_pkl_feature = result_dir + '/%s_feature.pkl' % name
    if re_extract_feature or not os.path.exists(save_name_pkl_feature):
        images = images_preprocessed
        print(images.shape)
        mu, sigma_sq = extract_feature_func(images)
        save_data = (mu, sigma_sq, issame_list)
        if name:
            with open(save_name_pkl_feature, 'wb') as f:
                cPickle.dump(save_data, f)
            print('save', save_name_pkl_feature)
    else:
        with open(save_name_pkl_feature, 'rb') as f:
            data = cPickle.load(f)
        if len(data) == 3:
            mu, sigma_sq, issame_list = data
        else:
            mu, sigma_sq = data
        print('load', save_name_pkl_feature)
    return mu, sigma_sq, issame_list


def eval_images_with_sigma(mu, sigma_sq, issame_list, nfolds=10, name='', filter_out_type='max', sigma_sizes=1):
    print('sigma_sq', sigma_sq.shape)
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)

    if name != '':
        np.save('o_sigma_%s.npy' % name, sigma_sq)

    # quality_score = -np.mean(np.log(sigma_sq), axis=1)
    # print('quality_score quality_score=-np.mean(np.log(sigma_sq),axis=1) percentile [0, 10, 30, 50, 70, 90, 100]')
    # print('quality_score ', np.percentile(quality_score.ravel(), [0, 10, 30, 50, 70, 90, 100]))

    s = 'sigma_sq ' + str(np.percentile(sigma_sq.ravel(), [0, 10, 30, 50, 70, 90, 100])) + \
        ' percentile [0, 10, 30, 50, 70, 90, 100]\n'
    # print(mu.shape)

    # print('sigma_sq', sigma_sq.shape)
    if sigma_sq.shape[1] == 2:
        sigma_sq_c = np.copy(sigma_sq)
        sigma_sq_list = [sigma_sq_c[:,:1], sigma_sq_c[:,1:]]
    elif type(sigma_sizes) == list:
        sigma_sq_list = []
        idx = 0
        for si in sigma_sizes:
            sigma = sigma_sq[:, idx:idx + si]
            if si > 1:
                sigma = 1/np.mean(1/(sigma+1e-6), axis=-1)
            sigma_sq_list += [sigma]
            idx += si
    elif sigma_sq.shape[1] > 2:
        sigma_sq_list = [1/np.mean(1/(sigma_sq+1e-6), axis=-1)]
    else:
        sigma_sq_list = [sigma_sq]
    for sigma_sq in sigma_sq_list:
        sigma_sq1 = sigma_sq[0::2]
        sigma_sq2 = sigma_sq[1::2]
        # filter_out_type = 'max'
        if filter_out_type == 'max':
            sigma_fuse = np.maximum(sigma_sq1, sigma_sq2)
        else:
            sigma_fuse = sigma_sq1 + sigma_sq2
        # reject_factor = 0.1
        error_list = []
        # reject_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # reject_factors = np.arange(50) / 100.
        # reject_factors = np.arange(30) / 100.
        # reject_factors = [0.0, 0.1, 0.2, 0.3]
        reject_factors_points = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        reject_factors = np.arange(0, 1.0, 0.01)
        for reject_factor in reject_factors:
            risk_threshold = np.percentile(sigma_fuse.ravel(), (1-reject_factor)*100)
            keep_idxes = np.where(sigma_fuse <= risk_threshold)[0]
            if len(keep_idxes) == 0:
                keep_idxes = None

            stats = evaluate(mu, issame_list, utils.pair_cosin_score, nrof_folds=nfolds, keep_idxes=keep_idxes)
            if reject_factor in reject_factors_points:
                s += 'reject_factor {:.4f} '.format(reject_factor)
                s += 'risk_threshold {:.6f} '.format(risk_threshold)
                s += 'keep_idxes {} / {} '.format(len(keep_idxes), len(sigma_fuse))
                s += 'Cosine score eer %f fmr100 %f fmr1000 %f\n' % (stats.eer, stats.fmr100, stats.fmr1000)
            error_list += [stats.fmr1000]
            # print('cosin', 'acc', accuracy, 'threshold', threshold)
            # print(s)
            # compare_func = lambda x,y: utils.pair_MLS_score(x, y, use_attention_only=False)
            # accuracy, threshold = evaluate(feat_pfe, issame_list, compare_func, nrof_folds=nfolds, keep_idxes=keep_idxes)
            # s += 'MLS score acc %f threshold %f' % (accuracy, threshold)
            # print('MLS', 'acc', accuracy, 'threshold', threshold)
            if keep_idxes is None:
                break
        # s_avg = 'reject_factor 0.5 risk_threshold 0.585041 keep_idxes 3500 / 7000 '
        s_avg = 'reject_factor mean --------------------------------------------- '
        s_avg += 'Cosine score fmr1000 %f\n' % (np.mean(error_list))
        s += s_avg
        tpr = error_list
        fpr = reject_factors
        auc = sklearn.metrics.auc(fpr, tpr)
        l = int(len(tpr)*0.3)
        auc30 = sklearn.metrics.auc(fpr[:l], tpr[:l])
        s += 'AUERC: %1.4f\n' % auc
        s += 'AUERC30: %1.4f\n' % auc30
        best = error_list[0]**2/2
        s += 'AUC: %1.4f\n' % (auc-best)
        best30 = (error_list[0] * min(error_list[0], 0.3))/2
        s += 'AUC30: %1.4f\n' % (auc30-best30)
        s += '\n'
    # print(s)
    return s[:-1]


def eval_images_mls(mu, sigma_sq_i, issame_list, nfolds=10, sigma_sizes=1):
    print('sigma_sq', sigma_sq_i.shape)
    if sigma_sq_i.shape[1] == 2:
        sigma_sq_c = np.copy(sigma_sq_i)
        sigma_sq_list = [sigma_sq_c[:,:1], sigma_sq_c[:,1:]]
    elif type(sigma_sizes) == list:
        sigma_sq_list = []
        idx = 0
        for si in sigma_sizes:
            sigma_sq_list += [sigma_sq_i[:, idx:idx+si]]
            idx += si
    else:
        sigma_sq_list = [sigma_sq_i]
    s = ''
    ret = {}
    accuracy, threshold = evaluate(mu, issame_list, utils.pair_cosin_score, nrof_folds=nfolds)
    ret['Cosine'] = accuracy
    s += 'Cosine score acc %.4f threshold %.4f\n' % (accuracy, threshold)
    for sigma_sq in sigma_sq_list:
        print('testing verification..')
        feat_pfe = np.concatenate([mu, sigma_sq], axis=1)
        # quality_score = -np.mean(np.log(sigma_sq), axis=1)
        compare_func = lambda x, y: utils.pair_MLS_score(x, y, use_attention_only=False)
        accuracy, threshold = evaluate(feat_pfe, issame_list, compare_func, nrof_folds=nfolds)
        s += 'MLS    score acc %.4f threshold %.4f\n' % (accuracy, threshold)
        ret['MLS'] = accuracy
    # print(s)
    return s, ret


def eval_images(images_preprocessed, issame_list, extract_feature_func, batch_size, nfolds=10, name='', result_dir='',
                re_extract_feature=True, filter_out_type='max', sigma_sizes=1, tt_flip=False, only_mls=True):
    mu, sigma_sq, issame_list = extract_features(images_preprocessed, issame_list, extract_feature_func, batch_size,
                                                 name=name, result_dir=result_dir,
                                                 re_extract_feature=re_extract_feature)
    # s_mls, ret = eval_images_mls(mu, sigma_sq, issame_list, nfolds=nfolds, sigma_sizes=sigma_sizes)
    s_mls, ret = '', [0]
    info = s_mls
    if not only_mls:
        s_reject = eval_images_with_sigma(mu, sigma_sq, issame_list, nfolds=nfolds, name='',
                                          filter_out_type=filter_out_type, sigma_sizes=sigma_sizes)
        info = s_mls + s_reject
    if tt_flip:
        info = info.replace('Cosine score acc', 'tt-flip Cosine score acc')
        info = info.replace('MLS    score acc', 'tt-flip MLS    score acc')
    return info, ret


def save_dataset_as_jpg(data_set, name):
    data_list = data_set[0]
    issame_list = data_set[1]
    data_list = data_list[0].asnumpy()
    root = r'F:\data\face-recognition\test\1v1'
    for i in range(len(data_list)):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%04d_%d.jpg' % (i, issame_list[i//2]))
        print(path)
        cv2.imwrite(path, data_list[i].astype(np.uint8)[...,::-1])


def eval(data_set, network, batch_size, nfolds=10, name='', result_dir='', re_extract_feature=True,
         filter_out_type='max', sigma_sizes=1, tt_flip=False):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    data_list = data_list[0].asnumpy()
    # images = preprocess(data_list, network.config, False)
    images = data_list
    del data_set
    for i in range(1):
        # name1 = name + '_keep0.9_%03d' % i
        name1 = name
        extract_feature_func = lambda x: network.extract_feature(x, batch_size=32, need_preprecess=True,
                                                                 tt_flip=tt_flip, verbose=True)
        ret, _ = eval_images(images, issame_list, extract_feature_func, batch_size, nfolds=nfolds, name=name1,
                          result_dir=result_dir, re_extract_feature=re_extract_feature,
                          filter_out_type=filter_out_type, sigma_sizes=sigma_sizes, tt_flip=tt_flip, only_mls=False)
        # ret = eval_images_cmp(images, issame_list, network, batch_size, nfolds=10, name=name, result_dir=result_dir,
        #                       re_extract_feature=re_extract_feature, filter_out_type=filter_out_type)
    return ret


def main_save_data(args):
    data_dir = args.dataset_path
    data_dir = r'F:\data\face-recognition\MS-Celeb-1M\faces_emore'
    data_dir = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'
    for name in args.target.split(','):
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
            image_size = [112, 112]
            data_set = load_bin(path, image_size)
            save_dataset_as_jpg(data_set, name)


def eval_images_from_pkl_all_data(model_dir, targets=None):
    filter_out_type = 'max'
    densefilter = False
    seperate = False
    nfolds = 10
    # featmodels = ['r64', 'r100', 'r50']
    # featmodels = ['r100', 'r50']
    featmodels = ['same']
    for featmodel in featmodels:
        mu_list = []
        idq_list = []
        iss_list = []
        if targets is None:
            targets = 'lfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp'
        for target in targets.split(','):
            path_pkl = model_dir + r'\%s_feature.pkl' % target
            with open(path_pkl, 'rb') as f:
                data = cPickle.load(f)
            mu2, id_quality, issame_list = data
            save_name_pkl_feature = None
            if featmodel == 'same':
                save_name_pkl_feature = r'{}\{}_feature.pkl'.format(model_dir, target)
            elif featmodel == 'r100':
                save_name_pkl_feature = r'F:\data\face-recognition\test\IJB_release\pretrained_models\MS1MV2-ResNet100-Arcface\{}_feature.pkl'.format(target)
            elif featmodel == 'r50':
                save_name_pkl_feature = r'F:\data\face-recognition\test\IJB_release\pretrained_models\VGG2-ResNet50-Arcface\{}_feature.pkl'.format(target)
            elif featmodel == 'r64':
                save_name_pkl_feature = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE_mbv3small_reject\20210116-040122-s16-m0.4-tau1\{}_feature.pkl'.format(
                    target)
            else:
                raise ('error', save_name_pkl_feature)
            with open(save_name_pkl_feature, 'rb') as f:
                data = cPickle.load(f)
            mu2, _, _ = data
            print('load verification model', save_name_pkl_feature)

            if seperate:
                s = eval_images_with_sigma(mu2, id_quality, issame_list, nfolds=nfolds, name='',
                                           filter_out_type=filter_out_type, densefilter=densefilter)
                print(s)
                logname = 'testing-log-fnmr-{}-{}.txt'.format(target, featmodel)
                if densefilter:
                    logname = 'testing-log-fnmr-{}-{}-densefilter.txt'.format(target, featmodel)
                with open(os.path.join(model_dir, logname), 'a') as f:
                    if save_name_pkl_feature is not None:
                        f.write(save_name_pkl_feature + '\n')
                    f.write(targets + '\n')
                    f.write(s + '\n')
                continue

            mu_list += list(mu2)
            idq_list += list(id_quality)
            iss_list += list(issame_list)
            print('load', path_pkl)
            # s = eval_images_with_sigma(mu, id_quality, issame_list, nfolds=10, name='', filter_out_type=filter_out_type)
            # print(s)
        if seperate:
            continue
        mu_list = np.array(mu_list)
        idq_list = np.array(idq_list)
        iss_list = np.array(iss_list)
        print('pos', np.sum(iss_list), 'neg', len(iss_list)-np.sum(iss_list), 'total', len(iss_list))
        if id_quality.shape[1] == 513:
            sigma_sizes = [512, 1]
        else:
            sigma_sizes = 1
        s, ret = eval_images_mls(mu_list, idq_list, iss_list, nfolds=10, sigma_sizes=sigma_sizes)
        s += eval_images_with_sigma(mu_list, idq_list, iss_list, nfolds=nfolds, name='',
                                    filter_out_type=filter_out_type, sigma_sizes=sigma_sizes)
        print(s)
        logname = 'testing-log-fnmr-{}-{}.txt'.format('alldata', featmodel)
        if densefilter:
            logname = 'testing-log-fnmr-{}-{}-densefilter.txt'.format('alldata', featmodel)
        with open(os.path.join(model_dir, logname), 'a') as f:
            if save_name_pkl_feature is not None:
                f.write(save_name_pkl_feature + '\n')
            f.write(targets + '\n')
            f.write(s + '\n')
        with open(os.path.join(model_dir, 'testing-log.txt'), 'a') as f:
            if save_name_pkl_feature is not None:
                f.write(save_name_pkl_feature + '\n')
            f.write(targets + '\n')
            f.write(s + '\n')


def main(args):
    data_dir = args.dataset_path
    # data_dir = r'F:\data\face-recognition\MS-Celeb-1M\faces_emore'
    # data_dir = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'
    # data_dir = r'F:\data\metric-learning\face\ms1m-retinaface-t1'
    re_extract_feature = True
    # filter_out_type = 'add'
    filter_out_type = 'max'
    tt_flip = False

    # Load model files and config file
    network = NetworkMCDropout(args.backbone_name, args.resume_backbone)

    # # images = np.random.random([1, 128, 128, 3])
    # images = np.random.random([1, 3, 112, 112])
    # img = cv2.imread(r'E:\chenkai\probface-pytorch\im_96x96.jpg')
    # images = np.array([img])
    # for _ in range(1):
    #     mu, sigma_sq = network.extract_feature(images, 1, need_preprecess=True, tt_flip=True, verbose=True)
    #     print(mu[0, :5])
    # exit(0)
    # log_dir = r'E:\chenkai\face-uncertainty-pytorch\mc_dropout'
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print(args.target)
    for namec in args.target.split(','):
        path = os.path.join(data_dir,namec+".bin")
        if os.path.exists(path):
            print(path)
            image_size = [112, 112]
            data_set = load_bin(path, image_size)
            name = namec
            print('ver', name)
            # save_pkl_name = ''   # donot save feature.pkl
            save_pkl_name = namec
            print(log_dir)
            sigma_sizes = network.uncertainty_size
            info = eval(data_set, network, args.batch_size, 10, name=save_pkl_name, result_dir=log_dir,
                        re_extract_feature=re_extract_feature, filter_out_type=filter_out_type,
                        sigma_sizes=sigma_sizes, tt_flip=tt_flip)
            # print(info)
            info_result = '--- ' + name + ' ---\n'
            info_result += data_dir + '\n'
            info_result += info + "\n"
            print("")
            print(info_result)

            with open(os.path.join(log_dir, 'testing-log-fnmr-{}-{}.txt'.format(name, filter_out_type)), 'a') as f:
                f.write(info_result + '\n')
            with open(os.path.join(log_dir, 'testing-log-fnmr.txt'), 'a') as f:
                f.write(info_result + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", help="", type=str, default='')
    parser.add_argument("--resume_backbone", help="", type=str, default='')
    parser.add_argument("--log_dir", help="", type=str, default='')
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\metric-learning\face\ms1m-retinaface-t1')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=16)
    parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
    args = parser.parse_args()
    # args.dataset_path = r''
    dataset_path_list = [
        r'F:\data\metric-learning\face\ms1m-retinaface-t1',
        # r'F:\data\metric-learning\face\ms1m-retinaface-t1\blur-r5-p0.05',
    ]
    # log_dir = r'G:\chenkai\probface-pytorch\log\ir50_pfe\20210327-132611'
    # log_dir = r'G:\chenkai\probface-pytorch\log\ir50_pfe\20210327-181146-mls-cl1-0.15'
    # args.target = 'lfw,cfp_fp,agedb_30'
    args.target = 'calfw,cplfw,cfp_ff,vgg2_fp'
    args.target = 'lfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp'
    # args.target = 'calfw'

    args.dataset_path = dataset_path_list[0]

    # args.log_dir = r'E:\chenkai\face-uncertainty-pytorch\log\glint-ir50\mcdropout'
    # args.backbone_name = 'backbones.iresnet.iresnet50'
    # args.resume_backbone = r'E:\chenkai\arcface_torch\glint360k_cosface_r50_fp16_0.1\backbone.pth'
    # main(args)
    #
    # args.log_dir = r'E:\chenkai\face-uncertainty-pytorch\log\glint-ir100\mcdropout'
    # args.backbone_name = 'backbones.iresnet.iresnet100'
    # args.resume_backbone = r'E:\chenkai\arcface_torch\glint360k_cosface_r100_fp16_0.1\backbone.pth'
    # main(args)
    #
    # args.log_dir = r'E:\chenkai\face-uncertainty-pytorch\log\ms1mv3-ir50\mcdropout'
    # args.backbone_name = 'backbones.iresnet.iresnet50'
    # args.resume_backbone = r'E:\chenkai\arcface_torch\ms1mv3_arcface_r50_fp16\backbone.pth'
    # main(args)

    args.log_dir = r'E:\chenkai\face-uncertainty-pytorch\log\ms1mv3-ir50\mcdropout'
    args.backbone_name = 'backbones.iresnet.iresnet50'
    args.resume_backbone = r'E:\chenkai\arcface_torch\ms1mv3_arcface_r50_fp16\backbone.pth'
    main(args)
