import argparse
import os,time
import sys
import torch
import logging
import torch.optim as optim
from easydict import EasyDict
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import imp

from dataset.build_dataset import build_dataset
from dataset.imageprocessing import preprocess_imgs_np
from models.build_model import build_model_backbone, build_model_uncertainty, build_model_backbone_mcdropout
from evaluation.verify_wrapper import VerifyWrapper
from utils.utils import create_log_dir, save_model_last, load_model_last, fix_bn, trainlog, dt_str, display_watch_list


class NetworkMCDropout(object):
    def __init__(self, backbone_name='', resume_backbone=''):
        self.backbone_name = backbone_name
        self.resume_backbone = resume_backbone
        # self.backbone_name = 'backbones.iresnet.iresnet50'
        # self.resume_backbone = r'E:\chenkai\arcface_torch\glint360k_cosface_r50_fp16_0.1\backbone.pth'
        self.uncertainty_module_chs = [512*7*7]
        self.embedding_size = 512
        self.uncertainty_size = 1
        self.build_model()

    def build_model(self):
        self.model_backbone, self.model_uncertainty = build_model_backbone_mcdropout(
            self.backbone_name, self.uncertainty_module_chs, resume=self.resume_backbone)
        # single-GPU setting
        self.model_backbone = self.model_backbone.cuda()
        self.model_uncertainty = self.model_uncertainty.cuda()
        self.model_backbone.eval()
        self.model_backbone.apply(fix_bn)
        self.model_uncertainty.eval()

    def extract_feature(self, images, batch_size, need_preprecess=False, tt_flip=False, verbose=False):
        # forward
        # inputs = Variable(inputs.cuda())
        # feature, sig_feat = self.model_backbone(inputs)
        # log_sig_sq = self.model_uncertainty(sig_feat)
        num_images = len(images)
        mu = np.ndarray((num_images, self.embedding_size), dtype=np.float32)
        sigma_sq = np.ndarray((num_images, self.uncertainty_size), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r'
                                 % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            images_batch = images[start_idx:end_idx]
            if need_preprecess:
                images_batch = preprocess_imgs_np(images_batch, is_training=False)
            with torch.no_grad():
                inputs = torch.from_numpy(images_batch.astype(np.float32))
                feature, sig_feat = self.model_backbone(inputs.cuda())
                output_u = self.model_uncertainty(sig_feat)
                if tt_flip:
                    # flip TTA, test-time augmentation
                    inputs2 = torch.flip(inputs, [3])
                    feature2, sig_feat2 = self.model_backbone(inputs2.cuda())
                    output_u2 = self.model_uncertainty(sig_feat2)
                    output_u = (output_u + output_u2) / 2
                    feature = feature + feature2
                feature = F.normalize(feature)
                sig = output_u
                feature = feature.cpu().detach().numpy()
                sig = sig.cpu().detach().numpy()
                mu[start_idx:end_idx] = feature
                sigma_sq[start_idx:end_idx] = sig
            # lprint(mu[0, :10])
            # print(sigma_sq[0, :10])
            # exit(0)
        if verbose:
            print('')
        return mu, sigma_sq


def main(args):
    network = NetworkMCDropout(args.backbone_name, args.resume_backbone)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", help="", type=str, default='')
    parser.add_argument("--resume_backbone", help="", type=str, default='')
    args = parser.parse_args()
    main(args)
