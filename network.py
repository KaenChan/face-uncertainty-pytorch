import argparse
import os,time
import sys
import torch
import logging
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import imp

from dataset.build_dataset import build_dataset
from dataset.imageprocessing import preprocess_imgs_np
from loss.build_loss import build_uncertainty_loss
from models.build_model import build_model_backbone, build_model_uncertainty
from evaluation.verify_wrapper import VerifyWrapper
from utils.utils import create_log_dir, save_model_last, load_model_last, fix_bn, trainlog, dt_str, display_watch_list


class Network(object):
    def __init__(self):
        self.config = None

    def initialize(self, config_file):
        config = imp.load_source('config', config_file)
        self.save_dir = create_log_dir(config, config_file)
        logfile = '%s/trainlog.log' % self.save_dir
        trainlog(logfile)
        logging.info('\nStart Training\nname: {}\nnum_epochs: {}\nepoch_size: {}\nbatch_size: {}'.format(
            config.name, config.num_epochs, config.epoch_size, config.batch_format['size']))
        logging.info('num_c_in_batch {} num_img_each_c {}'.format(
            config.batch_format['num_classes'], config.batch_format['size']/config.batch_format['num_classes']))
        self.config = config
        self.print_inter = config.print_inter
        self.build_model()
        self.build_dataloader()
        self.verify = VerifyWrapper(preprocess_imgs_np)
        self.optimizer = optim.SGD(self.model_uncertainty.parameters(),
                                   lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.exp_lr_scheduler = lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.learning_rate_milestones, gamma=0.1)
        print('uncertainty_loss_type', self.config.uncertainty_loss_type)
        u_criterion, is_confidence_prob = build_uncertainty_loss(self.config)
        self.criterion_uncertainty = u_criterion
        self.is_confidence_prob = is_confidence_prob
        self.embedding_size = config.embedding_size
        self.uncertainty_size = config.uncertainty_size

    def build_dataloader(self):
        self.use_pytorch_dataloader = False
        if self.use_pytorch_dataloader:
            self.trainset, self.train_loader = build_dataset(self.config, True)
            def cycle(iterable):
                while True:
                    for x in iterable:
                        yield x
            self.train_loader = iter(cycle(self.train_loader))
        else:
            self.trainset = build_dataset(self.config, False)

    def build_model(self):
        self.model_backbone = build_model_backbone(self.config.backbone_name, self.config.resume_backbone)
        self.model_uncertainty = build_model_uncertainty(self.config.uncertainty_name,
                                                         chs=self.config.uncertainty_module_chs,
                                                         resume=self.config.resume_uncertainty)
        if self.config.num_gpus > 1:
            # TODO: multi-GPU setting
            self.model_backbone = torch.nn.DataParallel(self.model_backbone, device_ids=self.config.GPU_ID)
            self.model_backbone = self.model_backbone.to(self.config.DEVICE)
            self.model_uncertainty = torch.nn.DataParallel(self.model_uncertainty, device_ids=self.config.GPU_ID)
            self.model_uncertainty = self.model_uncertainty.to(self.config.DEVICE)
        else:
            # single-GPU setting
            self.model_backbone = self.model_backbone.cuda()
            self.model_uncertainty = self.model_uncertainty.cuda()
        self.model_backbone.eval()
        self.model_backbone.apply(fix_bn)
        self.model_uncertainty.eval()

    def save_model(self, model_dir, global_step):
        # save_model_last(self.model_backbone, model_dir, global_step,
        #                 "backbone_{}".format(self.config.backbone_name), keep=1)
        save_model_last(self.model_uncertainty, model_dir, global_step,
                        "uncertainty_{}".format(self.config.uncertainty_name), keep=1)

    def load_model(self, model_dir):
        config_file = os.path.join(model_dir, 'config.py')
        config = imp.load_source('config', config_file)
        self.config = config
        # self.config.resume_backbone = ''
        self.config.resume_uncertainty = ''
        self.build_model()
        model_dir = os.path.expanduser(model_dir)
        # load_model_last(self.model_backbone, model_dir, 'backbone')
        load_model_last(self.model_uncertainty, model_dir, 'uncertainty')

    def train(self):
        for epoch in range(self.config.num_epochs + 1):
            # train phase
            if epoch > 0:
                # val phase
                self.model_backbone.train(False)  # Set model to evaluate mode
                self.model_uncertainty.train(False)  # Set model to evaluate mode
                extract_feature_func = lambda x: self.extract_feature(x, batch_size=32, verbose=True)
                sigma_sizes = self.config.uncertainty_size
                s_info = self.verify.run(extract_feature_func, sigma_sizes=sigma_sizes)
                self.save_model(self.save_dir, epoch)
                logging.info(s_info)
            if epoch == self.config.num_epochs:
                break

            # logging.info('current lr:%s' % self.exp_lr_scheduler.get_lr())
            for step in range(self.config.epoch_size):
                self.exp_lr_scheduler.step(epoch*self.config.epoch_size + step)
                step += 1
                self.model_backbone.train(False)
                self.model_uncertainty.train(True)
                if self.use_pytorch_dataloader:
                    inputs, labels = next(self.train_loader)
                else:
                    batch = self.trainset.pop_batch_queue()
                    inputs, labels = batch['image'], batch['label']
                    inputs = torch.from_numpy(np.array(inputs)).float()
                    labels = torch.from_numpy(np.array(labels)).long()

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                self.optimizer.zero_grad()

                watch_list = {}
                loss = 0
                feature, sig_feat = self.model_backbone(inputs)
                feature = F.normalize(feature)
                output_u = self.model_uncertainty(sig_feat)
                uloss, end_points = self.criterion_uncertainty(feature, output_u, labels)
                watch_list['uloss'] = uloss
                loss += uloss

                for k, v in end_points.items():
                    watch_list[k] = v

                loss.backward()
                self.optimizer.step()

                # batch loss
                if step % self.print_inter == 0:
                    s_info = '%s [%d-%d] | loss %.3f' % (dt_str(), epoch, step, loss.item())
                    s_info += ' lr%s' % self.exp_lr_scheduler.get_lr()[0]
                    s_info += display_watch_list(watch_list)
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    s_info += ' mem %.1f G' % memory
                    logging.info(s_info)

    def extract_feature(self, images, batch_size, need_preprecess=False, tt_flip=False, verbose=False):
        # forward
        # inputs = Variable(inputs.cuda())
        # feature, sig_feat = self.model_backbone(inputs)
        # log_sig_sq = self.model_uncertainty(sig_feat)
        num_images = len(images)
        mu = np.ndarray((num_images, self.config.embedding_size), dtype=np.float32)
        sigma_sq = np.ndarray((num_images, self.config.uncertainty_size), dtype=np.float32)
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
    network = Network()
    network.initialize(args.config_file)
    network.train()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="The path to the training configuration file",
                        type=str, default='config/config_ir50_ms.py')
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\face-recognition\lfw\lfw-112-mxnet')
    args = parser.parse_args()
    main(args)
