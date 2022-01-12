import logging

from easydict import EasyDict
from torch.utils.data import BatchSampler
from torchvision import datasets, transforms
import torch
import time

from dataset.imageprocessing import preprocess_imgs_np
from dataset.sampler import BalancedSampler


def build_dataset(config, use_pytorch_dataloader=False):
    if use_pytorch_dataloader:
        trans = transforms
        # TODO: pytorch_dataloader
        train_transform = trans.Compose([
            # trans.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        trainset = datasets.ImageFolder(config.train_dataset_path, train_transform)
        logging.info("| Dataset Info |datasize: {}|num_labels: {}|".format(len(trainset), len(trainset.classes)))

        if config.batch_IPC > 0:
            balanced_sampler = BalancedSampler(trainset, batch_size=config.batch_size, images_per_class=config.batch_IPC)
            batch_sampler = BatchSampler(balanced_sampler, batch_size=config.batch_size, drop_last=True)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                num_workers=4,
                pin_memory=True,
                batch_sampler=batch_sampler
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            )
        return trainset, train_loader
    else:
        from dataset.dataset_np_ipc import Dataset
        t1 = time.time()
        trainset = Dataset(config.train_dataset_path)
        trainset.set_base_seed(config.base_random_seed)
        print('time', time.time() - t1)
        proc_func = preprocess_imgs_np
        trainset.start_batch_queue(config.batch_format, proc_func=proc_func)
        return trainset


if __name__ == '__main__':
    from config import config_ir50_idq_pcloss_glint360k_r50 as configfig
    trainset = build_dataset(configfig, False)
    batch = trainset.pop_batch_queue()
