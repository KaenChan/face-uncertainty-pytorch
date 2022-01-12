''' Config Proto '''

import sys
import os

# The name of the current model for output
name = 'glint_ir50_pcloss'

# The folder to save log and model
log_base_dir = './log/'

# Training dataset path
train_dataset_path = r"F:\data\metric-learning\face\ms1m-retinaface-t1-img"

# Target image size for the input of network
image_size = [112, 112]

# Number of GPUs
num_gpus = 1

####### NETWORK #######

# The network architecture
backbone_name = 'backbones.iresnet.iresnet50'
# Number of dimensions in the embedding space
embedding_size = 512

uncertainty_name = 'head_fc2_prob'
# uncertainty_module_output_size = embedding_size
uncertainty_module_chs = [512*7*7, 256, 1]
uncertainty_size = uncertainty_module_chs[-1]


####### TRAINING STRATEGY #######

# Base Random Seed
base_random_seed = 0

# Number of samples per batch
batch_size = 80
batch_IPC = 8
batch_format = {
    'size': batch_size,
    'num_classes': batch_size // 8,
}

# Number of batches per epoch
epoch_size = 1000
print_inter = 100

# Number of epochs
# num_epochs = 32
num_epochs = 2

# learning rate strategy
# learning_rate_strategy = 'step'

# learning rate schedule
# lr = 3e-3
learning_rate = 3e-2
learning_rate_milestones = [
    num_epochs / 4 * 2 * epoch_size,
    num_epochs / 4 * 3 * epoch_size
]

# Restore model
resume_backbone = r'E:\chenkai\arcface_torch\glint360k_cosface_r50_fp16_0.1\backbone.pth'
# resume_backbone = r'G:\data\model_pytorch\pretrain\ms1m-ir50\backbone_ir50_ms1m_epoch120.pth'
resume_uncertainty = r''


# Weight decay for model variables
weight_decay = 5e-4

uncertainty_loss_type = 'pc_loss'
