# Face-uncertainty-pytorch

This is a demo code of face uncertainty quantification or estimation using PyTorch.
The uncertainty of face recognition is affected by the ability of the recognition model (model uncertainty)
and the quality of the input image (data uncertainty).

Model Uncertainty:
+ MC-Dropout

Data Uncertainty:
+ PCNet loss, https://arxiv.org/abs/2009.00603
+ IDQ loss, https://github.com/KaenChan/lightqnet
+ FastMLS loss, https://arxiv.org/abs/2102.04075
+ MLS loss, https://github.com/seasonSH/Probabilistic-Face-Embeddings


##  Usage

### Preprocessing

Download the MS-Celeb-1M dataset from 1 or 2:
1. insightface, https://github.com/deepinsight/insightface/wiki/Dataset-Zoo
2. face.evoLVe.PyTorch, https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo) 

Decode it using the code:
https://github.com/deepinsight/insightface/blob/master/recognition/common/rec2image.py

### Training
1. Download the base model from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

2. Modify the configuration files under ```config/``` folder.

4. Start the training:

    ``` Shell
    python network.py --config_file config/config_ir50_idq_loss_glint360k.py
    ```
   
    ``` 
    Start Training
    name: glint_ir50_idq
    num_epochs: 12
    epoch_size: 1000
    batch_size: 80
    num_c_in_batch 10 num_img_each_c 8.0
    IDQ_loss soft 16 0.45
    2022-01-12 23:37:48 [0-100] | loss 0.535 lr0.01 cos 0.55 1.00 0.18 pconf 0.77 1.00 0.15 t_soft 0.69 1.00 0.01 uloss 0.535 mem 3.1 G
    2022-01-12 23:38:12 [0-200] | loss 0.464 lr0.01 cos 0.58 0.93 0.08 pconf 0.75 1.00 0.05 t_soft 0.76 1.00 0.00 uloss 0.464 mem 3.1 G
    2022-01-12 23:38:37 [0-300] | loss 0.533 lr0.01 cos 0.52 1.00 0.04 pconf 0.78 0.99 0.25 t_soft 0.63 1.00 0.00 uloss 0.533 mem 3.1 G
    2022-01-12 23:39:02 [0-400] | loss 0.511 lr0.01 cos 0.52 0.99 0.09 pconf 0.77 0.99 0.16 t_soft 0.61 1.00 0.00 uloss 0.511 mem 3.1 G
    2022-01-12 23:39:27 [0-500] | loss 0.554 lr0.01 cos 0.48 0.97 0.05 pconf 0.77 0.99 0.18 t_soft 0.56 1.00 0.00 uloss 0.554 mem 3.1 G
    2022-01-12 23:39:52 [0-600] | loss 0.462 lr0.01 cos 0.55 0.95 0.19 pconf 0.78 0.99 0.23 t_soft 0.70 1.00 0.01 uloss 0.462 mem 3.1 G
    2022-01-12 23:40:17 [0-700] | loss 0.408 lr0.01 cos 0.55 0.96 0.07 pconf 0.78 0.99 0.07 t_soft 0.70 1.00 0.00 uloss 0.408 mem 3.1 G
    2022-01-12 23:40:42 [0-800] | loss 0.532 lr0.01 cos 0.51 0.99 0.03 pconf 0.80 0.99 0.25 t_soft 0.63 1.00 0.00 uloss 0.532 mem 3.1 G
    2022-01-12 23:41:06 [0-900] | loss 0.563 lr0.01 cos 0.54 1.00 0.03 pconf 0.80 0.99 0.13 t_soft 0.66 1.00 0.00 uloss 0.563 mem 3.1 G
    2022-01-12 23:41:27 [0-1000] | loss 0.570 lr0.01 cos 0.50 0.86 0.11 pconf 0.78 0.99 0.16 t_soft 0.61 1.00 0.00 uloss 0.570 mem 3.1 G
    ---cfp_fp
    sigma_sq [0.00263163 0.01750576 0.04416942 0.10698225 0.23958328 0.46090251
     0.92462665] percentile [0, 10, 30, 50, 70, 90, 100]
    reject_factor 0.0000 risk_threshold 0.924627 keep_idxes 7000 / 7000 Cosine score eer 0.012571 fmr100 0.012571 fmr1000 0.018286
    reject_factor 0.0500 risk_threshold 0.650710 keep_idxes 6655 / 7000 Cosine score eer 0.004357 fmr100 0.003900 fmr1000 0.006601
    reject_factor 0.1000 risk_threshold 0.556291 keep_idxes 6300 / 7000 Cosine score eer 0.003968 fmr100 0.003791 fmr1000 0.006003
    reject_factor 0.1500 risk_threshold 0.509630 keep_idxes 5951 / 7000 Cosine score eer 0.003864 fmr100 0.004013 fmr1000 0.005351
    reject_factor 0.2000 risk_threshold 0.459032 keep_idxes 5600 / 7000 Cosine score eer 0.003392 fmr100 0.003540 fmr1000 0.004248
    reject_factor 0.2500 risk_threshold 0.421400 keep_idxes 5251 / 7000 Cosine score eer 0.003236 fmr100 0.003407 fmr1000 0.003785
    reject_factor 0.3000 risk_threshold 0.389943 keep_idxes 4903 / 7000 Cosine score eer 0.002651 fmr100 0.002436 fmr1000 0.002842
    reject_factor mean --------------------------------------------- Cosine score fmr1000 0.002684
    AUERC: 0.0026
    AUERC30: 0.0017
    AUC: 0.0024
    AUC30: 0.0015
    ```
   
### Testing

We use lfw.bin, cfp_fp.bin, etc. from ms1m-retinaface-t1 as the test dataset.
    
``` Shell
python evaluation/verification_risk_fnmr.py
```
  

### MC-Dropout

``` Shell
python mc_dropout/verification_risk_mcdropout_fnmr.py
```

