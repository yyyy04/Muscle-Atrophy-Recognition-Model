import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import random
import os
from pathlib import Path


class CFG:
    test = False    #true为测试   false为训练
    using_ensemble_models = False
    thr = 0.55
    show = 255  # 255是显示，1是用来提交或者制作新label使用
    img_suffix = ".jpg"
    msk_suffix = ".png"

    seed = 42
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # =========model-cfg
    #   "Unet" "Unet++" "SegResNet_8"  "SegResNet_32" "SegResNet_64" "SwinUNETR" "AttentionUnet" "SegResNetDS" "DynUNet"
    #   "Segformer" "cbamUnet++" "ecaUnet++"
    model_name = "ecaUnet++"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    # =========ensemble model-cfg
    models_checkpoints = {
        "DynUNet": [
            "results/log/exp18_DynUNet/checkpoint/DynUNet-best_score_0.974943-max_score_th0.6-train_loss_0.071235-valid_loss0.029596-epoch_30.pth",],
        # "results/log/exp19_DynUNet/checkpoint/DynUNet-best_score_0.642290-max_score_th0.65-train_loss_0.006324-valid_loss0.002095-epoch_12.pth"],
        "SegResNet_64": [
            "results/log/exp14_SegResNet_64/checkpoint/SegResNet_64-best_score_0.974266-max_score_th0.6-train_loss_0.073523-valid_loss0.033847-epoch_29.pth",]
        # "results/log/exp20_SegResNet_64/checkpoint/SegResNet_64-best_score_0.642303-max_score_th0.65-train_loss_0.005462-valid_loss0.002036-epoch_26.pth"]
    }
    # =========pred target-cfg

    target = 1  #通道数

    exp = "exp1_{}".format(model_name)
    log_dir = "./results/log/{}".format(exp)
    log_samples_dir = log_dir + "/samples/"
    log_checkpoint_dir = log_dir + "/checkpoint/"
    if not Path(log_samples_dir).exists() and not test:
        os.makedirs(log_samples_dir, exist_ok=True)
    if not Path(log_checkpoint_dir).exists() and not test:
        os.makedirs(log_checkpoint_dir, exist_ok=True)
    # =========data-cfg
    root_dirs = "./data"
    data_dirs = root_dirs + "/train"

    mask_loation = False

    size = 640
    in_chans = 1

    tile_size = 640
    train_stride = tile_size // 5      #10
    valid_stride = tile_size // 5       #5

    rate_valid = 0.01
    rate_test = 0.01

    train_bs = 10
    valid_bs = 1

    # =========train-cfg
    epochs = 30
    save_log_turn = 5
    lr = 3e-4
    min_lr = 1e-7
    weight_decay = 1e-5
    pretrained = False
    # checkpoint = "results/log/exp2_cbamUnet++/checkpoint/cbamUnet++-score_0.665516-thr_0.6-dice_0.750946-iou_0.637380-hdf_0.628388-train_loss_0.007227-valid_loss_0.010186-epoch_11.pth"
    checkpoint = "results/log/exp1_ecaUnet++/checkpoint/ecaUnet++-score_0.736954-thr_0.55-dice_0.762655-iou_0.649737-hdf_0.820800-train_loss_0.016193-valid_loss_0.044256-epoch_23.pth"

    if pretrained:
        start_epoch = int(checkpoint[(checkpoint.find("epoch_") + len("epoch_")):].split(".")[0])
    else:
        start_epoch = 0

    # =========test-cfg
    valid_txt = "results/valid.txt"
    test_txt = "results/compare.txt"
    test_data_dirs = root_dirs + "/test"
    test_preds_dir = "results/log/exp1_Unet++/TTA3/"
    if not Path(test_preds_dir).exists():
        os.makedirs(test_preds_dir, exist_ok=True)

    test_tile_size = 640
    test_stride = tile_size // 5
    test_bs = 7
    tta = True

    # =========fixed
    num_workers = 4
    use_amp = True
    max_grad_norm = 1.0

    # =========augmentation

    train_aug_list = [
        A.Resize(size, size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.RandomGamma(gamma_limit=(30, 150)),
        ], p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
                        mask_fill_value=0, p=0.5),

        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    test_aug_list = [
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    base_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


seed = CFG.seed
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
