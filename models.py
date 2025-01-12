import torch.nn as nn
import segmentation_models_pytorch as smp
from monai.networks.nets.segresnet import SegResNet
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.segresnet_ds import SegResNetDS
from monai.networks.nets.dynunet import DynUNet
from cbamUnetpp import cbamUnetPP
from ECAUnetpp import ECAUnetPP
# from transformers import SegformerForSemanticSegmentation

from config import CFG

'''model'''


class STS2DModel(nn.Module):
    def __init__(self, cfg, model_name):
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        if self.model_name == "Unet":
            print("using {} for training.".format(self.model_name))
            self.model = smp.Unet(
                encoder_name=cfg.encoder_name,
                encoder_weights=cfg.encoder_weights,
                in_channels=cfg.in_chans,
                classes=cfg.target
            )
        elif self.model_name == "Unet++":
            print("using {} for training.".format(self.model_name))
            self.model = smp.UnetPlusPlus(
                encoder_name=cfg.encoder_name,
                encoder_weights=cfg.encoder_weights,
                in_channels=cfg.in_chans,
                classes=cfg.target
            )
        elif self.model_name == "SegResNet_8":
            print("using {} for training.".format(self.model_name))
            self.model = SegResNet(
                spatial_dims=2,
                init_filters=8,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
            )
        elif self.model_name == "SegResNet_32":
            print("using {} for training.".format(self.model_name))
            self.model = SegResNet(
                spatial_dims=2,
                init_filters=32,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
            )
        elif self.model_name == "SegResNet_64":
            print("using {} for training.".format(self.model_name))
            self.model = SegResNet(
                spatial_dims=2,
                init_filters=64,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
            )
        elif self.model_name == "SwinUNETR":
            print("using {} for training.".format(self.model_name))
            self.model = SwinUNETR(
                img_size=CFG.tile_size,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
                depths=(4, 4, 4, 4),
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                spatial_dims=2
            )
        elif self.model_name == "AttentionUnet":
            print("using {} for training.".format(self.model_name))
            self.model = AttentionUnet(
                spatial_dims=2,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
                channels=(64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                dropout=0.1,
            )
        elif self.model_name == "SegResNetDS":
            print("using {} for training.".format(self.model_name))
            self.model = SegResNetDS(
                spatial_dims=2,
                init_filters=64,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
                blocks_down=(2, 4, 4, 8),
                dsdepth=1,
            )
        elif self.model_name == "DynUNet":
            print("using {} for training.".format(self.model_name))
            self.model = DynUNet(
                spatial_dims=2,
                in_channels=CFG.in_chans,
                out_channels=CFG.target,
                kernel_size=[[3, 3]] * 5,
                strides=[[1, 1]] +  [[2, 2]] * 4,
                upsample_kernel_size=[[2, 2]] * 4,
                norm_name=("INSTANCE", {"affine": True}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01})
            )

        elif self.model_name == "cbamUnet++":
            print("using {} for training.".format(self.model_name))
            self.model = cbamUnetPP(in_channels=1,
                                  classes=1)

        elif self.model_name == "ecaUnet++":
            print("using {} for training.".format(self.model_name))
            self.model = ECAUnetPP(in_channels=1,
                                  classes=1)


    def forward(self, x, *args):
        if args is None:
            outputs = self.model(x)
        else:
            outputs = self.model(x, *args)
        return outputs

    def save_pretrained(self, save_directory):
        return self.model.save_pretrained(save_directory=save_directory)