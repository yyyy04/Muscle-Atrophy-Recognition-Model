from typing import Optional, Union, List
from config import CFG
import torch
from torch import nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
import torch.nn.functional as F
from tricks.ECANet.models.eca_module import eca_layer as ECA

class ECAUnetPP(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("UnetPlusPlus is not support encoder_name={}".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # 注意力机制（SA-MOS）
        self.eca_attention = nn.ModuleList([])
        for out_channel in [1, 64, 64, 128, 256, 512]:
            self.eca_attention.append(ECA(out_channel))

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "ECA-unetplusplus-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        # 分别进行通道注意力和时间注意力，得到融合的特征表示
        feat_maps_eca = []
        new_features = []
        for f, eca_attention in zip(features, self.eca_attention):
            feat_maps_eca.append((eca_attention(f)))
        for idx in range(len(features)):
            new_features.append(feat_maps_eca[idx] + features[idx])
        running_means = [torch.zeros(channel).to(CFG.device) for channel in [1, 64, 64, 128, 256, 512]]
        running_vars = [torch.ones(channel).to(CFG.device) for channel in [1, 64, 64, 128, 256, 512]]
        features = [F.batch_norm(feature, running_mean=running_means[i], running_var=running_vars[i]) for i, feature in enumerate(new_features)]
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks