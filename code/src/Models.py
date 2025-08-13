import torch
from torch import nn
import segmentation_models_pytorch as smp

class SegModel(nn.Module):
    def __init__(self, model_type, num_classes, encoder, pre_weight):
        super().__init__()

        if model_type.lower() == 'unet':
            self.model = smp.Unet(
                classes=num_classes,
                encoder_name=encoder,
                encoder_weights=pre_weight,
                in_channels=3
            )
        elif model_type.lower() == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                classes=num_classes,
                encoder_name=encoder,
                encoder_weights=pre_weight,
                in_channels=3
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        return self.model(x)
