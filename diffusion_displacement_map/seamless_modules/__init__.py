import torch
import torch.nn.functional as F
from torch import nn
from creativeai.image.encoders.base import Encoder, ModuleConfig as M
from creativeai.image.encoders.models import ConvBlock as NormalConvBlock

import contextvars

padding_context_circular = contextvars.ContextVar("padding_mode")


class CircularPad(nn.Module):
    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, "circular")


class SeamlessConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super().__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding_mode="circular"
        )
        self.activation = getattr(nn, activation)(inplace=True)

    def forward(self, x, **kwargs):
        if padding_context_circular.get() == "null":
            raise RuntimeError("Should not be reachable")
        # print()
        # print("  > ", padding_context_circular.get())
        x = F.pad(x, self.padding, padding_context_circular.get())
        x = self.conv(x)
        return self.activation(x)


def create_pool(kernel_size, padding=[1, 1, 1, 1]):
    # padding = [*kernel_size, *kernel_size]
    # add wrap-around padding on images

    return nn.Sequential(
        torch.nn.AvgPool2d(kernel_size)
        # CircularPad(padding), torch.nn.AvgPool2d(kernel_size)
        # CircularPad(padding), torch.nn.AvgPool2d(kernel_size), UndoPadding(padding)
    )


class WraparoundVGG11(Encoder):
    CONFIG = [M(f=64, b=1), M(f=128, b=1), M(f=256, b=2), M(f=512, b=2), M(f=512, b=2)]
    FILENAME = "v0.1/vgg11"
    HEXDIGEST = "2898532ef3f0910dfb1634482a1ca4ef"

    def __init__(
        self,
        block_type=SeamlessConvBlock,
        pool_type=create_pool,
        input_type="RGB",
        **kwargs
    ):
        super().__init__(block_type, pool_type, input_type=input_type, **kwargs)


# encoder: Encoder = MyVGG11(
#     pretrained=True, pool_type=torch.nn.AvgPool2d,
# )
# # reisn


class UndoPadding(nn.Module):
    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, inputs):
        return inputs[
            ..., self.padding[0] : -self.padding[1], self.padding[2] : -self.padding[3]
        ]
