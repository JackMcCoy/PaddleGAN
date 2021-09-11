import paddle
from paddle import nn
import paddle.nn.functional as F
from math import gcd,ceil
from collections import namedtuple
import numpy as np
from functools import partial, reduce

from .builder import GENERATORS
from . import ResnetBlock, ConvBlock, adaptive_instance_normalization


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ema_inplace(moving_avg, new, decay):
    return (moving_avg*decay)+(new * (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

class VectorQuantize(nn.Layer):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = 0.8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = paddle.randn((dim, n_embed))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', paddle.zeros(shape=(n_embed,1)))
        self.register_buffer('embed_avg', embed.clone())

    @property
    def codebook(self):
        return self.embed.transpose([1, 0])

    def forward(self, input):
        flatten = input.reshape((-1, self.dim))
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = (-dist).argmax(axis=1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed)
        embed_ind = paddle.reshape(embed_ind,shape=(input.shape[0],input.shape[1]))
        quantize = F.embedding(embed_ind, self.embed.transpose((1,0)))

        if self.training:
            self.cluster_size = ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose((1,0)) @ embed_onehot
            self.embed_avg = ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(axis=0)
            self.embed.data = embed_normalized

        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()
        return quantize, embed_ind, loss


@GENERATORS.register()
class DecoderQuantized(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(DecoderQuantized, self).__init__()

        self.quantize_4 = VectorQuantize(512, 1024)
        self.quantize_3 = VectorQuantize(1024, 1024)
        self.quantize_2 = VectorQuantize(1024, 1024)

        self.resblock_41 = ResnetBlock(512)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_31 = ConvBlock(256, 128)

        self.convblock_21 = ConvBlock(128, 128)
        self.convblock_22 = ConvBlock(128, 64)

        self.convblock_11 = ConvBlock(64, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))

    def forward(self, cF, sF):

        out = adaptive_instance_normalization(cF['r41'], sF['r41'])
        out = self.quantize_4(out)
        print(out.shape)
        print(out.flatten().shape)
        out = self.resblock_41(out)
        out = self.convblock_41(out)

        out = self.upsample(out)
        out += self.quantize_3(adaptive_instance_normalization(cF['r31'], sF['r31']))
        print(out.shape)
        print(out.flatten().shape)
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += self.quantize_2(adaptive_instance_normalization(cF['r21'], sF['r21']))
        print(out.shape)
        print(out.flatten().shape)
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out