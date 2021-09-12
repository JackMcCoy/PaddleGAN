import paddle
from paddle import nn
import paddle.nn.functional as F
from math import gcd,ceil
from collections import namedtuple
import numpy as np
from functools import partial, reduce
from einops.layers.paddle import Rearrange

from .builder import GENERATORS
from . import ResnetBlock, ConvBlock, adaptive_instance_normalization, Transformer


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ema_inplace(moving_avg, new, decay):
    moving_avg.add_((decay * moving_avg)-moving_avg).add_(new * (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

class ImageLinearAttention(nn.Layer):
    def __init__(self, chan, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = True):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2D(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2D(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2D(chan, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2D(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, context = None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q, k, v = map(lambda t: t.reshape((b, heads, -1, h * w)), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        if context is not None:
            context = context.reshape((b, c, 1, -1))
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape((b, heads, k_dim, -1)), (ck, cv))
            k = paddle.concat((k, ck), axis=3)
            v = paddle.concat((v, cv), axis=3)

        k = k.softmax(axis=-1)

        if self.norm_queries:
            q = q.softmax(axis=-2)

        context = paddle.matmul(k, v.transpose([0,1,3,2]))
        print('q size - '+str(q.shape))
        print('context size - '+str(context.shape))
        out = paddle.matmul(q, context)
        #out = torch.einsum('bhdn,bhde->bhen', q, context)
        out = out.reshape((b, -1, h, w))
        out = self.to_out(out)
        return out

class VectorQuantize(nn.Layer):
    def __init__(
        self,
        dim,
        codebook_size,
        transformer_size,
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
        self.register_buffer('cluster_size', paddle.zeros(shape=(n_embed,)))
        self.register_buffer('embed_avg', embed.clone())
        if codebook_size != 1280:
            self.rearrange = Rearrange('b c h w -> b (h w) c')
            self.decompose_axis = Rearrange('b (h w) c -> b c h w',h=dim)

        else:
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=4,p2=4)
            self.decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=16,w=16, e=4,d=4)

        if transformer_size==1:
            self.transformer = ImageLinearAttention(512, kernel_size = 1, padding = 0, stride = 1, key_dim = 16, value_dim = 16, heads = 8, norm_queries = False)
            #self.transformer = Transformer(dim**2*2, 6, 8, dim**2*2, dim**2*2, dropout=0.1)
            self.pos_embedding = paddle.create_parameter(shape=(1, 256, 512), dtype='float32')
        elif transformer_size==2:
            self.transformer = ImageLinearAttention(256, kernel_size = 1, padding = 0, stride = 1, key_dim = 32, value_dim = 32, heads = 8, norm_queries = False)
            #self.transformer = Transformer(256, 4, 8, 256, 256, dropout=0.1)
            self.pos_embedding = paddle.create_parameter(shape=(1, 1024, 256), dtype='float32')
        elif transformer_size==3:
            self.transformer = ImageLinearAttention(128, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = False)
            #self.transformer = Transformer(2048, 2, 8, 1024, 2048, dropout=0.1)
            self.pos_embedding = paddle.create_parameter(shape=(1, 256, 2048), dtype='float32')
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
        embed_ind = paddle.reshape(embed_ind,shape=(input.shape[0],input.shape[1],input.shape[2]))
        quantize = F.embedding(embed_ind, self.embed.transpose((1,0)))

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = paddle.matmul(flatten.transpose((1,0)), embed_onehot)
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(axis=0)
            self.embed = embed_normalized


        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = self.transformer(quantize)
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

        self.quantize_4 = VectorQuantize(16, 320, 1)
        self.quantize_3 = VectorQuantize(32, 320, 2)
        self.quantize_2 = VectorQuantize(64, 1280, 3)

        self.resblock_41 = ResnetBlock(512)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_31 = ConvBlock(256, 128)


        self.convblock_21 = ConvBlock(128, 128)
        self.convblock_22 = ConvBlock(128, 64)
        self.convblock_11 = ConvBlock(64, 64)

        self.downsample = nn.Upsample(scale_factor=.5, mode='nearest')
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))


    def forward(self, cF, sF):
        out = adaptive_instance_normalization(cF['r41'], sF['r41'])
        quantize, embed_ind, code_losses = self.quantize_4(out)
        out = self.resblock_41(quantize)
        out = self.convblock_41(out)

        upscale_4 = self.upsample(out)
        # Transformer goes here?
        quantize, embed_ind, loss = self.quantize_3(adaptive_instance_normalization(cF['r31'], sF['r31']))
        out = upscale_4+quantize
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        code_losses+=loss

        out = self.upsample(out)
        quantize, embed_ind, loss = self.quantize_2(adaptive_instance_normalization(cF['r21'], sF['r21']))
        code_losses+=loss
        out += quantize

        out = self.convblock_21(out)
        out = self.convblock_22(out)
        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out, code_losses