#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops.layers.paddle import Rearrange

from ..generators import Transformer
from .builder import DISCRIMINATORS

class CalcContentLoss():
    """Calc Content Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target, norm=False):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            norm(Bool): whether use mean_variance_norm for pred and target
        """
        if (norm == False):
            return self.mse_loss(pred, target)
        else:
            return self.mse_loss(mean_variance_norm(pred),
                                 mean_variance_norm(target))

def calc_mean_std(feat, eps=1e-5):
    """calculate mean and standard deviation.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
        eps (float): Default: 1e-5.

    Return:
        mean and std of feat
        shape: [N, C, 1, 1]
    """
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape([N, C, -1])
    feat_var = paddle.var(feat_var, axis=2) + eps
    feat_std = paddle.sqrt(feat_var)
    feat_std = feat_std.reshape([N, C, 1, 1])
    feat_mean = feat.reshape([N, C, -1])
    feat_mean = paddle.mean(feat_mean, axis=2)
    feat_mean = feat_mean.reshape([N, C, 1, 1])
    return feat_mean, feat_std

def mean_variance_norm(feat):
    """mean_variance_norm.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized feat with shape (N, C, H, W)
    """
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class NoiseBlock(nn.Layer):
    def __init__(self, channels):
        super().__init__()
    def forward(self,x):
        noise = paddle.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
        x = x + noise
        return x

@DISCRIMINATORS.register()
class LapStyleSingleDiscriminator(nn.Layer):
    def __init__(self, num_channels=32,kernel_size=3,padding=1,noise=0,num_layer=3):
        super(LapStyleSingleDiscriminator, self).__init__()
        num_channel = num_channels
        self.head = nn.Sequential(
            ('conv',
             nn.Conv2D(3, num_channel, kernel_size=kernel_size, stride=1, padding=padding)),
            ('norm', nn.BatchNorm2D(num_channel)),
            ('LeakyRelu', nn.LeakyReLU(0.2)),
            ('Quantization'), VectorDiscQuantize(128, 1280, 3))
        if noise==1:
            self.head.add_sublayer('noise',NoiseBlock(num_channel))
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            self.body.add_sublayer(
                'conv%d' % (i + 1),
                nn.Conv2D(num_channel,
                          num_channel,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=padding))
            self.body.add_sublayer('norm%d' % (i + 1),
                                   nn.BatchNorm2D(num_channel))
            self.body.add_sublayer('LeakyRelu%d' % (i + 1), nn.LeakyReLU(0.2))
            if noise == 1:
                self.body.add_sublayer('noise', NoiseBlock(num_channel))
        self.tail = nn.Conv2D(num_channel,
                              1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

@DISCRIMINATORS.register()
class LapStyleMultiresDiscriminator(nn.Layer):
    def __init__(self, num_channels=32,num_halvings=2):
        super(LapStyleMultiresDiscriminator, self).__init__()
        num_layer = 3
        self.resolutions=[]
        self.output_resolutions=[]
        for i in range(num_halvings+1):
            if i>0:
                net=LapStyleSingleDiscriminator(num_channels=int(num_channels/(2*i)),num_layer=i+1*3)
                print(type(net))
            else:
                net=LapStyleSingleDiscriminator(num_channels=num_channels)
                print(type(net))
            self.resolutions.append(net)

    def forward(self, x):
        out=self.resolutions[0](x)+self.resolutions[1](x)+self.resolutions[2](x)
        return out

class VectorDiscQuantize(nn.Layer):
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
        self.perceptual_loss = CalcContentLoss()

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
            self.transformer = Transformer(dim**2*2, 8, 16, 64, dim**2*2, dropout=0.1)
            self.pos_embedding = nn.Embedding(256, 512)
        elif transformer_size==2:
            self.transformer = Transformer(256, 8, 16, 64, 256, dropout=0.1)
            self.pos_embedding = nn.Embedding(1024, 256)
        elif transformer_size==3:
            self.transformer = Transformer(2048, 8, 16, 64, 768, dropout=0.1)
            self.pos_embedding = nn.Embedding(256, 2048)
    @property
    def codebook(self):
        return self.embed.transpose([1, 0])

    def forward(self, input):
        quantize = self.rearrange(input)
        b, n, _ = quantize.shape

        ones = paddle.ones((b, n), dtype="int64")
        seq_length = paddle.cumsum(ones, axis=1)
        position_ids = seq_length - ones
        position_ids.stop_gradient = True
        position_embeddings = self.pos_embedding(position_ids)

        quantize = self.transformer(quantize + position_embeddings)
        quantize = self.decompose_axis(quantize)

        quantize = input + (quantize - input).detach()

        flatten = quantize.reshape((-1, self.dim))
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

        loss = self.perceptual_loss(quantize.detach(), input, norm=True) * self.commitment

        return quantize, embed_ind, loss
