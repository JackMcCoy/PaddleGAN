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
from paddle.nn.layer import Linear
from paddle.nn.utils import spectral_norm

from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class LapStyleDiscriminator(nn.Layer):
    def __init__(self, num_channels=32,kernel_size=3,padding=1):
        super(LapStyleDiscriminator, self).__init__()
        num_layer = 3
        num_channel = num_channels
        self.head = nn.Sequential(
            ('conv',
             nn.Conv2D(3, num_channel, kernel_size=kernel_size, stride=1, padding=padding)),
            ('norm', nn.BatchNorm2D(num_channel)),
            ('LeakyRelu', nn.LeakyReLU(0.2)))
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

class OptimizedBlock(nn.Layer):
    """Residual block.

    It has a style of:
        ---Pad-Conv-ReLU-Pad-Conv-+-
         |________________________|

    Args:
        dim (int): Channel number of intermediate features.
    """
    def __init__(self, in_channels, dim):
        super(OptimizedBlock, self).__init__()
        out_size=(1,dim,256,256)
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        spectral_norm(nn.Conv2D(in_channels, dim, (3, 3))),
                                        nn.ReLU(),
                                        nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        spectral_norm(nn.Conv2D(dim, dim, (3, 3))),
                                        nn.AvgPool2D(kernel_size=2,stride=2))
        self.residual_connection = nn.Sequential(spectral_norm(nn.Conv2D(in_channels, dim, (1,1))),
                                        nn.AvgPool2D(kernel_size=2,stride=2))

    def forward(self, x):
        out = self.residual_connection(x) + self.conv_block(x)
        return out

class ResBlock(nn.Layer):
    """Residual block.

    It has a style of:
        ---Pad-Conv-ReLU-Pad-Conv-+-
         |________________________|

    Args:
        dim (int): Channel number of intermediate features.
    """
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        out_size=(1,dim,128,128)
        self.conv_block = nn.Sequential(nn.ReLU(),
                                        nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        spectral_norm(nn.Conv2D(dim, dim, (3, 3))),
                                        nn.ReLU(),
                                        nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        spectral_norm(nn.Conv2D(dim, dim*2, (3, 3))),
                                        )
        self.residual_connection = nn.Sequential(spectral_norm(nn.Conv2D(dim, dim*2, (1,1))))
    def forward(self, x):
        out = self.residual_connection(x) + self.conv_block(x)
        return out

def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x

def _l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = paddle.normal(mean=0,std=1,shape=(1, W.shape(0)))
    _u = u
    print(W.shape)
    for _ in range(Ip):
        _v = _l2normalize(paddle.matmul(_u, W), eps=1e-12)
        _u = _l2normalize(paddle.matmul(_v, paddle.transpose(W, [1, 0])), eps=1e-12)
    sigma = paddle.sum(nn.functional.linear(_u, paddle.transpose(W, [1, 0])) * _v)
    return sigma, _u

class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`
           W(Tensor): Spectrally normalized weight
           u (Tensor): the right largest singular value of W.
       """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', paddle.normal(mean=0,std=1,shape=(1, 1)))

    @property
    def W_(self):
        w_mat = paddle.reshape(self.weight,(self.weight.shape[0], -1))
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return paddle.nn.functional.linear(input, self.W_, self.bias)

@DISCRIMINATORS.register()
class LapStyleSpectralDiscriminator(nn.Layer):
    def __init__(self, num_channels=32,num_layer=3):
        super(LapStyleSpectralDiscriminator, self).__init__()
        self.num_layer=num_layer
        self.num_channels = num_channels
        self.head = OptimizedBlock(3,num_channels)
        self.body = nn.Sequential()
        for i in range(num_layer - 1):
            self.body.add_sublayer(
                'conv%d' % (i + 1),
                ResBlock(num_channels*max(1,2**i)))
        self.fc = nn.ReLU()

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.fc(x)
        x = nn.functional.avg_pool2d(x, 256, stride=1)
        x = paddle.reshape(x,(1,1,256,256))
        return x