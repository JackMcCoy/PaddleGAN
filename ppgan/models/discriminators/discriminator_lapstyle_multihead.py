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

from .builder import DISCRIMINATORS

@DISCRIMINATORS.register()
class LapStyleSingleDiscriminator(nn.Layer):
    def __init__(self, num_channels=32,kernel_size=3,padding=1,dropout_rate=.5,num_layer=3):
        super(LapStyleSingleDiscriminator, self).__init__()
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
