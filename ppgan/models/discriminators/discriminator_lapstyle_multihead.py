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
from .discriminator_lapstyle import LapStyleDiscriminator
import paddle.nn.functional as F

from .builder import DISCRIMINATORS

@DISCRIMINATORS.register()
class LapStyleMultiResDiscriminator(nn.Layer):
    def __init__(self, num_channels=32):
        super(LapStyleDiscriminator, self).__init__()
        num_layer = 3
        self.resolutions=[]
        halving = [num_channels,num_channels/2,numchannels/4]
        num_channel = num_channels
        for i in halving:
            if i>0:
                net_w_applicable_downsample=nn.Sequential(
                    F.interpolate(scale_factor=1/(i+1))
                    LapstyleDiscriminator(num_channels=num_channels)
                )
            else:
                net_w_applicable_downsample=LapstyleDiscriminator(num_channels=num_channels)
            self.resolutions.append(LapstyleDiscriminator(num_channels=num_channels))
        self.pooling = nn.AvgPool1d(3,stride=1,padding=1,)

    def forward(self, x):
        resolutions = []
        for i in self.resolutions:
            resolutions.append(i.forward(x))
        x = self.pooling(x)
        return x
