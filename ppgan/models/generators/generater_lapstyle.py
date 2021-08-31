#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn
from ...utils.download import get_path_from_url
from PIL import Image
from skimage import color
import numpy as np
from .builder import GENERATORS

if 0:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as linalg

    linalg.init()


def cal_maxpool_size(w, h, count=3):
    if count == 3:
        w = np.ceil(np.ceil(np.ceil(w / 2) / 2) / 2)
        h = np.ceil(np.ceil(np.ceil(h / 2) / 2) / 2)
    elif count == 2:
        w = np.ceil(np.ceil(w / 2) / 2)
        h = np.ceil(np.ceil(h / 2) / 2)
    elif count == 1:
        w = np.ceil(w / 2)
        h = np.ceil(h / 2)
    else:
        raise ValueError
    return int(w), int(h)

class KMeansGPU:
    def __init__(self, n_clusters, device='cuda', tol=1e-4, init='kmeans++'):
        self.n_clusters = n_clusters
        self.device = device
        self.tol = tol
        self.init = init
        self._labels = None
        self._cluster_centers = None
        self.init = init

    def _initial_state(self, data):
        # initial cluster centers by kmeans++ or random
        if self.init == 'kmeans++':
            print(data.shape)
            n, c = data.shape
            dis = paddle.zeros((n, self.n_clusters))
            initial_state = paddle.zeros((self.n_clusters, c))
            pr = np.repeat(1 / n, n)
            initial_state[0, :] = data[int(np.random.choice(np.arange(n), p=pr))]

            dis[:, 0] = paddle.sum((data - initial_state[0, :]) ** 2, axis=1)

            for k in range(1, self.n_clusters):
                pr = paddle.sum(dis, axis=1) / paddle.sum(dis)
                initial_state[k, :] = data[int(np.random.choice(np.arange(n), 1, p=pr.numpy()))]
                dis[:, k] = paddle.sum((data - initial_state[k, :]) ** 2, axis=1)
        else:
            n = data.shape[0]
            indices = np.random.choice(n, self.n_clusters)
            initial_state = data[indices]

        return initial_state

    @staticmethod
    def pairwise_distance(data1, data2=None):
        # using broadcast mechanism to calculate pairwise ecludian distance of data
        # the input data is N*M matrix, where M is the dimension
        # we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
        # then a simple elementwise operation of A and B will handle
        # the pairwise operation of points represented by data
        if data2 is None:
            data2 = data1

        # N*1*M
        a = paddle.unsqueeze(data1,axis=1)
        # 1*N*M
        b = paddle.unsqueeze(data2,axis=0)

        dis = (a - b) ** 2.0
        # return N*N matrix for pairwise distance
        dis = paddle.squeeze(paddle.sum(dis,axis=-1))

        return dis

    def fit(self, data):
        data = data.astype('float32')
        cluster_centers = self._initial_state(data)

        while True:
            dis = self.pairwise_distance(data, cluster_centers)

            labels = paddle.argmin(dis, axis=1)
            cluster_centers_pre = cluster_centers.clone()

            for index in range(self.n_clusters):
                selected = labels == index
                if selected.any():
                    selected = data[labels == index]
                    cluster_centers[index] = selected.mean(axis=0)
                else:
                    cluster_centers[index] = paddle.zeros_like(cluster_centers[0])

            center_shift = paddle.sum(paddle.sqrt(paddle.sum((cluster_centers - cluster_centers_pre) ** 2, axis=1)))

            if center_shift ** 2 < self.tol:
                break

        self._labels = labels
        self._cluster_centers = cluster_centers

    @property
    def labels_(self):
        return self._labels

    @property
    def cluster_centers_(self):
        return self._cluster_centers

def calc_k(image,
           max_cluster=5,
           threshold_min=0.1,
           threshold_max=0.7):
    img = paddle.transpose(image,(1,2,0)).numpy().astype(np.uint8)
    img = Image.fromarray(img).convert('RGB')
    w, h = img.size
    #     gb = 0.5 if max(w, h) < 1440 else 0
    w, h = cal_maxpool_size(w, h, 3)

    img = img.resize((w, h))
    #     img = img.filter(ImageFilter.GaussianBlur(gb))

    img = color.rgb2lab(img).reshape(w * h, -1)

    k = 2

    KMeans = KMeansGPU
    img = paddle.to_tensor(img)

    k_means_estimator = KMeans(k)
    k_means_estimator.fit(img)
    labels = k_means_estimator.labels_
    previous_labels = k_means_estimator.labels_
    previous_cluster_centers = k_means_estimator.cluster_centers_

    while True:
        cnt = Counter(labels.numpy().tolist())
        if k <= max_cluster and (cnt.most_common()[-1][1] / (w * h) > threshold_min or cnt.most_common()[0][1] / (
                w * h) > threshold_max):
            if cnt.most_common()[-2][1] / (w * h) < threshold_min:
                labels = previous_labels
                cluster_centers = previous_cluster_centers
                k = k - 1
                break
            k = k + 1
        else:
            if k > max_cluster:
                labels = previous_labels
                cluster_centers = previous_cluster_centers
                k = k - 1
            else:
                labels = k_means_estimator.labels_
                cluster_centers = k_means_estimator.cluster_centers_
            break

        previous_labels = k_means_estimator.labels_
        previous_cluster_centers = k_means_estimator.cluster_centers_

        k_means_estimator = KMeans(k)
        k_means_estimator.fit(img)
        labels = k_means_estimator.labels_

    new_clusters = paddle.argsort(paddle.norm(cluster_centers,axis=1),descending=False).numpy().tolist()
    new_clusters = [new_clusters.index(j) for j in range(k)]
    cluster_centers_norm, _ = paddle.sort(paddle.norm(cluster_centers,axis=1))
    cluster_centers_norm = cluster_centers_norm - paddle.min(cluster_centers_norm)
    cluster_centers_norm = cluster_centers_norm / paddle.max(cluster_centers_norm)

    new_labels = paddlle.zeros_like(labels)

    for i in range(k):
        new_labels[labels == i] = new_clusters[i]

    label = paddle.reshape(new_labels, (h, w))

    return label, cluster_centers_norm

def svd(a):
    a = a.numpy()
    a_gpu = gpuarray.to_gpu(a)
    u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 'S', 'S')
    return (u_gpu,vh_gpu,vh_gpu)

@GENERATORS.register()
class DecoderKMeans(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """

    def __init__(self):
        super(DecoderKMeans, self).__init__()

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

    def forward(self, ci,si,cF, sF,alpha=1):
        cs = []
        print(ci.shape)
        for i in range(ci.shape[0]):
            cp = ci[i,:,:,:]
            sp = si[i, :, :, :]
            print(cp.shape)
            content_label, content_center_norm = calc_k(cp)
            style_label, style_center_norm = calc_k(sp)

            match = cluster_matching(content_label, style_label, content_center_norm, style_center_norm)

            cf=paddle.unsqueeze(cF['r11'][i,:,:,:],axis=0)
            sf=paddle.unsqueeze(sF['r11'][i,:,:,:],axis=0)
            cs_feature = paddle.zeros_like(cf)
            for i, j in match.items():
                cl = paddle.expand_as(paddle.unsqueeze((content_label == i),axis=0),cf)
                sl = paddle.zeros_like(sf)
                for jj in j:
                    sl += paddle.expand_as(paddle.unsqueeze((style_label == jj),axis=0),sf)
                sl = sl.astype('bool')
                sub_sf = paddle.reshape(sf[sl], sf.shape[0], -1)
                cs_feature += labeled_whiten_and_color(cf, sub_sf, self.alpha, cp)

            cs.append(paddle.unsqueeze(cs_feature.unsqueeze,axis=0))

        cs = paddle.concat(cs, axis=0)
        out = adaptive_instance_normalization(cF['r41'], sF['r41'])
        out = self.resblock_41(cs)
        out = self.convblock_41(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r31'], sF['r31'])
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r21'], sF['r21'])
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out += cs
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out




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

def labeled_whiten_and_color(f_c, f_s, alpha, clabel):
    try:
        cc, ch, cw = f_c.shape
        cf = paddle.reshape((f_c * clabel), cc, -1)

        num_nonzero = paddle.sum(clabel).item() / cc
        c_mean = paddle.sum(cf, 1) / num_nonzero
        c_mean = paddle,reshape(c_mean,(cc, 1, 1)) * clabel

        cf = paddle.reshape(cf,(cc, ch, cw)) - c_mean
        cf = paddle.reshape(cf,(cc, -1))

        c_cov = paddle.mm(cf, cf.t()) / (num_nonzero - 1)
        c_u, c_e, c_v = svd(c_cov)

        c_e = paddle.Tensor(c_e.get())
        c_v = paddle.Tensor(c_v.get())

        c_d = c_e.pow(-0.5)

        w_step1 = paddle.mm(c_v, paddle.diag(c_d))
        w_step2 = paddle.mm(w_step1, (c_v.t()))
        whitened = paddle.mm(w_step2, cf)

        sf = f_s
        sc, shw = sf.shape
        s_mean = paddle.mean(f_s, 1, keepdim=True)
        sf = sf - s_mean

        s_cov = paddle.mm(sf, sf.t()) / (shw - 1)
        s_u, s_e, s_v = svd(s_cov)
        s_d = s_e.pow(0.5)

        s_v = paddle.Tensor(s_v/get())

        c_step1 = paddle.mm(s_v, paddle.diag(s_d))
        c_step2 = paddle.mm(c_step1, s_v.t())
        colored = paddle.mm(c_step2, whitened).reshape(cc, ch, cw)

        colored = colored + s_mean.reshape(sc, 1, 1) * clabel
        colored_feature = alpha * colored + (1 - alpha) * (f_c * clabel)
    except:
        colored_feature = f_c * clabel

    return colored_feature

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


def adaptive_instance_normalization(content_feat, style_feat):
    """adaptive_instance_normalization.

    Args:
        content_feat (Tensor): Tensor with shape (N, C, H, W).
        style_feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized content_feat with shape (N, C, H, W)
    """
    assert (content_feat.shape[:2] == style_feat.shape[:2])
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def thumb_adaptive_instance_normalization(content_thumb_feat, content_patch_feat, style_feat, thumb_or_patch='thumb'):
    """adaptive_instance_normalization.

    Args:
        content_feat (Tensor): Tensor with shape (N, C, H, W).
        content_patch_feat (Tensor): Tensor with shape (N, C, H, W).
        style_feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized content_feat with shape (N, C, H, W)
    """
    assert (content_thumb_feat.shape[:2] == style_feat.shape[:2])
    size = content_thumb_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_thumb_mean, content_thumb_std = calc_mean_std(content_thumb_feat)

    content_thumb_feat = (content_thumb_feat - content_thumb_mean.expand(size)) / content_thumb_std.expand(size)
    content_thumb_feat = content_thumb_feat * style_std.expand(size) + style_mean.expand(size)

    if thumb_or_patch == 'thumb':
        return content_thumb_feat

    elif thumb_or_patch == 'patch':
        content_patch_feat = (content_patch_feat - content_thumb_mean.expand(size)) / content_thumb_std.expand(size)
        content_patch_feat = content_patch_feat * style_std.expand(size) + style_mean.expand(size)

        return content_patch_feat

class NoiseBlock(nn.Layer):
    def __init__(self, channels,noise_weight=0):
        super().__init__()
        #self.weight = paddle.create_parameter((1,channels),dtype='float32',is_bias=True)
        self.noise_weight=noise_weight
    def change_noise_weight(self,weight):
        print('changing noise weight to '+str(weight))
        self.noise_weight = weight
    def forward(self,x):
        noise = paddle.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
        if self.noise_weight>0:
            x = x + noise * self.noise_weight
        else:
            x = x + noise
        return x

class ResnetBlock(nn.Layer):
    """Residual block.

    It has a style of:
        ---Pad-Conv-ReLU-Pad-Conv-+-
         |________________________|

    Args:
        dim (int): Channel number of intermediate features.
    """
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(dim, dim, (3, 3)), nn.ReLU(),
                                        nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(dim, dim, (3, 3)))

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvBlock(nn.Layer):
    """convolution block.

    It has a style of:
        ---Pad-Conv-ReLU---

    Args:
        dim1 (int): Channel number of input features.
        dim2 (int): Channel number of output features.
    """
    def __init__(self, dim1, dim2,noise=0):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(dim1, dim2, (3, 3)),
                                        nn.ReLU())
        if noise==1:
            self.conv_block.add_sublayer('noise',NoiseBlock(dim2))

    def forward(self, x):
        out = self.conv_block(x)
        return out


@GENERATORS.register()
class DecoderNet(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(DecoderNet, self).__init__()

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
        out = self.resblock_41(out)
        out = self.convblock_41(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r31'], sF['r31'])
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r21'], sF['r21'])
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out

@GENERATORS.register()
class DecoderNetDeep(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self,noise=0):
        super(DecoderNetDeep, self).__init__()

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
        out = self.resblock_41(out)
        out = self.convblock_41(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r31'], sF['r31'])
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r21'], sF['r21'])
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out

@GENERATORS.register()
class DecoderThumbDeep(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(DecoderThumbDeep, self).__init__()

        self.resblock_41 = ResnetBlock(512)
        self.convblock_411 = ConvBlock(512,512,noise)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_311 = ConvBlock(256,256,noise)
        self.convblock_31 = ConvBlock(256, 128)

        self.convblock_211 = ConvBlock(128, 128)
        self.convblock_21 = ConvBlock(128, 128,noise)
        self.convblock_22 = ConvBlock(128, 64)
        self.convblock_111 = ConvBlock(64, 64)
        self.convblock_11 = ConvBlock(64, 64,noise)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))

    def forward(self, cF, sF, cpF, thumb_or_patch='thumb'):

        #out = thumb_adaptive_instance_normalization(cF['r51'], cpF['r51'], sF['r51'], thumb_or_patch=thumb_or_patch)
        out = thumb_adaptive_instance_normalization(cF['r41'], cpF['r41'], sF['r41'], thumb_or_patch=thumb_or_patch)
        thumb_ada = {'r41':out.clone()}
        out = self.resblock_41(out)
        out = self.convblock_411(out)
        out = self.convblock_41(out)

        out = self.upsample(out)

        thumb_ada['r31'] = thumb_adaptive_instance_normalization(cF['r31'], cpF['r31'],sF['r31'], thumb_or_patch=thumb_or_patch)
        out += thumb_ada['r31']
        out = self.resblock_31(out)
        out = self.convblock_311(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        thumb_ada['r21'] = thumb_adaptive_instance_normalization(cF['r21'], cpF['r21'], sF['r21'],
                                                           thumb_or_patch=thumb_or_patch)
        out += thumb_ada['r21']
        out = self.convblock_211(out)
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_111(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out,thumb_ada

@GENERATORS.register()
class DecoderThumbNet(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(DecoderThumbNet, self).__init__()

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

    def forward(self, cF, sF, cpF, thumb_or_patch='thumb'):

        #out = thumb_adaptive_instance_normalization(cF['r51'], cpF['r51'], sF['r51'], thumb_or_patch=thumb_or_patch)
        out = thumb_adaptive_instance_normalization(cF['r41'], cpF['r41'], sF['r41'], thumb_or_patch=thumb_or_patch)
        thumb_ada = {'r41':out.clone()}
        out = self.resblock_41(out)
        out = self.convblock_41(out)

        out = self.upsample(out)

        '''
        if thumb_or_patch=='thumb':
            thumb_ada['r31']=adaptive_instance_normalization(cF['r31'], sF['r31'])
        else:
            thumb_ada['r31']=adaptive_instance_normalization(cpF['r31'], sF['r31'])
        
        '''
        thumb_ada['r31'] = thumb_adaptive_instance_normalization(cF['r31'], cpF['r31'],sF['r31'], thumb_or_patch=thumb_or_patch)
        out += thumb_ada['r31']
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        '''
        if thumb_or_patch=='thumb':
            thumb_ada['r21']=adaptive_instance_normalization(cF['r21'], sF['r21'])
        else:
            thumb_ada['r21']=adaptive_instance_normalization(cpF['r21'], sF['r21'])
        '''
        thumb_ada['r21'] = thumb_adaptive_instance_normalization(cF['r21'], cpF['r21'], sF['r21'],
                                                           thumb_or_patch=thumb_or_patch)
        out += thumb_ada['r21']
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out,thumb_ada


@GENERATORS.register()
class Encoder(nn.Layer):
    """Encoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        vgg_net = nn.Sequential(
            nn.Conv2D(3, 3, (1, 1)),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )

        weight_path = get_path_from_url(
            'https://paddlegan.bj.bcebos.com/models/vgg_normalised.pdparams')
        vgg_net.set_dict(paddle.load(weight_path))
        self.enc_1 = nn.Sequential(*list(
            vgg_net.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(
            vgg_net.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(
            vgg_net.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(
            vgg_net.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(
            vgg_net.children())[31:44])  # relu4_1 -> relu5_1

    def forward(self, x):
        out = {}
        x = self.enc_1(x)
        out['r11'] = x
        x = self.enc_2(x)
        out['r21'] = x
        x = self.enc_3(x)
        out['r31'] = x
        x = self.enc_4(x)
        out['r41'] = x
        x = self.enc_5(x)
        out['r51'] = x
        return out


@GENERATORS.register()
class RevisionNet(nn.Layer):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, input_nc=6):
        super(RevisionNet, self).__init__()
        DownBlock = []
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(input_nc, 64, (3, 3)),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU()
        ]

        self.resblock = ResnetBlock(64)

        UpBlock = []
        UpBlock += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 3, (3, 3))
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        out = self.DownBlock(input)
        out = self.resblock(out)
        res_block = out.clone()
        out = self.UpBlock(out)
        return out, res_block

@GENERATORS.register()
class RevisionNetThumb(nn.Layer):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, input_nc=6):
        super(RevisionNetThumb, self).__init__()
        DownBlock = []
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(input_nc, 64, (3, 3)),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU()
        ]

        self.resblock = ResnetBlock(64)

        UpBlock = []
        UpBlock += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 3, (3, 3))
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input,thumbnail=False,alpha=1):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        out = self.DownBlock(input)
        out = self.resblock(out)
        feats = out.clone()
        if type(thumbnail) != bool:
            feats = adaptive_instance_normalization(out, thumbnail)
            out = alpha * feats + (1 - alpha) * out
        out = self.UpBlock(out)
        return out,feats


@GENERATORS.register()
class RevisionNet32Feats(nn.Layer):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, input_nc=6):
        super(RevisionNet32Feats, self).__init__()
        DownBlock = []
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(input_nc, 128, (3, 3)),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3), stride=1),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3), stride=1),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 64, (3, 3), stride=1),
            nn.ReLU(),
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=1),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU(),
        ]

        self.resblock = ResnetBlock(64)

        UpBlock = []
        UpBlock += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 128, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 3, (3, 3))
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        out = self.DownBlock(input)
        out = self.resblock(out)
        out = self.UpBlock(out)
        return out

@GENERATORS.register()
class RevisionNetDeepThumb(nn.Layer):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, input_nc=6,noise=0,noise_weight=0):
        super(RevisionNetDeepThumb, self).__init__()
        DownBlock = []
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(input_nc, 128, (3, 3)),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3), stride=1),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3), stride=1),
            nn.ReLU()
        ]
        if noise==1:
            DownBlock+=[NoiseBlock(128,noise_weight)]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 64, (3, 3), stride=1),
            nn.ReLU(),
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=1),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU(),
        ]
        if noise==1:
            DownBlock+=[NoiseBlock(64,noise_weight)]

        self.resblock = ResnetBlock(64)

        UpBlock = []
        if noise==1:
            UpBlock+=[NoiseBlock(64,noise_weight)]
        UpBlock += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU()]

        if noise==1:
            UpBlock+=[NoiseBlock(64,noise_weight)]

        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 128, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3)),
            nn.ReLU()
        ]
        if noise==1:
            UpBlock+=[NoiseBlock(128,noise_weight)]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 3, (3, 3))
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def change_noise_weight(self,new_weight):
        def changeweight(input):
            if hasattr(input,'noise_weight'):
                input.change_noise_weight(new_weight)
            else:
                pass
        for layer in [self.DownBlock,self.UpBlock]:
            layer.apply(changeweight)

    def test_noise_weight_change(self):
        a = 1
        def test(input):
            if hasattr(input,'noise_weight'):
                print('has noise -'+str(a))
            else:
                pass
        for layer in [self.DownBlock,self.UpBlock]:
            layer.apply(test)

    def forward(self, input,thumbnail=False,alpha=1):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        out = self.DownBlock(input)
        out = self.resblock(out)
        if type(thumbnail) != bool:
            feats = adaptive_instance_normalization(out, thumbnail)
            out = alpha * feats + (1 - alpha) * out
        feats = out.clone()
        out = self.UpBlock(out)
        return out, feats