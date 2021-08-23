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
import paddle.nn.functional as F
import math,random,os,re
from PIL import Image
import numpy as np
from .base_model import BaseModel

import shutil
from .builder import MODELS
from .generators.builder import build_generator
from .criterions import build_criterion
from .discriminators.builder import build_discriminator

from ..modules.init import init_weights
from ..utils.visual import tensor2img, save_image
from ..utils.filesystem import makedirs, save, load


def xdog(im, g, g2,morph_conv,gamma=.94, phi=50, eps=-.5, morph_cutoff=8.88,morphs=1,minmax=False):
    # Source : https://github.com/CemalUnal/XDoG-Filter
    # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
    # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
    #imf1 = paddle.concat(x=[g(paddle.unsqueeze(im[:,0,:,:].detach(),axis=1)),g(paddle.unsqueeze(im[:,1,:,:].detach(),axis=1)),g(paddle.unsqueeze(im[:,2,:,:].detach(),axis=1))],axis=1)

    imf2=paddle.zeros_like(im)
    imf1=paddle.zeros_like(im)
    imf1.stop_gradient=True
    imf2.stop_gradient=True
    imf2=g2(im)
    imf1=g(im)
    #imf2 = g2(im.detach())
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps).astype('float32') * 1.0  + (imdiff >= eps).astype('float32') * (1.0 + paddle.tanh(phi * imdiff))
    if type(minmax)==bool:
        min = imdiff.min(axis=[2,3],keepdim=True)
        max = imdiff.max(axis=[2,3],keepdim=True)
    else:
        min=minmax[0]
        max=minmax[1]
    imdiff -= paddle.expand_as(min,imdiff)
    imdiff /= paddle.expand_as(max,imdiff)
    if type(minmax)==bool:
        mean = imdiff.mean(axis=[2,3],keepdim=True)
    else:
        mean=minmax[2]
    exmean=paddle.expand_as(mean,imdiff)
    for i in range(morphs):
        morphed=morph_conv(imdiff)
        morphed.stop_gradient=True
        passedlow= paddle.multiply((imdiff>= exmean).astype('float32'),(morphed>= morph_cutoff).astype('float32'))
    for i in range(morphs):
        passed = morph_conv(passedlow)
        passed= (passed>0).astype('float32')
    return passed, [min,max,mean]

def gaussian(kernel_size, sigma,channels=3):
    x_coord = paddle.arange(kernel_size)
    x_grid = paddle.expand(x_coord,(kernel_size,kernel_size))
    y_grid = x_grid.t()
    xy_grid = paddle.stack([x_grid, y_grid], axis=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      paddle.exp(
                          -paddle.sum((xy_grid - mean) ** 2., axis=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / paddle.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.reshape((1,1, kernel_size, kernel_size))

    return gaussian_kernel

@MODELS.register()
class LapStyleDraModel(BaseModel):
    def __init__(self,
                 generator_encode,
                 generator_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleDraModel, self).__init__()

        # define generators
        self.nets['net_enc'] = build_generator(generator_encode)
        self.nets['net_dec'] = build_generator(generator_decode)
        init_weights(self.nets['net_dec'])
        self.set_requires_grad([self.nets['net_enc']], False)

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)
        self.stylized = self.nets['net_dec'](self.cF, self.sF)
        self.visual_items['stylized'] = self.stylized

    def backward_Dec(self):
        self.tF = self.nets['net_enc'](self.stylized)
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers[:-1]:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """IDENTITY LOSSES"""
        self.Icc = self.nets['net_dec'](self.cF, self.cF)
        self.l_identity1 = self.calc_content_loss(self.Icc, self.ci)
        self.Fcc = self.nets['net_enc'](self.Icc)
        self.l_identity2 = 0
        for layer in self.content_layers:
            self.l_identity2 += self.calc_content_loss(self.Fcc[layer],
                                                       self.cF[layer])
        self.losses['l_identity1'] = self.l_identity1
        self.losses['l_identity2'] = self.l_identity2
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt

        self.loss = self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.l_identity1 * 50 + self.l_identity2 * 1 + self.loss_style_remd * 3 + \
                    self.loss_content_relt * 16
        self.loss.backward()

        return self.loss

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights"""
        self.forward()
        optimizers['optimG'].clear_grad()
        self.backward_Dec()
        self.optimizers['optimG'].step()

@MODELS.register()
class LapStyleDraXDOG(BaseModel):
    def __init__(self,
                 generator_encode,
                 generator_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gram_errors=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0,
                 morph_cutoff=8,
                 gamma=.96):

        super(LapStyleDraXDOG, self).__init__()

        # define generators
        self.nets['net_enc'] = build_generator(generator_encode)
        self.nets['net_dec'] = build_generator(generator_decode)
        init_weights(self.nets['net_dec'])
        self.set_requires_grad([self.nets['net_enc']], False)

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gram_errors = build_criterion(gram_errors)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.morph_cutoff=morph_cutoff
        self.gamma=gamma
        g = np.repeat(gaussian(7, 1).numpy(), 3, axis=0)
        g2 = np.repeat(gaussian(19, 3).numpy(), 3, axis=0)
        self.gaussian_filter = paddle.nn.Conv2D(3, 3, 7,
                                                groups=3, bias_attr=False,
                                                padding=3, padding_mode='reflect',
                                                weight_attr=paddle.ParamAttr(
                                                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                        value=g), trainable=False)
                                                )
        self.gaussian_filter_2 = paddle.nn.Conv2D(3, 3, 19,
                                                  groups=3, bias_attr=False,
                                                  padding=9, padding_mode='reflect',
                                                  weight_attr=paddle.ParamAttr(
                                                      initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                          value=g2), trainable=False)
                                                  )

        self.morph_conv = paddle.nn.Conv2D(3, 3, 3, padding=1, groups=3,
                                           padding_mode='reflect', bias_attr=False,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.Constant(
                                                   value=1), trainable=False)
                                           )

        print(gaussian(7, 1))
        self.set_requires_grad([self.morph_conv], False)
        self.set_requires_grad([self.gaussian_filter],False)
        self.set_requires_grad([self.gaussian_filter_2],False)

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)
        self.stylized = self.nets['net_dec'](self.cF, self.sF)
        self.visual_items['stylized'] = self.stylized

    def backward_Dec(self):

        self.cX,_ = xdog(self.ci.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,gamma=self.gamma,morph_cutoff=self.morph_cutoff,morphs=1)
        self.sX,_ = xdog(self.si.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,gamma=self.gamma,morph_cutoff=self.morph_cutoff,morphs=1)
        self.visual_items['cx'] = self.cX
        self.visual_items['sx'] = self.sX
        self.cXF = self.nets['net_enc'](self.cX)
        self.sXF = self.nets['net_enc'](self.sX)
        stylized_dog,_ = xdog(self.stylized,self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,gamma=self.gamma,morph_cutoff=self.morph_cutoff,morphs=1)
        self.cdogF = self.nets['net_enc'](stylized_dog)
        self.visual_items['cdog']=stylized_dog
        self.tF = self.nets['net_enc'](self.stylized)
        """content loss"""
        self.loss_c = 0
        for idx, layer in enumerate(self.content_layers):
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """IDENTITY LOSSES"""
        self.Icc = self.nets['net_dec'](self.cF, self.cF)
        self.l_identity1 = self.calc_content_loss(self.Icc, self.ci)
        self.Fcc = self.nets['net_enc'](self.Icc)
        self.l_identity2 = 0
        for layer in self.content_layers:
            self.l_identity2 += self.calc_content_loss(self.Fcc[layer],
                                                       self.cF[layer])
        self.losses['l_identity1'] = self.l_identity1
        self.losses['l_identity2'] = self.l_identity2
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt

        mxdog_content = self.calc_content_loss(self.tF['r31'], self.cXF['r31'])
        mxdog_content_contraint = self.calc_content_loss(self.cdogF['r31'], self.cXF['r31'])
        mxdog_content_img = self.gram_errors(self.cdogF['r31'],self.sXF['r31'])

        self.losses['loss_MD'] = mxdog_content*.3
        self.losses['loss_CnsC'] = mxdog_content_contraint*100
        self.losses['loss_CnsS'] = mxdog_content_img*1000

        self.loss = self.loss_c * self.content_weight + self.style_weight * (self.loss_s +3*self.loss_style_remd)+\
                    self.l_identity1 * 100 + self.l_identity2 * 1 + \
                    mxdog_content * .3 + mxdog_content_contraint *50 + mxdog_content_img * 1000+\
                    self.loss_content_relt * 16
        self.loss.backward()

        return self.loss

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights"""
        self.forward()
        optimizers['optimG'].clear_grad()
        self.backward_Dec()
        self.optimizers['optimG'].step()


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):
    """
    Laplacian

    return:
       x - upsample(downsample(x))
    """
    return x - tensor_resample(
        tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]),
        [x.shape[2], x.shape[3]])

def laplacian_conv(x,kernel):
    lap = kernel(x)
    return lap

def make_laplace_pyramid(x, levels):
    """
    Make Laplacian Pyramid
    """
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(
            current,
            (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid

def make_laplace_conv_pyramid(x, levels,kernel):
    """
    Make Laplacian Pyramid
    """
    pyramid = []
    current = x
    for i in range(levels):
        lap = kernel(current)
        pyramid.append(lap)
        current = tensor_resample(
            current,
            (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid


def fold_laplace_pyramid(pyramid):
    """
    Fold Laplacian Pyramid
    """
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    return current

def fold_laplace_patch(pyramid,patch=False):
    """
    Fold Laplacian Pyramid
    """
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    if not type(patch)==bool:
        for i in patch:
            current = current+i
    return current


@MODELS.register()
class LapStyleRevFirstModel(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevFirstModel, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)

        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define revision-net params
        self.nets['net_rev'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

        self.pyr_ci = make_laplace_pyramid(self.ci, 1)
        self.pyr_si = make_laplace_pyramid(self.si, 1)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[1])
        sF = self.nets['net_enc'](self.pyr_si[1])

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])

        self.stylized = stylized_rev
        self.visual_items['stylized'] = self.stylized

    def backward_G(self):
        self.tF = self.nets['net_enc'](self.stylized)
        self.cF = self.nets['net_enc'](self.pyr_ci[2])
        self.sF = self.nets['net_enc'](self.pyr_si[2])
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt
        """gan loss"""
        pred_fake = self.nets['netD'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN

        self.loss = self.loss_G_GAN + self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.loss_style_remd * 10 + self.loss_content_relt * 16
        self.loss.backward()
        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_real = self.nets['netD'](self.pyr_si[2])
        self.loss_D_real = self.gan_criterion(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D
        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G()
        optimizers['optimG'].step()

@MODELS.register()
class LapStyleRevFirstMXDOG(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_first_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gram_errors=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevFirstMXDOG, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)

        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define revision-net params
        self.nets['net_rev'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev'])
        self.nets['netD_first'] = build_discriminator(revnet_first_discriminator)
        init_weights(self.nets['netD_first'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gram_errors = build_criterion(gram_errors)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        g=np.repeat(gaussian(7, 1).numpy(),3,axis=0)
        g2=np.repeat(gaussian(19, 3).numpy(),3,axis=0)
        self.gaussian_filter = paddle.nn.Conv2D(3, 3,7,
                            groups=3, bias_attr=False,
                            padding=3, padding_mode='reflect',
                                            weight_attr=paddle.ParamAttr(
                                                initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                    value=g), trainable=False)
                                            )
        self.gaussian_filter_2 = paddle.nn.Conv2D(3, 3,19,
                                groups=3, bias_attr=False,
                                padding=9, padding_mode='reflect',
                                weight_attr = paddle.ParamAttr(
                                        initializer=paddle.fluid.initializer.NumpyArrayInitializer(value=g2), trainable=False)
                                    )

        self.morph_conv = paddle.nn.Conv2D(3,3,3,padding=1,groups=3,
                                           padding_mode='reflect',bias_attr=False,
                                           weight_attr = paddle.ParamAttr(
                                        initializer=paddle.fluid.initializer.Constant(
                                                        value=1), trainable=False)
                                    )
        l = np.repeat(np.array([[[[-8,-8,-8],[-8,1,-8],[-8,-8,-8]]]]),3,axis=0)
        self.lap_filter = paddle.nn.Conv2D(3,3,(3,3),stride=1,bias_attr=False,
                                padding=1, groups=3,padding_mode='reflect',
                                weight_attr = paddle.ParamAttr(
                                        initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                            value=l), trainable=False)
                                           )

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

        self.pyr_ci = make_laplace_conv_pyramid(self.ci, 1,self.lap_filter)
        self.pyr_si = make_laplace_conv_pyramid(self.si, 1,self.lap_filter)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[1])
        sF = self.nets['net_enc'](self.pyr_si[1])

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap,_ = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])

        self.stylized = stylized_rev
        self.visual_items['stylized'] = self.stylized

    def backward_G(self):
        self.tF = self.nets['net_enc'](self.stylized)
        self.cF = self.nets['net_enc'](self.pyr_ci[2])
        self.sF = self.nets['net_enc'](self.pyr_si[2])
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt
        """gan loss"""
        pred_fake = self.nets['netD_first'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN

        self.cX,_ = xdog(self.ci.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2)
        self.sX,_ = xdog(self.si.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2)
        self.cXF = self.nets['net_enc'](self.cX)
        self.sXF = self.nets['net_enc'](self.sX)
        self.visual_items['cx'] = self.cX
        self.visual_items['sx'] = self.sX
        stylized_dog,_ = xdog(self.stylized,self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2)
        self.cdogF = self.nets['net_enc'](stylized_dog)

        mxdog_content = self.calc_content_loss(self.tF['r31'], self.cXF['r31'])+self.calc_content_loss(self.tF['r41'], self.cXF['r41'])
        mxdog_content_contraint = self.calc_content_loss(self.cdogF['r31'], self.cXF['r31'])+self.calc_content_loss(self.cdogF['r41'], self.cXF['r41'])
        mxdog_content_img = self.gram_errors(self.cdogF['r31'],self.sXF['r31'])+self.gram_errors(self.cdogF['r41'],self.sXF['r41'])

        self.losses['loss_MD_p'] = mxdog_content*.3
        self.losses['loss_CnsC_p'] = mxdog_content_contraint*100
        self.losses['loss_CnsS_p'] = mxdog_content_img*1000
        mxdogloss=mxdog_content * .3 + mxdog_content_contraint *100 + mxdog_content_img * 1000

        self.loss = self.loss_G_GAN*1.5 + self.loss_c * self.content_weight + self.style_weight * (self.loss_s +\
                    self.loss_style_remd * 3) + self.loss_content_relt * 20 + mxdogloss
        self.loss.backward()
        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD_first'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_real = self.nets['netD_first'](self.pyr_si[2])
        self.loss_D_real = self.gan_criterion(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D
        self.set_requires_grad(self.nets['netD_first'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()

        # update G
        self.set_requires_grad(self.nets['netD_first'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G()
        optimizers['optimG'].step()


@MODELS.register()
class LapStyleRevSecondModel(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevSecondModel, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)
        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define the first revnet params
        self.nets['net_rev'] = build_generator(revnet_generator)
        self.set_requires_grad([self.nets['net_rev']], False)

        # define the second revnet params
        self.nets['net_rev_2'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev_2'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

        self.pyr_ci = make_laplace_pyramid(self.ci, 2)
        self.pyr_si = make_laplace_pyramid(self.si, 2)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[2])
        sF = self.nets['net_enc'](self.pyr_si[2])

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[1], stylized_up], axis=1)
        stylized_rev_lap = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
        self.visual_items['stylized_rev_first'] = stylized_rev
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap_second = self.nets['net_rev_2'](revnet_input)
        stylized_rev_second = fold_laplace_pyramid(
            [stylized_rev_lap_second, stylized_rev_lap, stylized_small])

        self.stylized = stylized_rev_second
        self.visual_items['stylized'] = self.stylized

    def backward_G(self):
        self.tF = self.nets['net_enc'](self.stylized)
        self.cF = self.nets['net_enc'](self.pyr_ci[3])
        self.sF = self.nets['net_enc'](self.pyr_si[3])
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(self.tF['r41'],
                                                        self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt
        """gan loss"""
        pred_fake = self.nets['netD'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN

        self.loss = self.loss_G_GAN + self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.loss_style_remd * 10 + self.loss_content_relt * 16
        self.loss.backward()
        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_real = self.nets['netD'](self.pyr_si[3])
        self.loss_D_real = self.gan_criterion(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D

        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G()
        optimizers['optimG'].step()

@MODELS.register()
class LapStyleDraThumbModel(BaseModel):
    def __init__(self,
                 generator_encode,
                 generator_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleDraThumbModel, self).__init__()

        # define generators
        self.nets['net_enc'] = build_generator(generator_encode)
        self.nets['net_dec'] = build_generator(generator_decode)
        init_weights(self.nets['net_dec'])
        self.set_requires_grad([self.nets['net_enc']], False)

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.cp = paddle.to_tensor(input['cp'])
        self.sp = paddle.to_tensor(input['sp'])
        self.visual_items['cp'] = self.cp
        self.position = input['position']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)
        self.cpF = self.nets['net_enc'](self.cp)
        self.stylized_thumb,self.stylized_thumb_feat = self.nets['net_dec'](self.cF, self.sF, self.cpF, 'thumb')
        #self.stylized_patch,self.stylized_patch_feat = self.nets['net_dec'](self.cF, self.sF, self.cpF, 'patch')
        self.visual_items['stylized_thumb'] = self.stylized_thumb
        #self.visual_items['stylized_patch'] = self.stylized_patch
        self.visual_items['style']=self.si

    def backward_Dec(self):
        with paddle.no_grad():
            g_t_thumb_up = F.interpolate(self.visual_items['stylized_thumb'], scale_factor=2, mode='bilinear', align_corners=False)
            g_t_thumb_crop = paddle.slice(g_t_thumb_up,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
            self.tt_cropF = self.nets['net_enc'](g_t_thumb_crop)
            #style_patch = F.interpolate(self.visual_items['si'], scale_factor=2, mode='bilinear', align_corners=False)
            #style_patch_crop = paddle.slice(style_patch,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
            #self.spCrop = self.nets['net_enc'](self.sp)
        self.ttF = self.nets['net_enc'](self.stylized_thumb)
        self.tpF = self.nets['net_enc'](self.stylized_patch)

        """content loss"""
        self.loss_content=0
        for layer in self.content_layers:
            self.loss_content += self.calc_content_loss(self.ttF[layer],
                                                      self.cF[layer],
                                                      norm=True)
        self.losses['loss_content'] = self.loss_content

        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.ttF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s

        self.Icc,_ = self.nets['net_dec'](self.cF, self.cF, self.cpF,'thumb')
        self.l_identity1 = self.calc_content_loss(self.Icc, self.ci)
        self.Fcc = self.nets['net_enc'](self.Icc)
        self.l_identity2 = 0
        for layer in self.content_layers:
            self.l_identity2 += self.calc_content_loss(self.Fcc[layer],
                                                       self.cF[layer])

        self.loss_style_remd = self.calc_style_emd_loss(
            self.ttF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.ttF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.ttF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.ttF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt

        """patch loss"""
        self.loss_patch = 0
        #self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            self.loss_patch += self.calc_content_loss(self.tpF[layer],
                                                      self.tt_cropF[layer])
        self.losses['loss_patch'] =  self.loss_patch

        self.losses['l_identity1'] = self.l_identity1
        self.losses['l_identity2'] = self.l_identity2

        self.loss = self.loss_s * self.style_weight +\
                    self.l_identity1 * 50 + self.l_identity2 * 1 +\
                    self.loss_content * self.content_weight+\
                    self.loss_style_remd * 18 +\
                    self.loss_content_relt * 24 +self.loss_patch * 18 * self.content_weight
        self.loss.backward()

        return self.loss

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights"""
        self.forward()
        optimizers['optimG'].clear_grad()
        self.backward_Dec()
        self.optimizers['optimG'].step()

@MODELS.register()
class LapStyleRevFirstThumb(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0,
                 ada_alpha=1.0,
                 style_patch_alpha=.5,
                 use_mxdog=0):

        super(LapStyleRevFirstThumb, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)

        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define revision-net params
        self.nets['net_rev'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])
        self.nets['netD_patch'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD_patch'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.ada_alpha = ada_alpha
        self.style_patch_alpha = style_patch_alpha
        self.use_mxdog = use_mxdog
        if self.use_mxdog==1:
            g=np.repeat(gaussian(7, 1).numpy(),3,axis=0)
            g2=np.repeat(gaussian(19, 3).numpy(),3,axis=0)
            self.gaussian_filter = paddle.nn.Conv2D(3, 3,7,
                                groups=3, bias_attr=False,
                                padding=3, padding_mode='reflect',
                                                weight_attr=paddle.ParamAttr(
                                                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                        value=g), trainable=False)
                                                )
            self.gaussian_filter_2 = paddle.nn.Conv2D(3, 3,19,
                                    groups=3, bias_attr=False,
                                    padding=9, padding_mode='reflect',
                                    weight_attr = paddle.ParamAttr(
                                            initializer=paddle.fluid.initializer.NumpyArrayInitializer(value=g2), trainable=False)
                                        )

            self.morph_conv = paddle.nn.Conv2D(3,3,3,padding=1,groups=3,
                                               padding_mode='reflect',bias_attr=False,
                                               weight_attr = paddle.ParamAttr(
                                            initializer=paddle.fluid.initializer.Constant(
                                                            value=1), trainable=False)
                                        )
            l = np.repeat(np.array([[[[-8,-8,-8],[-8,1,-8],[-8,-8,-8]]]]),3,axis=0)
            self.lap_filter = paddle.nn.Conv2D(3,3,(3,3),stride=1,bias_attr=False,
                                    padding=1, groups=3,padding_mode='reflect',
                                    weight_attr = paddle.ParamAttr(
                                            initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                value=l), trainable=False)
                                               )

    def setup_input(self, input):

        self.position = input['position']
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.cp = input['cp']
        self.sp = input['sp']
        self.visual_items['cp'] = self.cp

        self.pyr_ci = make_laplace_conv_pyramid(self.ci, 1,self.lap_filter)
        self.pyr_si = make_laplace_conv_pyramid(self.si, 1,self.lap_filter)
        self.pyr_cp = make_laplace_conv_pyramid(self.cp, 1,self.lap_filter)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)
        self.pyr_cp.append(self.cp)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[1])
        sF = self.nets['net_enc'](self.pyr_si[1])
        transformed = paddle.slice(self.sp, axes=[2, 3], starts=[self.position[0], self.position[2]],
                                   ends=[self.position[1], self.position[3]])
        self.spF = self.nets['net_enc'](transformed)
        self.cpF = self.nets['net_enc'](self.cp)

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap,stylized_feats = self.nets['net_rev'](revnet_input.detach())
        #self.ttF_res=self.ttF_res.detach()
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])

        stylized_up = F.interpolate(stylized_rev, scale_factor=2)
        p_stylized_up = paddle.slice(stylized_up,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
        p_revnet_input = paddle.concat(x=[self.pyr_cp[0], p_stylized_up], axis=1)
        p_stylized_rev_lap,stylized_feats = self.nets['net_rev'](p_revnet_input.detach(),stylized_feats.detach(),self.ada_alpha)
        p_stylized_rev = fold_laplace_pyramid([p_stylized_rev_lap, p_stylized_up.detach()])

        self.stylized = stylized_rev
        self.p_stylized = p_stylized_rev
        self.visual_items['stylized_up'] = stylized_up
        self.visual_items['stylized'] = self.stylized
        self.visual_items['stylized_patch'] = self.p_stylized

    def backward_G(self, optimizer):

        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)

        with paddle.no_grad():
            g_t_thumb_up = F.interpolate(self.visual_items['stylized'], scale_factor=2, mode='bilinear', align_corners=False)
            g_t_thumb_crop = paddle.slice(g_t_thumb_up,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
            self.tt_cropF = self.nets['net_enc'](g_t_thumb_crop)

        self.ttF = self.nets['net_enc'](self.stylized)
        self.tpF = self.nets['net_enc'](self.p_stylized)


        self.loss_content = 0
        for layer in self.content_layers:
            self.loss_content += self.calc_content_loss(self.ttF[layer],
                                                      self.cF[layer],
                                                      norm=True)
        self.losses['loss_content'] = self.loss_content

        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.ttF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s

        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.ttF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.ttF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.ttF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.ttF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt

        pred_fake = self.nets['netD'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN

        if self.use_mxdog==1:
            self.cX = xdog(self.ci.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv)
            self.sX = xdog(self.si.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv)
            self.cXF = self.nets['net_enc'](self.cX)
            self.sXF = self.nets['net_enc'](self.sX)
            self.visual_items['cx'] = self.cX
            self.visual_items['sx'] = self.sX
            stylized_dog = xdog(self.stylized,self.gaussian_filter,self.gaussian_filter_2,self.morph_conv)
            self.cdogF = self.nets['net_enc'](stylized_dog)

            mxdog_content = self.calc_content_loss(self.ttF['r31'], self.cXF['r31'])
            mxdog_content_contraint = self.calc_content_loss(self.cdogF['r31'], self.cXF['r31'])
            mxdog_content_img = self.calc_style_loss(self.cdogF['r31'],self.sXF['r31'])

            self.losses['loss_MD_p'] = mxdog_content*.05
            self.losses['loss_CnsC_p'] = mxdog_content_contraint*100
            self.losses['loss_CnsS_p'] = mxdog_content_img*500
            mxdogloss=mxdog_content * .0125 + mxdog_content_contraint *25 + mxdog_content_img * 125
        else:
            mxdogloss=0

        self.loss = self.loss_G_GAN + self.loss_s * self.style_weight +\
                    self.loss_content * self.content_weight+\
                    self.loss_style_remd * 16 +\
                    self.loss_content_relt * 16 + mxdogloss
        self.loss.backward()
        optimizer.step()


        """patch loss"""
        self.loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            self.loss_patch += self.calc_content_loss(self.tpF[layer],
                                                      self.tt_cropF[layer])
        self.losses['loss_patch'] = self.loss_patch

        self.loss_content_p = 0
        for layer in self.content_layers:
            self.loss_content_p += self.calc_content_loss(self.tpF[layer],
                                                      self.cpF[layer],
                                                      norm=True)
        self.losses['loss_content_p'] = self.loss_content_p

        self.loss_ps = 0
        for layer in self.style_layers:
            self.loss_ps += self.calc_style_loss(self.tpF[layer],
                                                          self.sF[layer])
        self.losses['loss_ps'] = self.loss_ps

        self.loss_psp = 0
        '''
        for layer in self.content_layers:
            self.loss_psp += self.calc_style_loss(self.tpF[layer],
                                                 self.spF[layer])
        self.losses['loss_psp'] = self.loss_psp
        '''
        style_mix_loss = self.loss_psp * self.style_patch_alpha + (1-self.style_patch_alpha)*self.loss_ps

        self.p_loss_style_remd = self.calc_style_emd_loss(
            self.tpF['r31'], self.spF['r31']) + self.calc_style_emd_loss(
            self.tpF['r41'], self.spF['r41'])
        self.p_loss_content_relt = self.calc_content_relt_loss(
            self.tpF['r31'], self.cpF['r31']) + self.calc_content_relt_loss(
            self.tpF['r41'], self.cpF['r41'])
        self.losses['p_loss_style_remd'] = self.p_loss_style_remd
        self.losses['p_loss_content_relt'] = self.p_loss_content_relt

        """gan loss"""
        pred_fake_p = self.nets['netD_patch'](self.p_stylized)
        self.loss_Gp_GAN = self.gan_criterion(pred_fake_p, True)
        self.losses['loss_gan_Gp'] = self.loss_Gp_GAN

        if self.use_mxdog==1:
            self.cX = xdog(self.cp.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv)
            self.cXF = self.nets['net_enc'](self.cX)
            self.visual_items['cx'] = self.cX
            stylized_dog = xdog(self.p_stylized,self.gaussian_filter,self.gaussian_filter_2,self.morph_conv)
            self.cdogF = self.nets['net_enc'](stylized_dog)
            mxdog_content = self.calc_content_loss(self.tpF['r31'], self.cXF['r31'])
            mxdog_content_contraint = self.calc_content_loss(self.cdogF['r31'], self.cXF['r31'])
            mxdog_content_img = self.calc_style_loss(self.cdogF['r31'],self.sXF['r31'])

            self.losses['loss_MD'] = mxdog_content*.05
            self.losses['loss_CnsC'] = mxdog_content_contraint*100
            self.losses['loss_CnsS'] = mxdog_content_img*500
            mxdogloss=mxdog_content * .0125 + mxdog_content_contraint *25 + mxdog_content_img * 125
        else:
            mxdogloss=0

        self.loss = self.loss_Gp_GAN * 2 +style_mix_loss * self.style_weight +\
                          self.loss_content_p * self.content_weight +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight * 20 +\
                    self.p_loss_style_remd * 26 + self.p_loss_content_relt * 26 + mxdogloss
        self.loss.backward()

        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)

        pred_real = self.nets['netD'](self.pyr_si[2])
        self.loss_D_real = self.gan_criterion(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real


    def backward_Dpatch(self):
        """Calculate GAN loss for the patch discriminator"""
        pred_p_fake = self.nets['netD_patch'](self.p_stylized.detach())
        self.loss_Dp_fake = self.gan_criterion(pred_p_fake, False)

        pred_Dp_real = 0
        reshaped = paddle.slice(self.sp, axes=[2, 3], starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
        self.loss_Dp_real = self.nets['netD_patch'](reshaped)
        pred_Dp_real += self.gan_criterion(self.loss_Dp_real, True)
        self.loss_D_patch = (self.loss_Dp_fake + pred_Dp_real) * 0.5

        self.loss_D_patch.backward()

        self.losses['Dp_fake_loss'] = self.loss_Dp_fake
        self.losses['Dp_real_loss'] = pred_Dp_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D
        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()

        self.set_requires_grad(self.nets['netD_patch'], True)
        optimizers['optimD_patch'].clear_grad()
        self.backward_Dpatch()
        optimizers['optimD_patch'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        self.set_requires_grad(self.nets['netD_patch'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G(optimizers['optimG'])
        optimizers['optimG'].step()

@MODELS.register()
class LapStyleRevSecondThumb(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevSecondThumb, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)
        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define the first revnet params
        self.nets['net_rev'] = build_generator(revnet_generator)
        self.set_requires_grad([self.nets['net_rev']], False)

        # define the second revnet params
        self.nets['net_rev_2'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev_2'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.sp = paddle.to_tensor(input['sp'])
        self.image_paths = input['ci_path']
        self.cp = paddle.to_tensor(input['cp'])
        self.visual_items['cp'] = self.cp
        self.position = input['position']

        self.pyr_ci = make_laplace_pyramid(self.ci, 2)
        self.pyr_si = make_laplace_pyramid(self.si, 2)
        self.pyr_cp = make_laplace_pyramid(self.cp, 2)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)
        self.pyr_cp.append(self.cp)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[2])
        sF = self.nets['net_enc'](self.pyr_si[2])
        cpF = self.nets['net_enc'](self.pyr_cp[2])
        self.spCrop = self.nets['net_enc'](self.sp)

        stylized_small, _ = self.nets['net_dec'](cF, sF, cpF, 'thumb')
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[1], stylized_up], axis=1)
        #rev_net thumb only calcs as patch if second parameter is passed
        stylized_rev_lap, self.stylized_thumb_feat = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
        self.visual_items['stylized_rev_first'] = stylized_rev
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap_second, self.stylized_thumb_large = self.nets['net_rev_2'](revnet_input)
        stylized_rev_second = fold_laplace_pyramid(
            [stylized_rev_lap_second, stylized_rev_lap, stylized_small])

        self.stylized = stylized_rev_second
        self.visual_items['stylized'] = self.stylized

        stylized_small, _ = self.nets['net_dec'](cF, sF, cpF, 'patch')
        self.visual_items['p_stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_cp[1], stylized_up], axis=1)
        stylized_rev_lap, _ = self.nets['net_rev'](revnet_input, self.stylized_thumb_feat)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
        self.visual_items['p_stylized_rev_first'] = stylized_rev
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_cp[0], stylized_up], axis=1)
        stylized_rev_lap_second,_ = self.nets['net_rev_2'](revnet_input,self.stylized_thumb_large)
        stylized_rev_second = fold_laplace_pyramid(
            [stylized_rev_lap_second, stylized_rev_lap, stylized_small])

        self.p_stylized = stylized_rev_second
        self.visual_items['p_stylized'] = self.p_stylized

    def backward_G(self):
        cF = self.nets['net_enc'](self.ci)
        sF = self.nets['net_enc'](self.si)
        ttF = self.nets['net_enc'](self.stylized)

        loss_content = 0
        for layer in self.content_layers:
            loss_content += self.calc_content_loss(ttF[layer],
                                                        cF[layer],
                                                        norm=True)
        self.losses['loss_content'] = loss_content

        """style loss"""
        loss_s = 0
        for layer in self.style_layers:
            loss_s += self.calc_style_loss(ttF[layer], sF[layer])
        self.losses['loss_s'] = loss_s

        """relative loss"""
        loss_style_remd = self.calc_style_emd_loss(
            ttF['r31'], sF['r31']) + self.calc_style_emd_loss(
            ttF['r41'], sF['r41'])
        loss_content_relt = self.calc_content_relt_loss(
            ttF['r31'], cF['r31']) + self.calc_content_relt_loss(
            ttF['r41'], cF['r41'])
        self.losses['loss_style_remd'] = loss_style_remd
        self.losses['loss_content_relt'] = loss_content_relt

        pred_fake = self.nets['netD'](self.stylized)
        loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = loss_G_GAN

        loss = loss_G_GAN + loss_s * self.style_weight + \
                    loss_content * self.content_weight + \
                    loss_style_remd * 20 + \
                    loss_content_relt * 24
        loss.backward()
        return loss

    def backward_G_p(self):
        spCrop = self.nets['net_enc'](self.sp)
        tpF = self.nets['net_enc'](self.p_stylized)
        cpF = self.nets['net_enc'](self.cp)
        with paddle.no_grad():
            g_t_thumb_up = F.interpolate(self.visual_items['stylized'], scale_factor=2, mode='bilinear',
                                         align_corners=False)
            g_t_thumb_crop = paddle.slice(g_t_thumb_up, axes=[2, 3], starts=[self.position[0], self.position[2]],
                                          ends=[self.position[1], self.position[3]])
            tt_cropF = self.nets['net_enc'](g_t_thumb_crop)
        """patch loss"""
        loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[3]]:
            loss_patch += self.calc_content_loss(tpF[layer],
                                                      tt_cropF[layer])
        self.losses['loss_patch'] = loss_patch

        loss_content_p = 0
        for layer in self.content_layers:
            loss_content_p += self.calc_content_loss(tpF[layer],
                                                          cpF[layer],
                                                          norm=True)
        self.losses['loss_content_p'] = loss_content_p

        loss_ps = 0
        for layer in self.style_layers:
            loss_ps += self.calc_style_loss(tpF[layer], spCrop[layer])
        self.losses['loss_ps'] = loss_ps

        p_loss_style_remd = self.calc_style_emd_loss(
            tpF['r31'], tt_cropF['r31']) + self.calc_style_emd_loss(
            tpF['r41'], tt_cropF['r41'])
        p_loss_content_relt = self.calc_content_relt_loss(
            tpF['r31'], cpF['r31']) + self.calc_content_relt_loss(
            tpF['r41'], cpF['r41'])
        self.losses['p_loss_style_remd'] = p_loss_style_remd
        self.losses['p_loss_content_relt'] = p_loss_content_relt

        patch_loss = loss_ps * self.style_weight + \
                          loss_content_p * self.content_weight + \
                          loss_patch * self.content_weight + \
                          p_loss_style_remd * 20 + p_loss_content_relt * 24
        patch_loss.backward()
        return patch_loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_p_fake = self.nets['netD'](self.p_stylized.detach())
        loss_Dp_fake = self.gan_criterion(pred_p_fake, False)

        pred_real = self.nets['netD'](self.pyr_si[3])
        loss_D_real = self.gan_criterion(pred_real, True)
        pred_p_real = self.nets['netD'](self.sp)
        loss_Dp_real = self.gan_criterion(pred_p_real, True)
        self.loss_D = (loss_D_fake + loss_Dp_fake + loss_Dp_real + loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = loss_D_fake
        self.losses['D_real_loss'] = loss_D_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D

        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G()
        optimizers['optimG'].step()
        self.set_requires_grad(self.nets['netD'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G_p()
        optimizers['optimG'].step()

def crop_upsized(stylized_up,positions,orig_size):
    ratio = orig_size/512
    stylized_up=paddle.slice(stylized_up,axes=[2,3],starts=[(positions[1]/ratio).astype('int32'),(positions[0]/ratio).astype('int32')],\
                             ends=[(positions[3]/ratio).astype('int32'),(positions[2]/ratio).astype('int32')])
    return stylized_up

def adjust(inp,size):
    num=math.ceil(inp/size)+1
    move=math.floor(inp/(num+1))
    return move

def positions_list(axis,length):
    def coords(axis,start):
        if axis=='x':
            return [0,start,start+256,256]
        else:
            return [start,0,256,start+256]
    length = length

    move=adjust(length,256)
    for i in range(0,length-move,move):
        for j in range(0,256-64,64):
            for k in range(0,256-64,64):
                #rev 1
                for l in range(0,192,64):
                    for m in range(0,192,64):
                        #rev 2
                        for n in range(0,192,64):
                            for o in range(0,192,64):
                                curr_=[n+m+k+i,o+l+j] if axis=='x' else [n+m+k,i+o+l+j]
                                lol=[coords(axis,i),
                                    [k,j,j+128,k+128],
                                    [m,l,l+128,m+128],
                                    [n,o,o+128,n+128],
                                    curr_]
                                yield lol


@MODELS.register()
class LapStyleRevSecondPatch(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 revnet_deep_generator,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0,
                 ada_alpha=1.0,
                 ada_alpha_2=1.0,
                 gan_thumb_weight=1.0,
                 gan_patch_weight=1.0):

        super(LapStyleRevSecondPatch, self).__init__()

        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)
        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define the first revnet params
        self.nets['net_rev'] = build_generator(revnet_generator)
        self.set_requires_grad([self.nets['net_rev']], False)

        # define the second revnet params
        self.nets['net_rev_2'] = build_generator(revnet_deep_generator)
        init_weights(self.nets['net_rev_2'])
        self.nets['net_rev_3'] = build_generator(revnet_deep_generator)
        init_weights(self.nets['net_rev_3'])
        self.nets['net_rev_4'] = build_generator(revnet_deep_generator)
        init_weights(self.nets['net_rev_4'])
        if self.is_train:
            self.nets['netD'] = build_discriminator(revnet_discriminator)
            init_weights(self.nets['netD'])
            self.nets['netD_patch'] = build_discriminator(revnet_discriminator)
            init_weights(self.nets['netD_patch'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.ada_alpha = ada_alpha
        self.ada_alpha_2 = ada_alpha_2
        self.gan_thumb_weight = gan_thumb_weight
        self.gan_patch_weight = gan_patch_weight

        l = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_kernel = paddle.nn.Conv2D(3, 3, (3, 3), stride=1, bias_attr=False,
                                           padding=1, groups=3, padding_mode='reflect',
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                   value=l), trainable=False)
                                           )

    def test_iter(self, output_dir=None,metrics=None):
        self.eval()
        self.output_dir=output_dir
        self.laplacians=[laplacian_conv(self.content_stack[0],self.lap_kernel)]
        self.out_images=[]
        print('content_size='+str(self.content.shape))
        for i in [.25,.5,1]:
            if i==1:
                self.laplacians.append(laplacian_conv(self.content,self.lap_kernel))
            else:
                self.laplacians.append(laplacian_conv(F.interpolate(self.content, scale_factor=i),self.lap_kernel))
        with paddle.no_grad():
            cF = self.nets['net_enc'](F.interpolate(self.content_stack[0],scale_factor=.5))
            sF = self.nets['net_enc'](F.interpolate(self.style_stack[0], scale_factor=.5))

            stylized_small= self.nets['net_dec'](cF, sF)
            self.stylized_up = F.interpolate(stylized_small, scale_factor=2)
            revnet_input = paddle.concat(x=[self.laplacians[0], self.stylized_up], axis=1)
            # rev_net thumb only calcs as patch if second parameter is passed
            stylized_rev_lap, self.stylized_feats = self.nets['net_rev'](revnet_input)
            stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
            self.stylized_slice = F.interpolate(stylized_rev, scale_factor=2)
            print('stylized_slice.shape = '+str(self.stylized_slice.shape))
            size_x = self.stylized_slice.shape[-2]
            self.in_size_x = math.floor(size_x / 2)
            move_x = adjust(size_x, self.in_size_x)
            ranges_x=list(range(0,size_x,self.in_size_x))
            size_y = self.stylized_slice.shape[-1]
            self.in_size_y = math.floor(size_y / 2)
            move_y = adjust(size_y, self.in_size_y)
            ranges_y = list(range(0,size_y,self.in_size_y))
            orig_len_y = len(ranges_y)
            orig_len_x = len(ranges_x)
            curr_last_x=ranges_x[-1]
            curr_last_y=ranges_y[-1]
            ranges_x = ranges_x + [i+math.floor(self.in_size_x/3) for i in ranges_x[:-1]]
            ranges_y = ranges_y + [i+math.floor(self.in_size_y/3) for i in ranges_y[:-1]]
            ranges_x.append(curr_last_x-math.floor(self.in_size_x/3))
            ranges_y.append(curr_last_y-math.floor(self.in_size_y/3))
            self.counter=1
            self.stylized_feats = self.nets['net_rev_2'].DownBlock(revnet_input.detach())
            self.stylized_feats = self.nets['net_rev_2'].resblock(self.stylized_feats)
            for idx,i in enumerate(ranges_x):
                self.counter+=1
                for idx2,j in enumerate(ranges_y):
                    self.second_set = 'b' if idx+1>orig_len_x or idx2+1>orig_len_y else 'a'
                    self.outer_loop=(i,j)
                    self.positions=[[i,j,i+self.in_size_x,j+self.in_size_y]]#!
                    self.test_forward(self.stylized_slice,self.stylized_feats)
            positions = [(int(re.split('_|\.',i)[0]),int(re.split('_|\.',i)[1])) for i in self.labels]
            set_letter = [re.split('_|\.',i)[2] for i in self.labels]
            max_x = 0
            max_y = 0
            for a,b in positions:
                if a>max_x:
                    max_x=a
                if b>max_y:
                    max_y=b
            max_x = max_x+self.in_size_x
            max_y = max_y+self.in_size_y
            print('max_x = '+str(max_x))
            print('max_y = ' + str(max_y))
            tiles_1 = np.zeros((max_x,max_y,3), dtype=np.uint8)
            weights = np.zeros((max_x, max_y), dtype=np.uint8)
            not_visited = np.empty((max_x,max_y))
            not_visited[:,:]=np.nan
            kernel = np.ones((self.in_size_x-64,self.in_size_y-64))
            kernel = np.pad(kernel,(32,32),'linear_ramp', end_values=(0, 0))
            #tiles_2 = np.zeros((max_x, max_y,3), dtype=np.uint8)
            for image,b,c in zip(self.out_images,positions,set_letter):
                empty = np.isnan(not_visited[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1]])
                k = kernel.copy()
                k = np.maximum(k,empty)
                w = weights[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1]]+k
                tiles_1[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1],0] = (image[:,:,0]*k +\
                                                                               (tiles_1[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1],0]*(w-k)))/w
                tiles_1[b[0]:b[0] + image.shape[0], b[1]:b[1] + image.shape[1], 1] = (image[:,:,1]*k +\
                                                                               (tiles_1[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1],1]*(w-k)))/w
                tiles_1[b[0]:b[0] + image.shape[0], b[1]:b[1] + image.shape[1], 2] = (image[:,:,2]*k +\
                                                                               (tiles_1[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1],2]*(w-k)))/w
                not_visited[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1]]=1
                weights[b[0]:b[0] + image.shape[0], b[1]:b[1] + image.shape[1]] = weights[b[0]:b[0] + image.shape[0], b[1]:b[1] + image.shape[1]]+w
            for a,b in zip([tiles_1],['tiled1']):
                im = Image.fromarray(a,'RGB')
                label = self.path[0]+' '+b
                makedirs(os.path.join(self.output_dir, 'visual_test'))
                img_path = os.path.join(self.output_dir, 'visual_test',
                                        '%s.png' % (label))
                im.save(img_path)
            shutil.rmtree(os.path.join(self.output_dir, 'visual_test','tiles'))
            self.paths=[]
        self.train()

    def setup_input(self, input):
        if self.is_train:
            self.content_stack = []
            self.style_stack = [paddle.to_tensor(input['style_stack_1']),paddle.to_tensor(input['style_stack_2']),paddle.to_tensor(input['style_stack_3'])]
            self.laplacians=[]
            for i in range(1,6):
                if 'content_stack_'+str(i) in input:
                    self.content_stack.append(paddle.to_tensor(input['content_stack_'+str(i)]))
            self.visual_items['ci'] = self.content_stack[0]

            self.positions = input['position_stack']
            self.size_stack = input['size_stack']
            self.laplacians.append(laplacian(self.content_stack[0]).detach())
            self.laplacians.append(laplacian(self.content_stack[1]).detach())
            self.laplacians.append(laplacian(self.content_stack[2]).detach())
            self.laplacians.append(laplacian(self.content_stack[3]).detach())
        else:
            self.labels=[]
            self.content_stack=[input['ci']]
            self.content=input['content']
            self.style_stack = [input['si']]
            self.path=input['ci_path']
    def test_forward(self,stylized_slice,stylized_feats):
        stylized_up = paddle.slice(stylized_slice,axes=[2,3],starts=[self.positions[0][0],self.positions[0][1]],\
                             ends=[self.positions[0][2],self.positions[0][3]])
        lap = paddle.slice(self.laplacians[1],axes=[2,3],starts=[self.positions[0][0],self.positions[0][1]],\
                             ends=[self.positions[0][2],self.positions[0][3]])
        revnet_input = paddle.concat(x=[lap, stylized_up], axis=1)
        stylized_rev_lap_second,stylized_feats = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats,self.ada_alpha)
        stylized_rev_second = fold_laplace_pyramid([stylized_rev_lap_second, stylized_up])
        stylized_up = F.interpolate(stylized_rev_second, scale_factor=2)
        size_x = stylized_up.shape[-2]
        in_size_x = math.floor(size_x / 2)
        move_x = adjust(size_x, in_size_x)
        size_y = stylized_up.shape[-1]
        in_size_y = math.floor(size_y / 2)
        move_y = adjust(size_y, in_size_y)
        for i in range(0,size_x,self.in_size_x):
            for j in range(0,size_y,self.in_size_y):
                label = str(self.outer_loop[0]*4+i*2)+'_'+str(self.outer_loop[1]*4+j*2)+'_'+self.second_set
                if label in self.labels:
                    notin=True
                    for k in range(0,size_x,move_x):
                        for l in range(0,size_y,move_y):
                            label = str(self.outer_loop[0]*4+i*2+k)+'_'+str(self.outer_loop[1]*4+j*2+l)
                            if not label in self.labels:
                                notin=False
                    if notin:
                        continue
                stylized_up_2 = paddle.slice(stylized_up,axes=[2,3],starts=[i,j],\
                             ends=[i+in_size_x,j+in_size_y])
                self.first_patch_in = stylized_up_2.detach()

                lap_2 = paddle.slice(self.laplacians[2],axes=[2,3],starts=[self.outer_loop[0]*2+i,self.outer_loop[1]*2+j],
                                   ends=[self.outer_loop[0]*2+i+in_size_x,self.outer_loop[1]*2+j+in_size_y])
                if lap_2.shape[-2]!=in_size_x or lap_2.shape[-1]!=in_size_y:
                    print('continue, line 1311')
                    continue
                if stylized_up_2.shape[-2]!=in_size_x or stylized_up_2.shape[-1]!=in_size_y:
                    print('continue, line 1314')
                    continue
                revnet_input_2 = paddle.concat(x=[lap_2, stylized_up_2.detach()], axis=1)
                stylized_feats = self.nets['net_rev_3'].DownBlock(revnet_input.detach())
                stylized_feats = self.nets['net_rev_3'].resblock(stylized_feats)
                stylized_rev_patch,stylized_feats = self.nets['net_rev_3'](revnet_input_2.detach(),stylized_feats.detach(),self.ada_alpha_2)
                stylized_rev_patch = fold_laplace_patch(
                    [stylized_rev_patch, stylized_up_2.detach()])

                stylized_up_3 = F.interpolate(stylized_rev_patch, scale_factor=2)
                for k in range(0,size_x,self.in_size_x):
                    for l in range(0,size_y,self.in_size_y):
                        label = str(self.outer_loop[0]*4+i*2+k)+'_'+str(self.outer_loop[1]*4+j*2+l)+'_'+str(self.counter)
                        if label in self.labels:
                            continue
                        if k+in_size_x>stylized_up_3.shape[-2] or l+in_size_y>stylized_up_3.shape[-1]:
                            print('continue, line 1331')
                            continue
                        stylized_up_4 = paddle.slice(stylized_up_3,axes=[2,3],starts=[k,l],\
                             ends=[k+in_size_x,l+in_size_y])
                        lap_3 = paddle.slice(self.laplacians[3],axes=[2,3],starts=[self.outer_loop[0]*4+i*2+k,self.outer_loop[1]*4+j*2+l*1],
                                   ends=[self.outer_loop[0]*4+i*2+k+in_size_x,self.outer_loop[1]*4+j*2+l+in_size_y])
                        if lap_3.shape[-2]!=in_size_x or lap_3.shape[-1]!=in_size_y:
                            print('continue, line 1338')
                            continue
                        if stylized_up_4.shape[-2]!=in_size_x or stylized_up_4.shape[-1]!=in_size_y:
                            print('continue, line 1341')
                            continue
                        revnet_input_3 = paddle.concat(x=[lap_3, stylized_up_4.detach()], axis=1)
                        stylized_feats = self.nets['net_rev_4'].DownBlock(revnet_input_2.detach())
                        stylized_feats = self.nets['net_rev_4'].resblock(stylized_feats)
                        stylized_rev_patch_second,_ = self.nets['net_rev_4'](revnet_input_3.detach(),stylized_feats.detach(),self.ada_alpha_2)
                        stylized_rev_patch_second = fold_laplace_patch(
                            [stylized_rev_patch_second, stylized_up_4.detach()])
                        image_numpy=tensor2img(stylized_rev_patch_second,min_max=(0., 1.))
                        makedirs(os.path.join(self.output_dir, 'visual_test','tiles'))
                        img_path = os.path.join(self.output_dir, 'visual_test','tiles',
                                                '%s.png' % (label))
                        self.out_images.append(image_numpy)
                        self.labels.append(label)
                        self.counter+=1
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.is_train:
            self.cF = self.nets['net_enc'](F.interpolate(self.content_stack[0],scale_factor=.5))
            self.sF = self.nets['net_enc'](F.interpolate(self.style_stack[0], scale_factor=.5))

            stylized_small= self.nets['net_dec'](self.cF, self.sF)
            self.visual_items['stylized_small'] = stylized_small
            stylized_up = F.interpolate(stylized_small, scale_factor=2)

            revnet_input = paddle.concat(x=[self.laplacians[0], stylized_up], axis=1)
            #rev_net thumb only calcs as patch if second parameter is passed
            stylized_rev_lap,stylized_feats = self.nets['net_rev'](revnet_input)
            stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
            self.visual_items['stylized_rev_first'] = stylized_rev
            stylized_up = F.interpolate(stylized_rev, scale_factor=2)
        else:
            stylized_up = self.stylized_up
            stylized_feats = self.stylized_feats
        stylized_up = crop_upsized(stylized_up,self.positions[0],self.size_stack[0])
        revnet_input = paddle.concat(x=[self.laplacians[1], stylized_up], axis=1)
        stylized_rev_lap_second,stylized_feats = self.nets['net_rev'](revnet_input.detach(),stylized_feats,self.ada_alpha)
        stylized_rev_second = fold_laplace_pyramid([stylized_rev_lap_second, stylized_up])
        self.visual_items['ci_2'] = self.content_stack[1]
        self.stylized= stylized_rev_second

        self.visual_items['stylized_rev_second'] = stylized_rev_second

        stylized_up = F.interpolate(stylized_rev_second, scale_factor=2)
        stylized_up = crop_upsized(stylized_up,self.positions[1],self.size_stack[1])
        self.first_patch_in = stylized_up.detach()

        stylized_feats = self.nets['net_rev_2'].DownBlock(revnet_input.detach())
        stylized_feats = self.nets['net_rev_2'].resblock(stylized_feats)

        revnet_input = paddle.concat(x=[self.laplacians[2], stylized_up.detach()], axis=1)
        stylized_rev_patch,stylized_feats = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats.detach(),self.ada_alpha_2)
        stylized_rev_patch = fold_laplace_patch(
            [stylized_rev_patch, stylized_up.detach()])
        self.visual_items['ci_3'] = self.content_stack[2]
        self.visual_items['stylized_rev_third'] = stylized_rev_patch

        stylized_up = F.interpolate(stylized_rev_patch, scale_factor=2)
        stylized_up = crop_upsized(stylized_up,self.positions[2],self.size_stack[2])
        self.second_patch_in = stylized_up.detach()

        revnet_input = paddle.concat(x=[self.laplacians[3], stylized_up.detach()], axis=1)
        stylized_rev_patch_second,_ = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats.detach(),self.ada_alpha_2)
        stylized_rev_patch_second = fold_laplace_patch(
            [stylized_rev_patch_second, stylized_up.detach()])
        self.visual_items['ci_4'] = self.content_stack[3]
        self.visual_items['stylized_rev_fourth'] = stylized_rev_patch_second

        self.stylized = stylized_rev_patch
        self.p_stylized = stylized_rev_patch_second

    def backward_G(self):
        self.cF = self.nets['net_enc'](self.content_stack[-2])

        with paddle.no_grad():
            self.tt_cropF = self.nets['net_enc'](self.first_patch_in)

        self.tpF = self.nets['net_enc'](self.stylized)

        """patch loss"""
        self.loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            self.loss_patch += paddle.clip(self.calc_content_loss(self.tpF[layer],
                                                      self.tt_cropF[layer]), 1e-5, 1e5)
        self.losses['loss_patch'] = self.loss_patch

        self.loss_content_p = 0
        for layer in self.content_layers[:-1]:
            self.loss_content_p += paddle.clip(self.calc_content_loss(self.tpF[layer],
                                                      self.cF[layer],
                                                      norm=True), 1e-5, 1e5)
        self.losses['loss_content_p'] = self.loss_content_p

        self.loss_ps = 0
        self.p_loss_style_remd = 0

        reshaped = paddle.split(self.style_stack[2], 2, 2)
        for i in reshaped:
            for j in paddle.split(i, 2, 3):
                spF = self.nets['net_enc'](j.detach())
                for layer in self.content_layers[:-1]:
                    self.loss_ps += paddle.clip(self.calc_style_loss(self.tpF[layer],
                                                          spF[layer]), 1e-5, 1e5)
                self.p_loss_style_remd += self.calc_style_emd_loss(
                    self.tpF['r31'], spF['r31']) + self.calc_style_emd_loss(
                    self.tpF['r41'], spF['r41'])
        self.losses['loss_ps'] = self.loss_ps
        self.p_loss_content_relt = self.calc_content_relt_loss(
            self.tpF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
            self.tpF['r41'], self.cF['r41'])
        self.p_loss_style_remd = paddle.clip(self.p_loss_style_remd, 1e-5, 1e5)
        self.p_loss_content_relt = paddle.clip(self.p_loss_content_relt, 1e-5, 1e5)
        self.losses['p_loss_style_remd'] = self.p_loss_style_remd
        self.losses['p_loss_content_relt'] = self.p_loss_content_relt

        """gan loss"""
        pred_fake_p = self.nets['netD'](self.stylized)
        self.loss_Gp_GAN = paddle.clip(self.gan_criterion(pred_fake_p, True), 1e-5, 1e5)
        self.losses['loss_gan_Gp'] = self.loss_Gp_GAN


        self.loss = self.loss_Gp_GAN *self.gan_thumb_weight +self.loss_ps/4 * self.style_weight +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight +\
                    self.p_loss_style_remd/4 * 18 + self.p_loss_content_relt * 18
        self.loss.backward()

        return self.loss

    def backward_G_p(self):
        cF = self.nets['net_enc'](self.content_stack[-1])

        with paddle.no_grad():
            tt_cropF = self.nets['net_enc'](self.second_patch_in)

        tpF = self.nets['net_enc'](self.p_stylized)

        """patch loss"""
        loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            loss_patch += paddle.clip(self.calc_content_loss(tpF[layer],
                                                      tt_cropF[layer]), 1e-5, 1e5)
        self.losses['loss_patch2'] = loss_patch

        loss_content_p = 0
        for layer in self.content_layers[:-1]:
            loss_content_p += paddle.clip(self.calc_content_loss(tpF[layer],
                                                      cF[layer],
                                                      norm=True), 1e-5, 1e5)
        self.losses['loss_content_p2'] = loss_content_p

        loss_ps = 0
        p_loss_style_remd = 0
        reshaped = paddle.split(self.style_stack[1], 2, 2)
        for i in reshaped:
            for j in paddle.split(i, 2, 3):
                spF = self.nets['net_enc'](j.detach())
                for layer in self.content_layers[:-1]:
                    loss_ps += paddle.clip(self.calc_style_loss(tpF[layer],
                                                          spF[layer]), 1e-5, 1e5)
                p_loss_style_remd += self.calc_style_emd_loss(
                    tpF['r31'], spF['r31']) + self.calc_style_emd_loss(
                    tpF['r41'], spF['r41'])
        self.losses['loss_ps2'] = loss_ps
        p_loss_content_relt = self.calc_content_relt_loss(
            tpF['r31'], cF['r31']) + self.calc_content_relt_loss(
            tpF['r41'], cF['r41'])
        p_loss_style_remd = paddle.clip(p_loss_style_remd, 1e-5, 1e5)
        p_loss_content_relt = paddle.clip(p_loss_content_relt, 1e-5, 1e5)
        self.losses['p_loss_style_remd2'] = self.p_loss_style_remd
        self.losses['p_loss_content_relt2'] = self.p_loss_content_relt

        """gan loss"""
        pred_fake_p = self.nets['netD_patch'](self.p_stylized)
        loss_Gp_GAN = paddle.clip(self.gan_criterion(pred_fake_p, True), 1e-5, 1e5)
        self.losses['loss_gan_Gp2'] = loss_Gp_GAN


        loss_patch = loss_Gp_GAN * self.gan_patch_weight +loss_ps/4 * self.style_weight +\
                    loss_content_p * self.content_weight*4 +\
                    loss_patch * self.content_weight * 4+\
                    p_loss_style_remd/4 *26 + p_loss_content_relt * 26
        loss_patch.backward()

        return loss_patch

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_p_fake = self.nets['netD'](self.stylized.detach())
        self.loss_Dp_fake = paddle.clip(self.gan_criterion(pred_p_fake, False), 1e-5, 1e5)

        pred_Dp_real = 0
        reshaped = paddle.split(self.style_stack[2], 2, 2)
        for i in reshaped:
            for j in paddle.split(i, 2, 3):
                self.loss_Dp_real = self.nets['netD'](j.detach())
                pred_Dp_real += paddle.clip(self.gan_criterion(self.loss_Dp_real, True), 1e-5, 1e5)
        self.loss_D_patch = (self.loss_Dp_fake + pred_Dp_real/4) * 0.5

        self.loss_D_patch.backward()

        self.losses['D_fake_loss'] = self.loss_Dp_fake
        self.losses['D_real_loss'] = pred_Dp_real

    def backward_Dpatch(self):
        """Calculate GAN loss for the discriminator"""
        pred_p_fake = self.nets['netD_patch'](self.p_stylized.detach())
        self.loss_Dp_fake = paddle.clip(self.gan_criterion(pred_p_fake, False), 1e-5, 1e5)

        pred_Dp_real = 0
        reshaped = paddle.split(self.style_stack[1], 2, 2)
        for i in reshaped:
            for j in paddle.split(i, 2, 3):
                self.loss_Dp_real = self.nets['netD_patch'](j.detach())
                pred_Dp_real += paddle.clip(self.gan_criterion(self.loss_Dp_real, True), 1e-5, 1e5)
        self.loss_D_patch = (self.loss_Dp_fake + pred_Dp_real/4) * 0.5

        self.loss_D_patch.backward()

        self.losses['Dp_fake_loss'] = self.loss_Dp_fake
        self.losses['Dp_real_loss'] = pred_Dp_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D
        self.set_requires_grad(self.nets['netD'], True)
        self.set_requires_grad(self.nets['netD_patch'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()
        self.set_requires_grad(self.nets['netD_patch'], True)
        optimizers['optimD_patch'].clear_grad()
        self.backward_Dpatch()
        optimizers['optimD_patch'].step()

        # update G

        self.set_requires_grad(self.nets['netD_patch'], False)
        self.set_requires_grad(self.nets['netD'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G()
        optimizers['optimG'].step()
        optimizers['optimG'].clear_grad()
        self.backward_G_p()
        optimizers['optimG'].step()

@MODELS.register()
class LapStyleRevSecondMXDOG(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator_1,
                 revnet_discriminator_2,
                 revnet_discriminator_3,
                 revnet_discriminator_4,
                 draftnet_encode,
                 draftnet_decode,
                 revnet_deep_generator,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 mse_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0,
                 train_layer=1,
                 ada_alpha=1.0,
                 ada_alpha_2=1.0,
                 gan_thumb_weight=1.0,
                 gan_patch_weight=1.0,
                 use_mdog=0,
                 morph_cutoff=47.9):

        super(LapStyleRevSecondMXDOG, self).__init__()

        self.train_layer=train_layer
        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)
        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_dec']], False)
        #init_weights(self.nets['net_dec'])
        print(train_layer)
        if train_layer>0:
            # define the first revnet params
            self.nets['net_rev'] = build_generator(revnet_generator)
            self.nets['netD_1'] = build_discriminator(revnet_discriminator_1)
            self.discriminators=['netD_1']
            self.o = ['optimD1']
            self.go = ['optimG1']
            self.generator = ['net_rev']
            if train_layer>1:
                self.set_requires_grad([self.nets['net_rev']], False)
                self.set_requires_grad([self.nets['netD_1']], False)
            else:
                print('init weights')
                init_weights(self.nets['net_rev'])
                init_weights(self.nets['netD_1'])
        if train_layer>1:
            self.nets['net_rev_2'] = build_generator(revnet_deep_generator)
            self.nets['netD_2'] = build_discriminator(revnet_discriminator_1)
            self.discriminators.append('netD_2')
            self.o.append('optimD2')
            self.generator.append('net_rev_2')
            self.go.append('optimG2')
            if train_layer>2:
                self.set_requires_grad([self.nets['net_rev_2']], False)
                self.set_requires_grad([self.nets['netD_2']], False)
            else:
                init_weights(self.nets['net_rev_2'])
                init_weights(self.nets['netD_2'])
        if train_layer>2:
            self.nets['net_rev_3'] = build_generator(revnet_deep_generator)
            self.nets['netD_3'] = build_discriminator(revnet_discriminator_1)
            self.discriminators.append('netD_3')
            self.o.append('optimD3')
            self.generator.append('net_rev_3')
            self.go.append('optimG3')
            if train_layer>3:
                self.set_requires_grad([self.nets['net_rev_3']], False)
                self.set_requires_grad([self.nets['netD_3']], False)
            else:
                init_weights(self.nets['net_rev_3'])
                init_weights(self.nets['netD_3'])
        if train_layer>3:
            self.nets['net_rev_4'] = build_generator(revnet_deep_generator)
            self.nets['netD_4'] = build_discriminator(revnet_discriminator_1)
            self.discriminators.append('netD_4')
            self.generator.append('net_rev_4')
            self.o.append('optimD4')
            self.go.append('optimG4')
            if train_layer>4:
                self.set_requires_grad([self.nets['net_rev_4']], False)
                self.set_requires_grad([self.nets['netD_4']], False)
            else:
                init_weights(self.nets['net_rev_4'])
                init_weights(self.nets['netD_4'])

        l = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_filter = paddle.nn.Conv2D(3, 3, (3, 3), stride=1, bias_attr=False,
                                           padding=1, groups=3, padding_mode='reflect',
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                   value=l), trainable=False)
                                           )

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.mse_loss = build_criterion(mse_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.ada_alpha = ada_alpha
        self.ada_alpha_2 = ada_alpha_2
        self.gan_thumb_weight = gan_thumb_weight
        self.gan_patch_weight = gan_patch_weight
        self.morph_cutoff = morph_cutoff
        g = np.repeat(gaussian(7, 1).numpy(), 3, axis=0)
        g2 = np.repeat(gaussian(19, 3).numpy(), 3, axis=0)
        self.gaussian_filter = paddle.nn.Conv2D(3, 3, 7,
                                                groups=3, bias_attr=False,
                                                padding=3, padding_mode='reflect',
                                                weight_attr=paddle.ParamAttr(
                                                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                        value=g), trainable=False)
                                                )
        self.gaussian_filter_2 = paddle.nn.Conv2D(3, 3, 19,
                                                  groups=3, bias_attr=False,
                                                  padding=9, padding_mode='reflect',
                                                  weight_attr=paddle.ParamAttr(
                                                      initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                          value=g2), trainable=False)
                                                  )

        self.morph_conv = paddle.nn.Conv2D(3, 3, 3, padding=1, groups=3,
                                           padding_mode='reflect', bias_attr=False,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.Constant(
                                                   value=1), trainable=False)
                                           )
        self.morph_conv_2 = paddle.nn.Conv2D(3, 3, 7, padding=3, groups=3,
                                           padding_mode='reflect', bias_attr=False,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.Constant(
                                                   value=1), trainable=False)
                                           )

    def setup_input(self, input):
        if self.is_train:
            self.content_stack = []
            self.style_stack = [paddle.to_tensor(input['style_stack_1']),paddle.to_tensor(input['style_stack_2'])]
            self.laplacians=[]
            for i in range(1,6):
                if 'content_stack_'+str(i) in input:
                    self.content_stack.append(paddle.to_tensor(input['content_stack_'+str(i)]))
            self.visual_items['ci'] = self.content_stack[0]
            self.visual_items['si'] = self.style_stack[0]

            self.content=input['content']
            self.positions = input['position_stack']
            self.size_stack = input['size_stack']
            if self.train_layer>0:
                self.laplacians.append(laplacian_conv(self.content_stack[0],self.lap_filter).detach())
            if self.train_layer>1:
                self.laplacians.append(laplacian_conv(self.content_stack[1],self.lap_filter).detach())
            if self.train_layer>2:
                self.laplacians.append(laplacian_conv(self.content_stack[2],self.lap_filter).detach())
            if self.train_layer>3:
                self.laplacians.append(laplacian_conv(self.content_stack[3],self.lap_filter).detach())
            self.cX = False
            self.sX = False

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        cF = self.nets['net_enc'](F.interpolate(self.content_stack[0],scale_factor=.5))
        sF = self.nets['net_enc'](F.interpolate(self.style_stack[0], scale_factor=.5))

        stylized_small= self.nets['net_dec'](cF, sF)
        self.stylized=[stylized_small]
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.laplacians[0], stylized_up.detach()], axis=1)
        #rev_net thumb only calcs as patch if second parameter is passed
        stylized_rev_lap,stylized_feats = self.nets['net_rev'](revnet_input.detach())
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small.detach()])
        self.stylized.append(stylized_rev)
        self.visual_items['stylized_rev_first'] = stylized_rev

        if self.train_layer>1:
            stylized_up = F.interpolate(stylized_rev, scale_factor=2)
            stylized_up = crop_upsized(stylized_up,self.positions[0],self.size_stack[0])
            self.patches_in = [stylized_up.detach()]
            stylized_feats = self.nets['net_rev_2'].DownBlock(revnet_input.detach())
            stylized_feats = self.nets['net_rev_2'].resblock(stylized_feats)
            revnet_input = paddle.concat(x=[self.laplacians[1].detach(), stylized_up.detach()], axis=1)
            stylized_rev_lap_second,stylized_feats = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats,self.ada_alpha)
            stylized_rev_second = fold_laplace_pyramid([stylized_rev_lap_second, stylized_up.detach()])
            self.visual_items['ci_2'] = self.content_stack[1]
            self.stylized.append(stylized_rev_second)

            self.visual_items['stylized_rev_second'] = stylized_rev_second
        if self.train_layer>2:
            stylized_up = F.interpolate(stylized_rev_second, scale_factor=2)
            stylized_up = crop_upsized(stylized_up,self.positions[1],self.size_stack[1])
            self.patches_in.append(stylized_up.detach())

            stylized_feats = self.nets['net_rev_3'].DownBlock(revnet_input.detach())
            stylized_feats = self.nets['net_rev_3'].resblock(stylized_feats)

            revnet_input = paddle.concat(x=[self.laplacians[2], stylized_up], axis=1)
            stylized_rev_patch,stylized_feats = self.nets['net_rev_3'](revnet_input.detach(),stylized_feats,self.ada_alpha_2)
            stylized_rev_patch = fold_laplace_patch(
                [stylized_rev_patch, stylized_up.detach()])
            self.visual_items['ci_3'] = self.content_stack[2]
            self.visual_items['stylized_rev_third'] = stylized_rev_patch
            self.stylized.append(stylized_rev_patch)
        if self.train_layer>3:
            stylized_up = F.interpolate(stylized_rev_patch, scale_factor=2)
            stylized_up = crop_upsized(stylized_up,self.positions[2],self.size_stack[2])
            self.patches_in.append(stylized_up.detach())

            stylized_feats = self.nets['net_rev_4'].DownBlock(revnet_input.detach())
            stylized_feats = self.nets['net_rev_4'].resblock(stylized_feats)

            revnet_input = paddle.concat(x=[self.laplacians[3], stylized_up], axis=1)
            stylized_rev_patch_second,_ = self.nets['net_rev_4'](revnet_input.detach(),stylized_feats,self.ada_alpha_2)
            stylized_rev_patch_second = fold_laplace_patch(
                [stylized_rev_patch_second, stylized_up.detach()])
            self.visual_items['ci_4'] = self.content_stack[3]
            self.visual_items['stylized_rev_fourth'] = stylized_rev_patch_second

            self.stylized.append(stylized_rev_patch_second)

    def backward_G(self,i):
        cF = self.nets['net_enc'](self.content_stack[i].detach())
        if i>2:
            style_conv = self.morph_conv_2
            morph_cutoff= 11*.9445
            morph_num=2
        else:
            style_conv = self.morph_conv
            morph_cutoff= 8.5
            morph_num=2
        tpF = self.nets['net_enc'](self.stylized[i+1]) 

        """patch loss"""
        self.loss_patch = 0
        if i!=0:
            tt_cropF = self.nets['net_enc'](self.patches_in[i-1].detach())
            for layer in [self.content_layers[-2]]:
                self.loss_patch += self.calc_content_loss(tpF[layer],
                                                          tt_cropF[layer])
            self.losses['loss_patch_'+str(i+1)] = self.loss_patch

        self.loss_content_p = 0
        for layer in self.content_layers:
            self.loss_content_p += self.calc_content_loss(tpF[layer],
                                                      cF[layer],
                                                      norm=True)
        self.losses['loss_content_'+str(i+1)] = self.loss_content_p

        self.loss_ps = 0
        self.p_loss_style_remd = 0
        if type(self.cX)==bool:
            cx,cxminmax = xdog(self.content.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=1)
            sx,sxminmax = xdog(self.style_stack[1].detach(),self.gaussian_filter,self.gaussian_filter_2,style_conv,morphs=2,morph_cutoff=morph_cutoff)
        for j in range(i):
            cx = paddle.slice(cx,axes=[2,3],starts=[(self.positions[j][1]).astype('int32'),(self.positions[j][0]).astype('int32')],\
                             ends=[(self.positions[j][3]).astype('int32'),(self.positions[j][2]).astype('int32')])
        if cx.shape[-1]!=256:
            cx=F.interpolate(cx,size=(256,256))
        cXF = self.nets['net_enc'](cx.detach())
        stylized_dog,_ = xdog(self.stylized[i+1],self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=1,minmax=cxminmax)
        cdogF = self.nets['net_enc'](stylized_dog)
        mxdog_content = self.calc_content_loss(tpF['r31'], cXF['r31'])+self.calc_content_loss(tpF['r41'], cXF['r41'])
        mxdog_content_contraint = self.calc_content_loss(cdogF['r31'], cXF['r31'])+self.calc_content_loss(cdogF['r41'], cXF['r41'])

        if i>0:
            reshaped = self.style_stack[1].detach()
            for j in range(i):
                k = random_crop_coords(reshaped.shape[-1])
                reshaped=paddle.slice(reshaped,axes=[2,3],starts=[k[0],k[2]],ends=[k[1],k[3]])
                sx = paddle.slice(sx,axes=[2,3],starts=[k[0],k[2]],ends=[k[1],k[3]])
            if not reshaped.shape[-1]==256:
                reshaped = F.interpolate(reshaped,size=(256,256))
            spF = self.nets['net_enc'](reshaped.detach())
            for layer in self.content_layers:
                self.loss_ps += self.calc_style_loss(tpF[layer],
                                                      spF[layer])
            self.p_loss_style_remd += self.calc_style_emd_loss(
                tpF['r31'], spF['r31']) + self.calc_style_emd_loss(
                tpF['r41'], spF['r41'])
            sXF = self.nets['net_enc'](sx)
            mxdog_style=0
            mxdog_style+=self.mse_loss(cdogF['r31'], sXF['r31'])+self.mse_loss(cdogF['r41'], sXF['r41'])
            self.loss_ps = self.loss_ps
            self.p_loss_style_remd=self.p_loss_style_remd
            #mxdog_style=mxdog_style
        else:
            spF = self.nets['net_enc'](self.style_stack[0].detach())
            sXF = self.nets['net_enc'](sx)
            for layer in self.content_layers:
                self.loss_ps += self.calc_style_loss(tpF[layer],
                                                      spF[layer])
            self.p_loss_style_remd += self.calc_style_emd_loss(
                tpF['r31'], spF['r31']) + self.calc_style_emd_loss(
                tpF['r41'], spF['r41'])
            mxdog_style=self.mse_loss(cdogF['r31'], sXF['r31'])+self.mse_loss(cdogF['r41'], sXF['r41'])

        self.visual_items['cX']=cx
        self.visual_items['sX']=sx

        self.losses['loss_ps_'+str(i+1)] = self.loss_ps
        self.p_loss_content_relt = self.calc_content_relt_loss(
            tpF['r31'], cF['r31']) + self.calc_content_relt_loss(
            tpF['r41'], cF['r41'])
        self.p_loss_content_relt = self.p_loss_content_relt
        self.losses['p_loss_style_remd_'+str(i+1)] = self.p_loss_style_remd
        self.losses['p_loss_content_relt_'+str(i+1)] = self.p_loss_content_relt

        self.losses['loss_MD_'+str(i+1)] = mxdog_content*.3
        self.losses['loss_CnsC_'+str(i+1)] = mxdog_content_contraint*100

        """gan loss"""
        self.loss_Gp_GAN=0
        pred_fake_p = self.nets[self.discriminators[-1]](self.stylized[i+1])
        self.loss_Gp_GAN += self.gan_criterion(pred_fake_p, True)

        if i==0:
            a=11
            b=16
            c=1
            d=0
        elif i>0 and i<3:
            a=26
            b=26
            c=2.5
            d=3
        else:
            a=26
            b=26
            c=5
            d=5
        if i>1 and i<3:
            e=2000
            f=1
        elif i==3:
            e=2000
            f=2.5
        else:
            e=1000
            f=1

        self.losses['loss_CnsS_'+str(i+1)] = mxdog_style*e
        mxdogloss=mxdog_content * .3 + mxdog_content_contraint *100 + mxdog_style * e

        self.loss = self.loss_Gp_GAN *c+self.loss_ps * self.style_weight*f +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch*d +\
                    self.p_loss_style_remd * a + self.p_loss_content_relt * b + mxdogloss

        return self.loss

    def backward_D(self,dec,i,name):
        """Calculate GAN loss for the discriminator"""
        fake = self.stylized[i+1].detach()
        pred_p_fake = dec(fake)
        loss_Dp_fake = self.gan_criterion(pred_p_fake, False)

        pred_Dp_real = 0
        reshaped = self.style_stack[1]
        if i>0:
            for j in range(i):
                k = random_crop_coords(reshaped.shape[-1])
                reshaped=paddle.slice(reshaped,axes=[2,3],starts=[k[0],k[2]],ends=[k[1],k[3]])
            if not reshaped.shape[-1]==256:
                reshaped = F.interpolate(reshaped,size=(256,256))
            loss_Dp_real = dec(reshaped.detach())
            pred_Dp_real += self.gan_criterion(loss_Dp_real, True)
            pred_Dp_real=pred_Dp_real
        else:
            reshaped = F.interpolate(reshaped,size=(256,256))
            loss_Dp_real = dec(reshaped.detach())
            pred_Dp_real += self.gan_criterion(loss_Dp_real, True)
        self.loss_D_patch = (loss_Dp_fake + pred_Dp_real) * 0.5
        self.losses[name+'_fake_loss_'+str(i)] = loss_Dp_fake
        self.losses[name+'_real_loss_'+str(i)] = pred_Dp_real
        return self.loss_D_patch

    def train_iter(self, optimizers=None):
        loops=1
        '''
        if self.iters>=self.rev3_iter:
            loops+=1
        if self.iters>=self.rev4_iter:
            loops+=1
        '''
        # compute fake images: G(A)
        self.forward()
        # update D
        optimizers[self.o[-1]].clear_grad()
        self.set_requires_grad(self.nets[self.discriminators[-1]],True)
        loss = self.backward_D(self.nets[self.discriminators[-1]],self.train_layer-1,str(self.train_layer))
        loss.backward()
        optimizers[self.o[-1]].step()
        self.set_requires_grad(self.nets[self.discriminators[-1]],False)
        optimizers[self.o[-1]].clear_grad()

        self.set_requires_grad(self.nets['spectral_D2'],True)
        loss=0
        optimizers['optimSD'].clear_grad()
        for i in range(4):
            loss+=self.backward_D(self.nets['spectral_D2'],self.train_layer-1,str(self.train_layer-1)+'s')
        loss.backward()
        optimizers['optimSD'].step()
        self.set_requires_grad(self.nets['spectral_D2'],False)

        optimizers[self.go[-1]].clear_grad()
        loss = self.backward_G(self.train_layer-1)
        loss.backward()
        optimizers[self.go[-1]].step()
        optimizers[self.go[-1]].clear_grad()

@MODELS.register()
class LapStyleRevSecondMiddle(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 revnet_deep_generator,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0,
                 ada_alpha=1.0,
                 ada_alpha_2=1.0,
                 gan_thumb_weight=1.0,
                 gan_patch_weight=1.0,
                 use_mdog=0,
                 morph_cutoff=47.9,
                 rev3_iter=0,
                 rev4_iter=0):

        super(LapStyleRevSecondMiddle, self).__init__()

        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)
        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define the first revnet params
        self.nets['net_rev'] = build_generator(revnet_generator)
        self.set_requires_grad([self.nets['net_rev']], False)

        # define the second revnet params
        self.nets['net_rev_2'] = build_generator(revnet_deep_generator)
        init_weights(self.nets['net_rev_2'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])
        self.discriminators=[self.nets['netD']]

        l = np.repeat(np.array([[[[-8, -8, -8], [-8, 1, -8], [-8, -8, -8]]]]), 3, axis=0)
        self.lap_filter = paddle.nn.Conv2D(3, 3, (3, 3), stride=1, bias_attr=False,
                                           padding=1, groups=3, padding_mode='reflect',
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                   value=l), trainable=False)
                                           )

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.ada_alpha = ada_alpha
        self.ada_alpha_2 = ada_alpha_2
        self.gan_thumb_weight = gan_thumb_weight
        self.gan_patch_weight = gan_patch_weight
        self.morph_cutoff = morph_cutoff
        g = np.repeat(gaussian(7, 1).numpy(), 3, axis=0)
        g2 = np.repeat(gaussian(19, 3).numpy(), 3, axis=0)
        self.gaussian_filter = paddle.nn.Conv2D(3, 3, 7,
                                                groups=3, bias_attr=False,
                                                padding=3, padding_mode='reflect',
                                                weight_attr=paddle.ParamAttr(
                                                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                        value=g), trainable=False)
                                                )
        self.gaussian_filter_2 = paddle.nn.Conv2D(3, 3, 25,
                                                  groups=3, bias_attr=False,
                                                  padding=9, padding_mode='reflect',
                                                  weight_attr=paddle.ParamAttr(
                                                      initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                                                          value=g2), trainable=False)
                                                  )

        self.morph_conv = paddle.nn.Conv2D(3, 3, 3, padding=1, groups=3,
                                           padding_mode='reflect', bias_attr=False,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.Constant(
                                                   value=1), trainable=False)
                                           )
        self.morph_conv_2 = paddle.nn.Conv2D(3, 3, 11, padding=5, groups=3,
                                           padding_mode='reflect', bias_attr=False,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.fluid.initializer.Constant(
                                                   value=1), trainable=False)
                                           )



    def setup_input(self, input):
        if self.is_train:
            self.content_stack = []
            self.style_stack = [paddle.to_tensor(input['style_stack_1']),paddle.to_tensor(input['style_stack_2'])]
            self.laplacians=[]
            for i in range(1,5):
                if 'content_stack_'+str(i) in input:
                    self.content_stack.append(paddle.to_tensor(input['content_stack_'+str(i)]))
            self.visual_items['ci'] = self.content_stack[0]

            self.content=input['content']
            self.positions = input['position_stack']
            self.size_stack = input['size_stack']
            self.laplacians.append(laplacian_conv(self.content_stack[0],self.lap_filter).detach())
            self.laplacians.append(laplacian_conv(self.content_stack[1],self.lap_filter).detach())
            self.laplacians.append(laplacian_conv(self.content_stack[2],self.lap_filter).detach())
            self.sX=False
            self.cX = False

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        cF = self.nets['net_enc'](F.interpolate(self.content_stack[0],scale_factor=.5))
        sF = self.nets['net_enc'](F.interpolate(self.style_stack[0], scale_factor=.5))

        stylized_small= self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.laplacians[0], stylized_up], axis=1)
        #rev_net thumb only calcs as patch if second parameter is passed
        stylized_rev_lap,stylized_feats = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
        self.visual_items['stylized_rev_first'] = stylized_rev
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)
        stylized_up = crop_upsized(stylized_up,self.positions[0],self.size_stack[0])
        self.patches_in = [stylized_up.detach()]
        revnet_input = paddle.concat(x=[self.laplacians[1], stylized_up], axis=1)
        stylized_rev_lap_second,stylized_feats = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats,self.ada_alpha)
        stylized_rev_second = fold_laplace_pyramid([stylized_rev_lap_second, stylized_up])
        self.visual_items['ci_2'] = self.content_stack[1]
        self.stylized= [stylized_rev_second]

        self.visual_items['stylized_rev_second'] = stylized_rev_second

        stylized_up = F.interpolate(stylized_rev_second, scale_factor=2)
        stylized_up = crop_upsized(stylized_up,self.positions[1],self.size_stack[1])
        self.patches_in.append(stylized_up)

        revnet_input = paddle.concat(x=[self.laplacians[2], stylized_up.detach()], axis=1)
        stylized_rev_patch,stylized_feats = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats.detach(),self.ada_alpha_2)
        stylized_rev_patch = fold_laplace_patch(
            [stylized_rev_patch, stylized_up.detach()])
        self.visual_items['ci_3'] = self.content_stack[2]
        self.visual_items['stylized_rev_third'] = stylized_rev_patch
        self.stylized.append(stylized_rev_patch)

        stylized_up = F.interpolate(stylized_rev_patch, scale_factor=2)
        stylized_up = crop_upsized(stylized_up,self.positions[2],self.size_stack[2])
        self.patches_in.append(stylized_up)

    def backward_G(self,i):
        cF = self.nets['net_enc'](self.content_stack[i])

        with paddle.no_grad():
            tt_cropF = self.nets['net_enc'](self.patches_in[i])

        tpF = self.nets['net_enc'](self.stylized[i])

        """patch loss"""
        self.loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            self.loss_patch += self.calc_content_loss(tpF[layer],
                                                      tt_cropF[layer])
        self.losses['loss_patch_'+str(i+1)] = self.loss_patch

        self.loss_content_p = 0
        for layer in self.content_layers:
            self.loss_content_p += self.calc_content_loss(tpF[layer],
                                                      cF[layer],
                                                      norm=True)
        self.losses['loss_content_'+str(i+1)] = self.loss_content_p

        self.loss_ps = 0
        self.p_loss_style_remd = 0

        mxdog_style=0
        style_counter=0
        if type(self.cX)==bool:
            _,cxminmax = xdog(self.content.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2)
            _,sxminmax = xdog(self.style_stack[1].detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2)
        cX,_ = xdog(self.content_stack[i].detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2,minmax=cxminmax)
        cXF = self.nets['net_enc'](cX.detach())
        stylized_dog,_ = xdog(self.stylized[i],self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2,minmax=cxminmax)
        cdogF = self.nets['net_enc'](stylized_dog)

        mxdog_content = self.calc_content_loss(tpF['r31'], cXF['r31'])
        mxdog_content_contraint = self.calc_content_loss(cdogF['r31'], cXF['r31'])

        reshaped = self.style_stack[1]
        for j in range(i):
            k = random_crop_coords(reshaped.shape[-1])
            reshaped=paddle.slice(reshaped,axes=[2,3],starts=[k[0],k[2]],ends=[k[1],k[3]])
        if not reshaped.shape[-1]==512:
            reshaped = F.interpolate(reshaped,size=(512,512))
        reshaped = paddle.split(reshaped, 2, 2)
        for idx,k in enumerate(reshaped):
            for itx,j in enumerate(paddle.split(k, 2, 3)):
                spF = self.nets['net_enc'](j.detach())
                for layer in self.content_layers:
                    self.loss_ps += paddle.clip(self.calc_style_loss(tpF[layer],
                                                          spF[layer]), 1e-5, 1e5)
                self.p_loss_style_remd += self.calc_style_emd_loss(
                    tpF['r31'], spF['r31']) + self.calc_style_emd_loss(
                    tpF['r41'], spF['r41'])
                sX,_ = xdog(j.detach(),self.gaussian_filter,self.gaussian_filter_2,self.morph_conv,morphs=2,minmax=sxminmax)
                sXF = self.nets['net_enc'](sX.detach())
                mxdog_style+=self.calc_style_loss(cdogF['r31'], sXF['r31'])
                style_counter += 1
                if style_counter==4:
                    self.visual_items['sX_'+str(i)]=sX

        self.losses['loss_ps_'+str(i+1)] = self.loss_ps/4
        self.p_loss_content_relt = self.calc_content_relt_loss(
            tpF['r31'], cF['r31']) + self.calc_content_relt_loss(
            tpF['r41'], cF['r41'])
        self.p_loss_style_remd = paddle.clip(self.p_loss_style_remd, 1e-5, 1e5)
        self.p_loss_content_relt = paddle.clip(self.p_loss_content_relt, 1e-5, 1e5)
        self.losses['p_loss_style_remd_'+str(i+1)] = self.p_loss_style_remd/4
        self.losses['p_loss_content_relt_'+str(i+1)] = self.p_loss_content_relt

        """gan loss"""
        pred_fake_p = self.discriminators[0](self.stylized[i])
        self.loss_Gp_GAN = self.gan_criterion(pred_fake_p, True)
        self.losses['loss_gan_Gp_'+str(i+1)] = self.loss_Gp_GAN*self.gan_thumb_weight

        self.losses['loss_MD_'+str(i+1)] = mxdog_content*.0125
        self.losses['loss_CnsC_'+str(i+1)] = mxdog_content_contraint*25
        self.losses['loss_CnsS_'+str(i+1)] = mxdog_style*125/4
        mxdogloss=mxdog_content * .0125 + mxdog_content_contraint *25 + (mxdog_style/4) * 125

        self.loss = self.loss_Gp_GAN *self.gan_thumb_weight +self.loss_ps/4 * self.style_weight +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch +\
                    self.p_loss_style_remd/4 * 22 + self.p_loss_content_relt * 22 + mxdogloss/(i+1)

        return self.loss

    def backward_D(self,dec,i):
        """Calculate GAN loss for the discriminator"""
        pred_p_fake = dec(self.stylized[i].detach())
        loss_Dp_fake = self.gan_criterion(pred_p_fake, False)

        pred_Dp_real = 0
        reshaped = self.style_stack[1]
        for j in range(i):
            k = random_crop_coords(reshaped.shape[-1])
            reshaped=paddle.slice(reshaped,axes=[2,3],starts=[k[0],k[2]],ends=[k[1],k[3]])
        if not reshaped.shape[-1]==512:
            reshaped = F.interpolate(reshaped,size=(512,512))
        reshaped = paddle.split(reshaped, 2, 2)
        for k in reshaped:
            for j in paddle.split(k, 2, 3):
                loss_Dp_real = dec(j.detach())
                pred_Dp_real += self.gan_criterion(loss_Dp_real, True)
        self.loss_D_patch = (loss_Dp_fake + pred_Dp_real/4) * 0.5
        self.losses['Dp_fake_loss_'+str(i)] = loss_Dp_fake
        self.losses['Dp_real_loss_'+str(i)] = pred_Dp_real/4
        return self.loss_D_patch


    def train_iter(self, optimizers=None):
        loops=2
        # compute fake images: G(A)
        self.forward()
        # update D

        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimD'].clear_grad()
        l1=self.backward_D(self.nets['netD'],0)
        l2=self.backward_D(self.nets['netD'],1)
        (l1+l2).backward()
        optimizers['optimD'].step()
        self.set_requires_grad(self.nets['netD'], False)

        # update G
        optimizers['optimG'].clear_grad()
        l1=self.backward_G(0)
        l2=self.backward_G(1)
        (l1+l2).backward()
        optimizers['optimG'].step()
        optimizers['optimG'].clear_grad()

def random_crop_coords(size):
    halfsize=math.floor(size/2)
    bottommost = random.choice(list(range(0, size - halfsize,2)))
    leftmost = random.choice(list(range(0, size - halfsize,2)))
    return (bottommost,bottommost+halfsize,leftmost,leftmost+halfsize)

@MODELS.register()
class LapStyleRevFirstPatch(BaseModel):
    def __init__(self,
             revnet_generator,
             revnet_discriminator,
             draftnet_encode,
             draftnet_decode,
             calc_style_emd_loss=None,
             calc_content_relt_loss=None,
             calc_content_loss=None,
             calc_style_loss=None,
             gan_criterion=None,
             content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
             style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
             content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevFirstPatch, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)

        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_enc']], False)

        # define revision-net params
        self.nets['net_rev'] = build_generator(revnet_generator)
        self.set_requires_grad([self.nets['net_rev']], False)
        self.nets['net_rev_2'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev_2'])
        self.nets['netD_patch'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD_patch'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):

        self.position = input['position']
        self.half_position = input['half_position']
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.sp = paddle.to_tensor(input['sp'])
        self.cp = paddle.to_tensor(input['cp'])
        self.visual_items['cp'] = self.cp

        self.pyr_ci = make_laplace_pyramid(self.ci, 2)
        self.pyr_si = make_laplace_pyramid(self.si, 2)
        self.pyr_cp = make_laplace_pyramid(self.cp, 2)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)
        self.pyr_cp.append(self.cp)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[2])
        sF = self.nets['net_enc'](self.pyr_si[2])

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[1], stylized_up], axis=1)
        stylized_rev_lap,stylized_feats = self.nets['net_rev'](revnet_input.detach())
        #self.ttF_res=self.ttF_res.detach()
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap.detach(), stylized_small.detach()])
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)
        p_stylized_up = paddle.slice(stylized_up,axes=[2,3],starts=[self.half_position[0],self.half_position[2]],ends=[self.half_position[1],self.half_position[3]])
        p_revnet_input = paddle.concat(x=[self.pyr_cp[1], p_stylized_up], axis=1)
        p_stylized_rev_lap,stylized_feats = self.nets['net_rev'](p_revnet_input.detach(),stylized_feats.detach())
        p_stylized_rev = fold_laplace_pyramid([p_stylized_rev_lap.detach(), p_stylized_up.detach()])

        stylized_up = F.interpolate(p_stylized_rev, scale_factor=2)
        patch_origin_size = 512
        i = random_crop_coords(patch_origin_size)

        stylized_feats = self.nets['net_rev_2'].DownBlock(p_revnet_input.detach())
        stylized_feats = self.nets['net_rev_2'].resblock(stylized_feats)

        self.input_crop = paddle.slice(stylized_up.detach(),axes=[2,3],starts=[i[0],i[2]],ends=[i[1],i[3]])
        cp_crop = paddle.slice(self.pyr_cp[0],axes=[2,3],starts=[i[0],i[2]],ends=[i[1],i[3]])
        p_revnet_input = paddle.concat(x=[cp_crop, self.input_crop], axis=1)
        p_stylized_rev_patch,_ = self.nets['net_rev_2'](p_revnet_input.detach(),stylized_feats)
        p_stylized_rev_patch = p_stylized_rev_patch+ self.input_crop.detach()

        stylized = stylized_rev
        self.p_stylized = p_stylized_rev_patch
        self.content_patch = paddle.slice(self.cp,axes=[2,3],starts=[i[0],i[2]],ends=[i[1],i[3]])
        self.visual_items['stylized'] = stylized
        self.visual_items['stylized_patch'] = p_stylized_rev
        self.visual_items['stylized_patch_2'] = p_stylized_rev_patch
        self.crop_marks = i
        self.style_patch = paddle.slice(self.sp,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
        self.style_patch = paddle.slice(self.style_patch,axes=[2,3],starts=[self.crop_marks[0],self.crop_marks[2]],ends=[self.crop_marks[1],self.crop_marks[3]])



    def backward_G(self, optimizer):

        self.cF = self.nets['net_enc'](self.content_patch)

        with paddle.no_grad():
            self.tt_cropF = self.nets['net_enc'](self.input_crop)

        self.tpF = self.nets['net_enc'](self.p_stylized)

        """patch loss"""
        self.loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            self.loss_patch += paddle.clip(self.calc_content_loss(self.tpF[layer],
                                                      self.tt_cropF[layer]), 1e-5, 1e5)
        self.losses['loss_patch'] = self.loss_patch

        self.loss_content_p = 0
        for layer in self.content_layers:
            self.loss_content_p += paddle.clip(self.calc_content_loss(self.tpF[layer],
                                                      self.cF[layer],
                                                      norm=True), 1e-5, 1e5)
        self.losses['loss_content_p'] = self.loss_content_p

        self.loss_ps = 0
        self.p_loss_style_remd = 0

        style_patches = paddle.slice(self.sp,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
        reshaped = paddle.split(style_patches, 2, 2)
        for i in reshaped:
            for j in paddle.split(i, 2, 3):
                spF = self.nets['net_enc'](j.detach())
                for layer in self.content_layers:
                    self.loss_ps += paddle.clip(self.calc_style_loss(self.tpF[layer],
                                                          spF[layer]), 1e-5, 1e5)
                self.p_loss_style_remd += self.calc_style_emd_loss(
                    self.tpF['r31'], spF['r31']) + self.calc_style_emd_loss(
                    self.tpF['r41'], spF['r41'])
        self.losses['loss_ps'] = self.loss_ps
        self.p_loss_content_relt = self.calc_content_relt_loss(
            self.tpF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
            self.tpF['r41'], self.cF['r41'])
        self.p_loss_style_remd = paddle.clip(self.p_loss_style_remd, 1e-5, 1e5)
        self.p_loss_content_relt = paddle.clip(self.p_loss_content_relt, 1e-5, 1e5)
        self.losses['p_loss_style_remd'] = self.p_loss_style_remd
        self.losses['p_loss_content_relt'] = self.p_loss_content_relt

        """gan loss"""
        pred_fake_p = self.nets['netD_patch'](self.p_stylized)
        self.loss_Gp_GAN = paddle.clip(self.gan_criterion(pred_fake_p, True), 1e-5, 1e5)
        self.losses['loss_gan_Gp'] = self.loss_Gp_GAN


        self.loss = self.loss_Gp_GAN +self.loss_ps/4 * self.style_weight +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight * 20 +\
                    self.p_loss_style_remd/4 * 22 + self.p_loss_content_relt * 22
        self.loss.backward()

        return self.loss


    def backward_Dpatch(self):
        """Calculate GAN loss for the patch discriminator"""
        pred_p_fake = self.nets['netD_patch'](self.p_stylized.detach())
        self.loss_Dp_fake = paddle.clip(self.gan_criterion(pred_p_fake, False), 1e-5, 1e5)

        pred_Dp_real = 0
        style_patches = paddle.slice(self.sp,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
        reshaped = paddle.split(style_patches, 2, 2)
        for i in reshaped:
            for j in paddle.split(i, 2, 3):
                self.loss_Dp_real = self.nets['netD_patch'](j.detach())
                pred_Dp_real += paddle.clip(self.gan_criterion(self.loss_Dp_real, True), 1e-5, 1e5)
        self.loss_D_patch = (self.loss_Dp_fake + pred_Dp_real/4) * 0.5

        self.loss_D_patch.backward()

        self.losses['Dp_fake_loss'] = self.loss_Dp_fake
        self.losses['Dp_real_loss'] = pred_Dp_real

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()
        # update D
        self.set_requires_grad(self.nets['netD_patch'], True)
        optimizers['optimD_patch'].clear_grad()
        self.backward_Dpatch()
        optimizers['optimD_patch'].step()

        # update G

        self.set_requires_grad(self.nets['netD_patch'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G(optimizers['optimG'])
        optimizers['optimG'].step()

@MODELS.register()
class LapStyleDraK(BaseModel):
    def __init__(self,
                 generator_encode,
                 generator_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleDraK, self).__init__()

        # define generators
        self.nets['net_enc'] = build_generator(generator_encode)
        self.nets['net_dec'] = build_generator(generator_decode)
        init_weights(self.nets['net_dec'])
        self.set_requires_grad([self.nets['net_enc']], False)

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)
        self.stylized = self.nets['net_dec'](self.cF, self.sF)
        self.visual_items['stylized'] = self.stylized

    def backward_Dec(self):
        self.tF = self.nets['net_enc'](self.stylized)
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers[:-1]:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """IDENTITY LOSSES"""
        self.Icc = self.nets['net_dec'](self.cF, self.cF)
        self.l_identity1 = self.calc_content_loss(self.Icc, self.ci)
        self.Fcc = self.nets['net_enc'](self.Icc)
        self.l_identity2 = 0
        for layer in self.content_layers:
            self.l_identity2 += self.calc_content_loss(self.Fcc[layer],
                                                       self.cF[layer])
        self.losses['l_identity1'] = self.l_identity1
        self.losses['l_identity2'] = self.l_identity2
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt

        self.loss = self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.l_identity1 * 50 + self.l_identity2 * 1 + self.loss_style_remd * 10 + \
                    self.loss_content_relt * 16
        self.loss.backward()

        return self.loss

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights"""
        self.forward()
        optimizers['optimG'].clear_grad()
        self.backward_Dec()
        self.optimizers['optimG'].step()