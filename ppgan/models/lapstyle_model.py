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

from .builder import MODELS
from .generators.builder import build_generator
from .criterions import build_criterion
from .discriminators.builder import build_discriminator

from ..modules.init import init_weights
from ..utils.visual import tensor2img, save_image
from ..utils.filesystem import makedirs, save, load


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
                 style_weight=3.0):

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

    def setup_input(self, input):

        self.position = input['position']
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.sp = paddle.to_tensor(input['sp'])
        self.cp = paddle.to_tensor(input['cp'])
        self.visual_items['cp'] = self.cp

        self.pyr_ci = make_laplace_pyramid(self.ci, 1)
        self.pyr_si = make_laplace_pyramid(self.si, 1)
        self.pyr_cp = make_laplace_pyramid(self.cp, 1)
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
        p_stylized_rev_lap,stylized_feats = self.nets['net_rev'](p_revnet_input.detach(),stylized_feats.detach())
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

        self.loss = self.loss_G_GAN + self.loss_s * self.style_weight +\
                    self.loss_content * self.content_weight+\
                    self.loss_style_remd * 16 +\
                    self.loss_content_relt * 16
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
        for layer in self.content_layers:
            self.loss_ps += self.calc_style_loss(self.tpF[layer],
                                                          self.spF[layer])
        self.losses['loss_ps'] = self.loss_ps

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


        self.loss = self.loss_Gp_GAN +self.loss_ps * self.style_weight*2 +\
                          self.loss_content_p * self.content_weight +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight * 50 +\
                    self.p_loss_style_remd * 26 + self.p_loss_content_relt * 26
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
                 style_weight=3.0):

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

    def test_iter(self, output_dir=None,metrics=None):
        self.eval()
        self.output_dir=output_dir
        self.laplacians=[laplacian(self.content_stack[0])]
        print('content_size='+str(self.content.shape))
        for i in [.25,.5,1]:
            if i==1:
                self.laplacians.append(laplacian(self.content))
            else:
                self.laplacians.append(laplacian(F.interpolate(self.content, scale_factor=i)))
        with paddle.no_grad():
            cF = self.nets['net_enc'](F.interpolate(self.content_stack[0],scale_factor=.5))
            sF = self.nets['net_enc'](F.interpolate(self.style_stack[0], scale_factor=.5))

            stylized_small= self.nets['net_dec'](cF, sF)
            self.stylized_up = F.interpolate(stylized_small, scale_factor=2)
            small_side=min(self.stylized_up.shape[-1],self.stylized_up.shape[-2])


            revnet_input = paddle.concat(x=[self.laplacians[0], self.stylized_up], axis=1)
            # rev_net thumb only calcs as patch if second parameter is passed
            stylized_rev_lap, self.stylized_feats = self.nets['net_rev'](revnet_input)
            stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
            self.stylized_slice = F.interpolate(stylized_rev, scale_factor=2)
            print('stylized_up='+str(self.stylized_slice.shape))
            if small_side==self.stylized_up.shape[-1]:
                size_x = self.stylized_slice.shape[-2]
                self.in_size_x = math.floor(size_x / 2)
                move_x = adjust(size_x, self.in_size_x)
                ranges_x=list(range(0,size_x,move_x))
                size_y = 512
                self.in_size_y = 256
                move_y = 256
                ranges_y = list(range(0,size_y,move_y))
            else:
                size_x=512
                self.in_size_x = 256
                move_x = 256
                ranges_x = list(range(0,size_x,move_x))
                size_y = self.stylized_slice.shape[-1]
                self.in_size_y = math.floor(size_y / 2)
                move_y = adjust(size_y, self.in_size_y)
                ranges_y=list(range(0,size_y,move_y))
            ranges_x = ranges_x + [i+math.floor(self.in_size_x/16) for i in ranges_x]
            ranges_y = ranges_y + [i+math.floor(self.in_size_y/16) for i in ranges_y]
            self.save_width=False
            self.save_height=False
            print('ranges x: '+str(ranges_x))
            print('ranges y: '+str(ranges_y))
            for i in ranges_x:
                if i == ranges_x[-1]:
                    self.save_width = True
                for j in ranges_y:
                    if j==ranges_y[=1]:
                        self.save_height=True
                    self.outer_loop=(i,j)
                    self.positions=[[i,j,i+self.in_size_x,j+self.in_size_y]]#!
                    self.test_forward(self.stylized_slice,self.stylized_feats)
            style_paths = [i for i in os.listdir(os.path.join(self.output_dir, 'visual_test'))]
            style_paths = [i for i in style_paths if '_' in i]
            print(style_paths[0])
            positions = [(int(re.split('_|\.',i)[0]),int(re.split('_|\.',i)[1])) for i in style_paths]
            max_x = 0
            max_y = 0
            for a,b in positions:
                if a>max_x:
                    max_x=a
                if b>max_y:
                    max_y=b
            max_x = max_x+self.rm_width
            max_y = max_y+self.bm_height
            print(max_x)
            print(max_y)
            tiles_1 = np.zeros((max_x,max_y,3), dtype=np.uint8)
            print(tiles_1.shape)
            data_visits = np.zeros((max_x,max_y,3), dtype=np.uint32)
            weights_sum = np.zeros((max_x,max_y,3))
            #tiles_2 = np.zeros((max_x, max_y,3), dtype=np.uint8)
            for a,b in zip(style_paths,positions):
                with Image.open(os.path.join(self.output_dir, 'visual_test',a)) as file:
                    image = np.asarray(file)
                    '''
                    if b[0]%size_x==0 and b[1]%size_y==0:
                        tiles_1[b[0]:b[0]+image.shape[0],b[1]:b[1]+image.shape[1],:]=image
                    else:
                        tiles_2[b[0]:b[0] + image.shape[0], b[1]:b[1] + image.shape[1],:] = image
                    '''
                    x_mod_1=8
                    x_mod_2=8
                    y_mod_1=8
                    y_mod_2=8
                    if b[0]==0:
                        x_mod_1=0
                    if b[1]==0:
                        y_mod_1=0
                    if b[0]+self.in_size_x==max_x:
                        x_mod_2=0
                    if b[1]+self.in_size_y==max_y:
                        y_mod_2=0
                    tiles_1[b[0]+x_mod_1:b[0]+image.shape[0]-x_mod_2,b[1]+y_mod_1:b[1]+image.shape[1]-y_mod_2,:]=image[x_mod_1:image.shape[0]-x_mod_2,y_mod_1:image.shape[1]-y_mod_2,:]
            for a,b in zip([tiles_1],['tiled']):
                print(self.path)
                im = Image.fromarray(a,'RGB')
                label = self.path[0]+' '+b
                makedirs(os.path.join(self.output_dir, 'visual_test'))
                img_path = os.path.join(self.output_dir, 'visual_test',
                                        '%s.png' % (label))
                im.save(img_path)
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
        stylized_rev_lap_second,stylized_feats = self.nets['net_rev'](revnet_input.detach(),stylized_feats)
        stylized_rev_second = fold_laplace_pyramid([stylized_rev_lap_second, stylized_up])
        stylized_up = F.interpolate(stylized_rev_second, scale_factor=2)
        size_x = stylized_up.shape[-2]
        in_size_x = math.floor(size_x / 2)
        move_x = adjust(size_x, in_size_x)
        size_y = stylized_up.shape[-1]
        in_size_y = math.floor(size_y / 2)
        move_y = adjust(size_y, in_size_y)
        print('size_x='+str(size_x))
        print('size_y='+str(size_y))
        print('in_size_x='+str(in_size_x))
        print('in_size_y='+str(in_size_y))
        for i in range(0,size_x,move_x):
            for j in range(0,size_y,move_y):
                label = str(self.outer_loop[0]*4+i*2)+'_'+str(self.outer_loop[1]*4+j*2)
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

                stylized_feats_2 = self.nets['net_rev_2'].DownBlock(revnet_input.detach())
                stylized_feats_2 = self.nets['net_rev_2'].resblock(stylized_feats_2)
                lap_2 = paddle.slice(self.laplacians[2],axes=[2,3],starts=[self.outer_loop[0]*2+i,self.outer_loop[1]*2+j],
                                   ends=[self.outer_loop[0]*2+i+in_size_x,self.outer_loop[1]*2+j+in_size_y])
                print('lap_2.shape[-2]='+str(lap_2.shape[-2]))
                print('lap_2.shape[-1]='+str(lap_2.shape[-1]))
                if lap_2.shape[-2]!=in_size_x or lap_2.shape[-1]!=in_size_y:
                    continue
                if stylized_up_2.shape[-2]!=in_size_x or stylized_up_2.shape[-1]!=in_size_y:
                    continue
                revnet_input_2 = paddle.concat(x=[lap_2, stylized_up_2.detach()], axis=1)
                stylized_rev_patch,stylized_feats = self.nets['net_rev_2'](revnet_input_2.detach(),stylized_feats_2.detach())
                stylized_rev_patch = fold_laplace_patch(
                    [stylized_rev_patch, stylized_up_2.detach()])

                stylized_up_3 = F.interpolate(stylized_rev_patch, scale_factor=2)
                for k in range(0,size_x,move_x):
                    for l in range(0,size_y,move_y):
                        label = str(self.outer_loop[0]*4+i*2+k)+'_'+str(self.outer_loop[1]*4+j*2+l)
                        if label in self.labels:
                            continue
                        else:
                            self.labels.append(label)
                        if k+in_size_x>stylized_up_3.shape[-2] or l+in_size_y>stylized_up_3.shape[-1]:
                            continue
                        stylized_up_4 = paddle.slice(stylized_up_3,axes=[2,3],starts=[k,l],\
                             ends=[k+in_size_x,l+in_size_y])
                        lap_3 = paddle.slice(self.laplacians[3],axes=[2,3],starts=[self.outer_loop[0]*4+i*2+k,self.outer_loop[1]*4+j*2+l*1],
                                   ends=[self.outer_loop[0]*4+i*2+k+in_size_x,self.outer_loop[1]*4+j*2+l+in_size_y])
                        if lap_3.shape[-2]!=in_size_x or lap_3.shape[-1]!=in_size_y:
                            continue
                        if stylized_up_4.shape[-2]!=in_size_x or stylized_up_4.shape[-1]!=in_size_y:
                            continue
                        revnet_input_3 = paddle.concat(x=[lap_3, stylized_up_4.detach()], axis=1)
                        stylized_rev_patch_second,_ = self.nets['net_rev_2'](revnet_input_3.detach(),stylized_feats_2.detach())
                        stylized_rev_patch_second = fold_laplace_patch(
                            [stylized_rev_patch_second, stylized_up_4.detach()])
                        image_numpy=tensor2img(stylized_rev_patch_second,min_max=(0., 1.))
                        makedirs(os.path.join(self.output_dir, 'visual_test'))
                        img_path = os.path.join(self.output_dir, 'visual_test',
                                                '%s.png' % (label))
                        save_image(image_numpy, img_path)
                        if self.save_width:
                            self.rm_width=stylized_rev_patch_second.shape[-2]
                        if self.save_width:
                            self.bm_height=stylized_rev_patch_second.shape[-1]
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
        stylized_rev_lap_second,stylized_feats = self.nets['net_rev'](revnet_input.detach(),stylized_feats)
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
        stylized_rev_patch,stylized_feats = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats.detach())
        stylized_rev_patch = fold_laplace_patch(
            [stylized_rev_patch, stylized_up.detach()])
        self.visual_items['ci_3'] = self.content_stack[2]
        self.visual_items['stylized_rev_third'] = stylized_rev_patch

        stylized_up = F.interpolate(stylized_rev_patch, scale_factor=2)
        stylized_up = crop_upsized(stylized_up,self.positions[2],self.size_stack[2])
        self.second_patch_in = stylized_up.detach()

        revnet_input = paddle.concat(x=[self.laplacians[3], stylized_up.detach()], axis=1)
        stylized_rev_patch_second,_ = self.nets['net_rev_2'](revnet_input.detach(),stylized_feats.detach())
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
        for layer in self.content_layers:
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
        pred_fake_p = self.nets['netD'](self.stylized)
        self.loss_Gp_GAN = paddle.clip(self.gan_criterion(pred_fake_p, True), 1e-5, 1e5)
        self.losses['loss_gan_Gp'] = self.loss_Gp_GAN


        self.loss = self.loss_Gp_GAN * 1.5 +self.loss_ps/4 * self.style_weight*1.08625 +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight *20 +\
                    self.p_loss_style_remd/4 * 18 + self.p_loss_content_relt * 26
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
        for layer in self.content_layers:
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
                for layer in self.content_layers:
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


        loss_patch = loss_Gp_GAN * 1.5+loss_ps/4 * self.style_weight*1.25 +\
                    loss_content_p * self.content_weight +\
                    loss_patch * self.content_weight *20 +\
                    p_loss_style_remd/4 * 120 + p_loss_content_relt * 26
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
