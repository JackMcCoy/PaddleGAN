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
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .criterions import build_criterion
from .discriminators.builder import build_discriminator

from ..modules.init import init_weights


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
        self.stylized_patch,self.stylized_patch_feat = self.nets['net_dec'](self.cF, self.sF, self.cpF, 'patch')
        self.visual_items['stylized_thumb'] = self.stylized_thumb
        self.visual_items['stylized_patch'] = self.stylized_patch

    def backward_Dec(self):
        with paddle.no_grad():
            g_t_thumb_up = F.interpolate(self.visual_items['stylized_thumb'], scale_factor=2, mode='bilinear', align_corners=False)
            g_t_thumb_crop = paddle.slice(g_t_thumb_up,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
            self.tt_cropF = self.nets['net_enc'](g_t_thumb_crop)
            #style_patch = F.interpolate(self.visual_items['si'], scale_factor=2, mode='bilinear', align_corners=False)
            #style_patch_crop = paddle.slice(style_patch,axes=[2,3],starts=[self.position[0],self.position[2]],ends=[self.position[1],self.position[3]])
            self.spCrop = self.nets['net_enc'](self.sp)
        self.ttF = self.nets['net_enc'](self.stylized_thumb)
        self.tpF = self.nets['net_enc'](self.stylized_patch)
        """content loss"""
        self.loss_c = 0
        for layer in [self.content_layers[-2]]:
            self.loss_c +=self.calc_content_loss(self.ttF[layer],self.stylized_thumb_feat[layer])

        self.losses['loss_c'] = self.loss_c

        self.loss_content = 0
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

        self.loss = self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.l_identity1 * 55 + self.l_identity2 * 1 +\
                    self.loss_content * self.content_weight+\
                    self.loss_style_remd * 10 +\
                    self.loss_content_relt * 16
        self.loss.backward()
        self.optimizers['optimG'].step()

        """patch loss"""
        self.loss_patch = 0
        #self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[-2]]:
            self.loss_patch += self.calc_content_loss(self.tpF[layer],
                                                      self.tt_cropF[layer])
        self.loss_patch = paddle.clip(self.loss_patch, 1e-5, 1e5)
        self.losses['loss_patch'] =  self.loss_patch

        self.loss_content_p = 0
        for layer in self.content_layers:
            self.loss_content_p += self.calc_content_loss(self.tpF[layer],
                                                      self.cpF[layer],
                                                      norm=True)
        self.losses['loss_content_p'] = self.loss_content_p
        """style loss"""

        self.loss_ps = 0
        for layer in self.style_layers:
            self.loss_ps += self.calc_style_loss(self.tpF[layer], self.spCrop[layer])
        self.losses['loss_ps'] = self.loss_ps

        """IDENTITY LOSSES"""
        self.Ipcc,_ = self.nets['net_dec'](self.cpF, self.cpF, self.cpF,'thumb')
        self.l_identity3 = self.calc_content_loss(self.Ipcc, self.cp)
        self.Fpcc = self.nets['net_enc'](self.Ipcc)
        self.l_identity4 = 0
        for layer in self.content_layers:
            self.l_identity4 += self.calc_content_loss(self.Fpcc[layer],
                                                       self.cpF[layer])
        self.visual_items['content_identity']=self.Icc
        self.losses['l_identity1'] = self.l_identity1
        self.losses['l_identity2'] = self.l_identity2
        self.losses['l_identity3'] = self.l_identity3
        self.losses['l_identity4'] = self.l_identity4

        """relative loss"""

        self.p_loss_style_remd = self.calc_style_emd_loss(
            self.tpF['r31'], self.tt_cropF['r31']) + self.calc_style_emd_loss(self.tpF['r41'], self.tt_cropF['r41'])
        self.p_loss_content_relt = self.calc_content_relt_loss(
            self.tpF['r31'], self.cpF['r31']) + self.calc_content_relt_loss(
                self.tpF['r41'], self.cpF['r41'])
        self.losses['p_loss_style_remd'] = self.p_loss_style_remd
        self.losses['p_loss_content_relt'] = self.p_loss_content_relt
        self.losses['p_loss_content_relt'] = self.p_loss_content_relt

        self.loss = self.loss_ps * self.style_weight*1.5 + self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight * 40 +\
                    self.l_identity3 * 55 + self.l_identity4 * 1 +\
                    self.p_loss_style_remd * 20 + self.p_loss_content_relt * 32
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

        self.sp = paddle.to_tensor(input['sp'])
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.cp = paddle.to_tensor(input['cp'])
        self.visual_items['cp'] = self.cp
        self.sp = paddle.to_tensor(input['sp'])
        self.position = input['position']

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
        cpF = self.nets['net_enc'](self.pyr_cp[1])
        self.spCrop = self.nets['net_enc'](self.sp)

        stylized_small,_ = self.nets['net_dec'](cF, sF, cpF,'thumb')
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        p_stylized_small,_ = self.nets['net_dec'](cF, sF, cpF,'patch')
        self.visual_items['p_stylized_small'] = p_stylized_small
        p_stylized_up = F.interpolate(p_stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap,self.stylized_thumb_feat = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])

        p_revnet_input = paddle.concat(x=[self.pyr_cp[0], p_stylized_up], axis=1)
        p_stylized_rev_lap,_ = self.nets['net_rev'](p_revnet_input,self.stylized_thumb_feat)
        p_stylized_rev = fold_laplace_pyramid([p_stylized_rev_lap, p_stylized_small])

        self.stylized = stylized_rev
        self.p_stylized = p_stylized_rev
        self.visual_items['stylized'] = self.stylized
        self.visual_items['p_stylized'] = self.p_stylized

    def backward_G(self, optimizer):

        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)
        self.cpF = self.nets['net_enc'](self.cp)

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

        self.loss = self.loss_s * self.style_weight +\
                    self.loss_content * self.content_weight+\
                    self.loss_style_remd * 10 +\
                    self.loss_content_relt * 16
        self.loss.backward()
        optimizer.step()

        """patch loss"""
        self.loss_patch = 0
        # self.loss_patch= self.calc_content_loss(self.tpF['r41'],self.tt_cropF['r41'])#+\
        #                self.calc_content_loss(self.tpF['r51'],self.tt_cropF['r51'])
        for layer in [self.content_layers[3]]:
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
            self.loss_ps += self.calc_style_loss(self.tpF[layer], self.spCrop[layer])
        self.losses['loss_ps'] = self.loss_ps

        self.p_loss_style_remd = self.calc_style_emd_loss(
            self.tpF['r31'], self.tt_cropF['r31']) + self.calc_style_emd_loss(
            self.tpF['r41'], self.tt_cropF['r41'])
        self.p_loss_content_relt = self.calc_content_relt_loss(
            self.tpF['r31'], self.cpF['r31']) + self.calc_content_relt_loss(
            self.tpF['r41'], self.cpF['r41'])
        self.losses['p_loss_style_remd'] = self.p_loss_style_remd
        self.losses['p_loss_content_relt'] = self.p_loss_content_relt

        """gan loss"""

        pred_fake = self.nets['netD'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN
        pred_fake_p = self.nets['netD'](self.p_stylized)
        self.loss_Gp_GAN = self.gan_criterion(pred_fake_p, True)
        self.losses['loss_gan_Gp'] = self.loss_Gp_GAN

        self.loss = self.loss_G_GAN + self.loss_Gp_GAN + self.loss_ps * self.style_weight *1.5 +\
                    self.loss_content_p * self.content_weight +\
                    self.loss_patch * self.content_weight * 40 +\
                    self.p_loss_style_remd * 25 + self.p_loss_content_relt * 32
        self.loss.backward()

        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_p_fake = self.nets['netD'](self.p_stylized.detach())
        self.loss_Dp_fake = self.gan_criterion(pred_p_fake, False)

        pred_real = self.nets['netD'](self.pyr_si[2])
        self.loss_D_real = self.gan_criterion(pred_real, True)
        pred_p_real = self.nets['netD'](self.sp)
        self.loss_Dp_real = self.gan_criterion(pred_p_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_Dp_fake + self.loss_Dp_real + self.loss_D_real) * 0.5

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
        self.backward_G(optimizers['optimG'])
        optimizers['optimG'].step()