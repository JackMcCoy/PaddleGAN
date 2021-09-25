import paddle
from paddle import nn
import paddle.nn.functional as F
from math import gcd,ceil
from collections import namedtuple
import numpy as np
from functools import partial, reduce
from einops.layers.paddle import Rearrange

from .builder import GENERATORS
from . import ResnetBlock, ConvBlock, adaptive_instance_normalization, Transformer, LinearAttentionTransformer


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

def einsum(equation, *operands):
    r"""
    Executes the sum of product of provided operands based on the Einstein summation convention.
    Einsum can be used to complete a variety of operations, such as sum, transpose,
    batch matrix multiplication.

    Args:
        equation (`str`):
            Uses uncased letters to specify the dimension of the operands and result. The input
            equation is on the left hand before `->` while the output equation is on the right side.
            Einsum can infer the result shape so that the `->` and the result label letters can be omitted.
            Operands in the input equation are splited by commas (','), e.g. 'abc,cde' describes two 3D
            operands. The dimensions labeled with same letter should be same or be 1. Ellipsis ('...') can
            be used to specify the broadcast dimensions.

        operands (`Tensor`):
            The operands to compute the Einstein sum of. The number of operands should be the same as the
            the operands described in input equation.

    Returns:
        `Tensor`: The result of Einstein sum product.
    """

    def _mul_sum(left, right, sum_dims):
        assert left.rank() == right.rank(), "number of rank should be equal."
        if len(sum_dims) == 0:
            return left * right
        sum_dims_set = set(sum_dims)
        batch_dims = []
        left_out_dims = []
        right_out_dims = []
        batch_size = summed_size = left_size = right_size = 1
        dim = len(left.shape)
        for i in range(dim):
            is_left_summed_dim = left.shape[i] > 1  # not broadcast dim
            is_right_summed_dim = right.shape[i] > 1
            if i in sum_dims_set:
                if is_left_summed_dim and is_right_summed_dim:
                    assert left.shape[i] == right.shape[
                        i], "Non-brocast dim should be equal."
                    summed_size *= left.shape[i]
                elif is_left_summed_dim:
                    left = left.sum(axis=i, keepdim=True)
                elif is_right_summed_dim:
                    right = right.sum(axis=i, keepdim=True)
            elif is_left_summed_dim and is_right_summed_dim:
                assert left.shape[i] == right.shape[
                    i], "Non-brocast dim should be equal."
                batch_dims.append(i)
                batch_size *= left.shape[i]
            elif is_left_summed_dim:
                left_out_dims.append(i)
                left_size *= left.shape[i]
            else:
                right_out_dims.append(i)
                right_size *= right.shape[i]
        out_shape = [left.shape[i] for i in batch_dims + left_out_dims]
        out_shape.extend([1] * len(sum_dims))
        out_shape.extend([right.shape[i] for i in right_out_dims])

        left_perm = list(batch_dims)
        left_perm.extend(left_out_dims)
        left_perm.extend(sum_dims)
        left_perm.extend(right_out_dims)

        right_perm = list(batch_dims)
        right_perm.extend(sum_dims)
        right_perm.extend(right_out_dims)
        right_perm.extend(left_out_dims)

        output_perm = [-1] * (len(batch_dims) + len(left_out_dims) +
                              len(sum_dims) + len(right_out_dims))
        for i, j in enumerate(batch_dims + left_out_dims + sum_dims +
                              right_out_dims):
            output_perm[j] = i

        left = paddle.reshape(
            paddle.transpose(
                left, perm=left_perm), (batch_size, left_size, summed_size))
        right = paddle.reshape(
            paddle.transpose(
                right, perm=right_perm), (batch_size, summed_size, right_size))
        result = paddle.matmul(left, right)
        result = paddle.reshape(result, out_shape)
        result = paddle.transpose(result, output_perm)
        return result

    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    # Equation is case insensitive
    num_letters = 26
    letters_to_idx = [-1] * num_letters
    equation = equation.lower().replace(' ', '')
    # 1. Parse the equation
    eqns = equation.split("->")
    num_eqns_size = len(eqns)
    assert num_eqns_size <= 2, "The '->' should exist at most only once"

    input_eqn = eqns[0]
    output_eqn = None if num_eqns_size <= 1 else eqns[1]
    operand_eqns = input_eqn.split(",")
    assert len(operand_eqns) == len(
        operands
    ), "Number of operands in equation and the tensors provided should be equal."

    # Parse input equation
    num_total_idxes = 0
    input_operand_idxes = []
    letter_frequence = [0] * num_letters
    idxes_last_operand = []
    num_ell_idxes = -1
    first_ell_idx = 0
    for i, term in enumerate(operand_eqns):
        ell_char_count = 0
        operand_rank = int(operands[i].rank().numpy())
        curr_num_ell_idxes = operand_rank - len(term) + 3
        dims_in_terms = 0
        curr_operand_idxes = []
        for ch in term:
            if ch == '.':
                ell_char_count += 1
                assert ell_char_count <= 3, "The '.' should only exist in one ellispis '...' in term {}".format(
                    term)
                if ell_char_count == 3:
                    if num_ell_idxes == -1:
                        num_ell_idxes = curr_num_ell_idxes
                        first_ell_idx = num_total_idxes
                        num_total_idxes += num_ell_idxes
                    else:
                        assert curr_num_ell_idxes == num_ell_idxes, "Ellispis in all terms should represent same dimensions ({}).".format(
                            num_ell_idxes)

                    for j in range(num_ell_idxes):
                        curr_operand_idxes.append(j + first_ell_idx)
                        idxes_last_operand.append(i)
                    dims_in_terms += num_ell_idxes
            else:
                assert (
                    (ell_char_count == 0) or (ell_char_count == 3)
                ), "'.' must only occur in ellipsis, operand {}".format(term)
                assert (ord('a') <= ord(ch) and
                        ord(ch) <= ord('z')), "only accept alphabet (a-zA-Z)"
                letter_num = ord(ch) - ord('a')
                if letters_to_idx[letter_num] == -1:
                    letters_to_idx[letter_num] = num_total_idxes
                    num_total_idxes += 1
                    idxes_last_operand.append(i)
                else:
                    idxes_last_operand[letters_to_idx[letter_num]] = i
                letter_frequence[letter_num] += 1
                curr_operand_idxes.append(letters_to_idx[letter_num])
                dims_in_terms += 1

        assert dims_in_terms == operand_rank, "Dimension dismatch for operand {}: equation {}, tensor {}".format(
            i, dims_in_terms, operand_rank)
        input_operand_idxes.append(curr_operand_idxes)
    # Parse output equation
    idxes_to_output_dims = [-1] * num_total_idxes
    num_output_dims = 0
    if num_eqns_size == 2:
        ell_char_count = 0
        for ch in output_eqn:
            if ch == '.':
                ell_char_count += 1
                assert ell_char_count <= 3, "The '.' should only exist in one ellispis '...' in term {}".format(
                    output_eqn)
                if ell_char_count == 3:
                    assert num_ell_idxes > -1, "Input equation '{}' don't have ellispis.".format(
                        input_eqn)
                    for j in range(num_ell_idxes):
                        idxes_to_output_dims[first_ell_idx +
                                             j] = num_output_dims
                        num_output_dims += 1

            else:
                assert ((ell_char_count == 0) or (ell_char_count == 3)
                        ), "'.' must only occur in ellipsis, operand {}".format(
                            output_eqn)
                assert (ord('a') <= ord(ch) and
                        ord(ch) <= ord('z')), "only accept alphabet (a-zA-Z)"
                letter_num = ord(ch) - ord('a')
                assert letters_to_idx[
                    letter_num] != -1, "character {} doesn't exist in input".format(
                        ch)
                assert idxes_to_output_dims[letters_to_idx[
                    letter_num]] == -1, "character {} occurs twice in output".format(
                        ch)

                idxes_to_output_dims[letters_to_idx[
                    letter_num]] = num_output_dims
                num_output_dims += 1
    else:  #  num_eqns_size == 1
        # Infer the output dims
        if num_ell_idxes >= 0:
            for j in range(num_ell_idxes):
                idxes_to_output_dims[first_ell_idx + j] = num_output_dims
                num_output_dims += 1
        for j in range(num_letters):
            if letter_frequence[j] == 1:
                idxes_to_output_dims[letters_to_idx[j]] = num_output_dims
                num_output_dims += 1

    # Mark sum index
    sum_dim = num_output_dims
    for i in range(num_total_idxes):
        if idxes_to_output_dims[i] == -1:
            idxes_to_output_dims[i] = sum_dim
            sum_dim += 1

    preprocessed_operands = []
    size_dims = [-1] * num_total_idxes
    for i, preprocessed_operand in enumerate(operands):
        idx_to_dims = [-1] * num_total_idxes
        curr_operand_idxes = input_operand_idxes[i]
        dim = 0
        for j, idx in enumerate(curr_operand_idxes):
            output_dim = idxes_to_output_dims[idx]
            if idx_to_dims[output_dim] == -1:
                idx_to_dims[output_dim] = dim
                if size_dims[idx] == -1:
                    size_dims[idx] = preprocessed_operand.shape[dim]
                else:
                    assert size_dims[idx] == preprocessed_operand.shape[
                        dim], "Dimension size does not match previous size. "
                dim += 1
            else:
                # Diagonal repeated index
                # TODO(zhoushunjie): Need to develop a paddle.diagonal api
                raise NotImplementedError("Can't support diagonal.")
        perm = []
        for input_dim in idx_to_dims:
            if input_dim > -1:
                perm.append(input_dim)
        # Transpose the tensor by perm
        preprocessed_operand = paddle.transpose(preprocessed_operand, perm=perm)

        for dim, input_dim in enumerate(idx_to_dims):
            if input_dim == -1:
                preprocessed_operand = paddle.unsqueeze(preprocessed_operand,
                                                        dim)

        preprocessed_operands.append(preprocessed_operand)

    # 2. Execute the mul_sum
    sum_dims = []
    result = preprocessed_operands[0]
    for i in range(num_total_idxes):
        if idxes_last_operand[i] == 0 and idxes_to_output_dims[
                i] >= num_output_dims:
            result = result.sum(axis=idxes_to_output_dims[i], keepdim=True)
    for i in range(1, len(preprocessed_operands)):
        for j in range(num_total_idxes):
            if idxes_last_operand[j] == i and idxes_to_output_dims[
                    j] >= num_output_dims:
                sum_dims.append(idxes_to_output_dims[j])
        result = _mul_sum(result, preprocessed_operands[i], sum_dims)

    squeeze_dims = [
        i for i in range(len(result.shape) - 1, num_output_dims - 1, -1)
    ]
    if len(squeeze_dims) != 0:
        result = paddle.squeeze(result, squeeze_dims)
    return result

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ema_inplace(moving_avg, new, decay):
    moving_avg.add_((decay * moving_avg)-moving_avg).add_(new * (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


class VectorQuantize(nn.Layer):
    def __init__(
        self,
        dim,
        codebook_size,
        transformer_size,
        decay = 0.8,
        commitment = 1.,
        eps = 1e-5,
    ):
        super().__init__()

        self.dim = dim
        self.n_embed = codebook_size
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        self.perceptual_loss = CalcContentLoss()
        self.transformer_size = transformer_size

        embed = paddle.randn((dim, self.n_embed))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', paddle.zeros(shape=(self.n_embed,)))
        self.register_buffer('embed_avg', embed.clone())
        if self.n_embed != 1280:
            self.rearrange = Rearrange('b c h w -> b (h w) c')
            self.decompose_axis = Rearrange('b (h w) c -> b c h w',h=dim)
        else:
            self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',p1=4,p2=4)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,w=16, e=4,d=4)

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

        loss = self.perceptual_loss(quantize.detach(), input, norm=True) * self.commitment

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

        self.quantize_4 = VectorQuantize(16, 640, 1)
        self.quantize_3 = VectorQuantize(32, 640, 2)
        self.quantize_2 = VectorQuantize(64, 1280, 3)
        self.vit = Transformer(192, 8, 16, 256, 192, dropout=0.05, shift_tokens= True)

        patch_height, patch_width = (8,8)

        num_patches = (128 // patch_height) * (128 // patch_width)
        patch_dim = 3 * patch_height * patch_width

        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width)
        self.decompose_axis=Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,d=8,e=8)
        self.to_patch_embedding = nn.Linear(256, 192)

        self.pos_embedding = nn.Embedding(256, 192)


        self.resblock_41 = ResnetBlock(512)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_31 = ConvBlock(256, 128)


        self.convblock_21 = ConvBlock(128, 128)
        self.convblock_22 = ConvBlock(128, 64)
        self.convblock_11 = ConvBlock(64, 64)

        self.downsample = nn.Upsample(scale_factor=.5, mode='nearest')
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.transformer_res = ResnetBlock(3)
        self.transformer_conv = ConvBlock(3, 3)
        self.transformer_relu = nn.ReLU()
        #self.skip_connect_weight = paddle.create_parameter(shape=(1, ), dtype='float32', is_bias=True)

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))


'''
      if transformer_size==1:
            self.transformer = Transformer(dim**2*2, 8, 16, 64, dim**2*2, dropout=0.1)
            self.pos_embedding = nn.Embedding(256, 512)
        elif transformer_size==2:
            self.transformer = Transformer(256, 8, 16, 64, 256, dropout=0.05)
            self.pos_embedding = nn.Embedding(1024, 256)
        elif transformer_size==3:
            self.transformer = Transformer(2048, 8, 16, 64, 768, dropout=0.05)
            self.pos_embedding = nn.Embedding(256, 2048)
        elif transformer_size==4:
            self.transformer = Transformer(1024, 1, 8, 64, 64, dropout=0.05)
            self.pos_embedding = nn.Embedding(256, 1024)
            self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = 4, p2 = 4)
            self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=32,w=32, e=4,d=4)
'''
'''
if self.transformer_size != 4:
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
        else:
            quantize = input
'''

@GENERATORS.register()
class VQGAN(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, vgg_state_dict):
        super(VQGAN, self).__init__()

        self.context_mod = VGG(vgg_state_dict)
        self.z_mod = VGG(vgg_state_dict)

        self.quantize_4_z = VectorQuantize(16, 8192, 1)
        self.quantize_4_s = VectorQuantize(16, 8192, 1)
        #self.quantize_3_z = VectorQuantize(256, 1024, 2)
        #self.quantize_3_c = VectorQuantize(256, 1024, 2)
        #self.quantize_2_z = VectorQuantize(256, 2048, 3)
        #self.quantize_2_c = VectorQuantize(256, 2048, 3)
        self.transformer_4 = Transformer(16, 8, 16, 256, 16, dropout=0.1, shift_tokens = True)
        #self.transformer_3 = Transformer(1024, 8, 16, 256, 1024, dropout=0.1, shift_tokens = True)
        #self.transformer_2 = Transformer(2048, 8, 16, 256, 2048, dropout=0.1, shift_tokens = True)

        '''
        patch_height, patch_width = (8,8)

        num_patches = (128 // patch_height) * (128 // patch_width)
        patch_dim = 3 * patch_height * patch_width

        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width)
        self.decompose_axis=Rearrange('b (h w) (c e d) -> b c (h e) (w d)',h=16,d=8,e=8)
        self.to_patch_embedding = nn.Linear(256, 192)

        self.pos_embedding = nn.Embedding(256, 192)
        '''

        self.resblock_41 = ResnetBlock(512)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_31 = ConvBlock(256, 128)


        self.convblock_21 = ConvBlock(128, 128)
        self.convblock_22 = ConvBlock(128, 64)
        self.convblock_11 = ConvBlock(64, 64)

        self.downsample = nn.Upsample(scale_factor=.5, mode='nearest')
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.transformer_res = ResnetBlock(3)
        self.transformer_conv = ConvBlock(3, 3)
        self.transformer_relu = nn.ReLU()
        #self.skip_connect_weight = paddle.create_parameter(shape=(1, ), dtype='float32', is_bias=True)

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))


    def forward(self, ci, si):
        zF = self.z_mod(ci)
        sF = self.context_mod(si)

        quant_z, z4_info, loss1 = self.quantize_4_z(zF['r41'])
        quant_s, s4_info, loss2 = self.quantize_4_s(sF['r41'])
        target = z_indices
        map_loss = loss1+loss2
        b, n, h, w = s_indices.shape
        zs = paddle.concatenate([s_indices,z_indices],axis=1).reshape((b,n*2,-1))
        logits = self.transformer_4(zs[:, :-1])
        logits = logits[:, s_indices.shape[1]-1:]

        out = self.resblock_41(logits.reshape((b,n,h,w)))
        out = self.convblock_41(out)

        upscale_4 = self.upsample(out)

        out += adaptive_instance_normalization(zF['r31'], sF['r31'])

        out = upscale_4
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(zF['r21'], sF['r21'])

        out = self.convblock_21(out)
        out = self.convblock_22(out)
        out = self.upsample(out)

        out = self.convblock_11(out)
        out = self.final_conv(out)

        return out, map_loss, target, logits

@GENERATORS.register()
class PretrainedGenerator(nn.Layer):
    def __init__(self, label):
        super(PretrainedGenerator, self).__init__()

        self.DownBlock = nn.Sequential(
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            (label+'_conv1', nn.Conv2D(6, 128, (3, 3))),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            (label+'_conv2', nn.Conv2D(128, 128, (3, 3))),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            (label+'_conv3', nn.Conv2D(128, 64, (3, 3))),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            (label+'conv4', nn.Conv2D(64, 64, (3, 3), stride=2)),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.DownBlock(input)
        return x

@GENERATORS.register()
class QuantizedRev(nn.Layer):
    def __init__(self):
        super(QuantizedRev, self).__init__()
        self.DownBlock = nn.Sequential(
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(6, 64, (3, 3)),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU()
        )

        self.resblock = ResnetBlock(64)

        self.UpBlock = nn.Sequential(
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(3, 64, (3, 3)),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU()
        )

        self.vit = Transformer(192, 8, 16, 256, 192, dropout=0.05, shift_tokens=True)
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_height,
                                   p2=patch_width)
        self.decompose_axis = Rearrange('b (h w) (c e d) -> b c (h e) (w d)', h=16, d=8, e=8)
        self.to_patch_embedding = nn.Linear(256, 192)

        self.pos_embedding = nn.Embedding(256, 192)

        self.transformer_res = ResnetBlock(3)
        self.transformer_conv = ConvBlock(3, 3)
        self.transformer_relu = nn.ReLU()

        self.ones = paddle.ones((1, 256), dtype="int64")
        self.seq_length = paddle.cumsum(self.ones, axis=1)
        self.position_ids = self.seq_length - self.ones
        self.position_ids.stop_gradient = True

    def forward(self, input):

        position_embeddings = self.pos_embedding(self.position_ids)
        input = self.rearrange(input)
        input = input + position_embeddings
        input = self.vit(input)
        input = self.decompose_axis(input)
        input = self.transformer_res(input)
        input = self.transformer_conv(input)
        input = self.DownBlock(input)


class VGG(nn.Layer):
    """Encoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, state_dict):
        super(VGG, self).__init__()
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

        vgg_net.set_dict(state_dict)
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