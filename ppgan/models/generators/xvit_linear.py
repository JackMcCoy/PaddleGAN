import paddle
from paddle import nn
import paddle.nn.functional as F
from math import gcd,ceil
from collections import namedtuple
import numpy as np
from functools import partial, reduce

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .builder import GENERATORS

# helpers

TOKEN_SELF_ATTN_VALUE = -5e4

def exists(val):
    return val is not None

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def default(val, d):
    return val if exists(val) else d

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, [amount, -amount],data_format='NCL', value = 0.)

class PreShiftTokens(nn.Layer):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, axis = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = paddle.concat((*segments_to_shift, *rest), axis = -1)
        return self.fn(x, **kwargs)

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = paddle.unbind(x, axis = -1)
    x = paddle.stack((-x2, x1), axis = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotory_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = paddle.unbind(sinu_pos,axis = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), data_format='NCL',value=value)

def look_around(x, cls=None,backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, [*dims[:-2],backward,forward], data_format='NCL' if len(x.shape)==3 else 'NCHW',value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), :] for ind in range(forward + backward + 1)]
    tensors=paddle.concat(tensors, axis=dim)
    if not cls is None:
        tensors = paddle.concat([cls,tensors],axis=2)
    return tensors

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def max_neg_value(tensor):
    return -np.finfo(np.float32).max

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def layer_drop(layers, prob):
    to_drop = paddle.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

class SequentialSequence(nn.Layer):
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x

class SinusoidalEmbeddings(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (paddle.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, )

        freqs = paddle.multiply(paddle.reshape(t,(1,-1)), paddle.reshape(self.inv_freq,(-1,1)))
        emb = paddle.concat((freqs, freqs), axis=-1)
        return emb[None, :, :]

class Chunk(nn.Layer):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = paddle.chunk(x, self.chunks, axis = self.dim)
        return paddle.concat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)


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

class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FoldAxially(nn.Layer):
    def __init__(self, axial_dim, fn):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim
    def forward(self, x, input_mask = None, **kwargs):
        b, t, d, ax = *x.shape, self.axial_dim
        x = paddle.transpose(x.reshape((b, -1, ax, d)),(0, 2, 1)).reshape(b * ax, -1, d)

        mask = None
        if exists(input_mask):
            mask = paddle.transpose(input_mask.reshape((b, -1, ax)),(0,2,1)).reshape(b * ax, -1)

        x = self.fn(x, input_mask = mask, **kwargs)
        x = paddle.transpose(x.reshape((b, ax, -1, d)),(0,2,1)).reshape((b, t, d))
        return x

# feedforward

class FeedForward(nn.Layer):
    def __init__(self, dim, mult = 2, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = paddle.chunk(self.w1(x),2, axis=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# attention

def linear_attn(q, k, v, kv_mask = None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask
    soft_q=nn.Softmax(axis=-1)
    soft_k=nn.Softmax(axis=-2)
    q = soft_q(q)
    k = soft_k(k)

    q = q * dim ** -0.5

    context = paddle.matmul(paddle.transpose(k,(0,1,3,2)), v)
    attn = paddle.matmul(q, context)
    return attn.reshape(q.shape)

class LocalAttention(nn.Layer):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 0,
        look_forward = 0,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.exact_windowsize = exact_windowsize
        self.autopad = autopad

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        self.rel_pos = None
        if exists(rel_pos_emb_config) or exists(dim):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]
                print(dim)
            print('dim'+str(dim))
            self.rel_pos = SinusoidalEmbeddings(dim)

    def forward(self, q, k, v, input_mask = None):
        shape = q.shape
        v_cls=v[:,:,0]
        q_cls=q[:,:,0]
        k_cls=q[:,:,0]
        merge_into_batch = lambda t: t.reshape((-1, t.shape[-2],t.shape[-1]))

        q, k, v,v_cls,k_cls = map(merge_into_batch, (q[:,:,1:], k[:,:,1:], v[:,:,1:],v_cls,k_cls))
        if exists(self.rel_pos):
            pos_emb = self.rel_pos(q)
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        if self.autopad:
            orig_t = q.shape[1]
            q, k, v = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        window_size, causal, look_backward, look_forward, shared_qk = self.window_size, self.causal, self.look_backward, self.look_forward, self.shared_qk
        b, t, e = q.shape
        assert (t % window_size) == 0, f'sequence length {t} must be divisible by window size {window_size} for local attention'

        windows = t // window_size

        if shared_qk:
            k = F.normalize(k, 2, axis=-1)

        ticker = paddle.reshape(paddle.arange(t),(1,-1))
        b_t = ticker.reshape((1, windows, window_size))

        bucket_fn = lambda t: t.reshape((b, windows, window_size, -1))
        bq, bk, bv = map(bucket_fn, (q, k, v))
        cls_fn = lambda t: t.reshape((b, 1, 1, -1))
        v_cls,k_cls = map(cls_fn, (v_cls,k_cls))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk,cls=k_cls, **look_around_kwargs)
        bv = look_around(bv,cls=v_cls, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t,cls=paddle.ones((k_cls.shape)) **look_around_kwargs)

        dots = paddle.matmul(bq, paddle.transpose(bk,(0,1,3,2))) * (e ** -0.5)

        mask_value = max_neg_value(dots)
        a,b,c,d = bq_t.shape
        reshaped_bq_t = paddle.reshape(bq_t,(a,b,c,1,d))
        a,b,c,d = bq_k.shape
        reshaped_bq_k = paddle.reshape(bq_k,(a,b,c,1,d))
        if shared_qk:
            mask = reshaped_bq_t == reshape_bq_k
            dots[mask] = TOKEN_SELF_ATTN_VALUE
            del mask

        if causal:
            mask = reshaped_bq_t < reshape_bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                mask = mask | (reshaped_bq_t > (reshape_bq_k + max_causal_window_size))

            dots[mask] = mask_value
            del mask

        mask = reshaped_bq_k == -1
        assert(paddle.any(mask)==False)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            if self.autopad:
                input_mask = pad_to_multiple(input_mask, window_size, dim=-1, value=False)
            input_mask = input_mask.reshape((-1, windows, window_size))
            mq = mk = input_mask
            a,b,c,d = mq.shape
            reshaped_mq = paddle.reshape(mq,(a,b,c,1,d))
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            a,b,c,d = mk.shape
            reshaped_mk = paddle.reshape(mk,(a,b,c,1,d))
            mask = (reshaped_mq * reshaped_mk)
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots[~mask]= mask_value
            del mask

        attn = nn.Softmax(dots)
        attn = self.dropout(attn)

        out = paddle.matmul(attn, bv)
        out = out.reshape((-1, t, e))

        if self.autopad:
            out = out[:, :orig_t, :]

        return out.reshape((*shape))

class SelfAttention(nn.Layer):
    def __init__(self, dim, heads, causal = False, dim_head = None, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128, receives_context = False, dropout = 0., attn_dropout = 0.):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads
        self.receives_context = receives_context

        self.global_attn_heads = heads - n_local_attn_heads
        self.global_attn_fn = linear_attn

        self.to_q = nn.Linear(dim, d_heads * heads, bias_attr = False)

        kv_heads = heads

        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias_attr = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias_attr = False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None, pos_emb = None, **kwargs):
        assert not (self.receives_context and not exists(context)), 'context must be supplied if self attention is in receives context mode'

        if not self.receives_context:
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        merge_heads = lambda x: paddle.transpose(x.reshape((*x.shape[:2], -1, dh)),(0,2,1,3))

        q, k, v = map(merge_heads, (q, k, v))

        if exists(pos_emb) and not self.receives_context:
            q, k = apply_rotory_pos_emb(q, k, pos_emb)

        out = []

        kv_mask = input_mask if not self.receives_context else context_mask
        global_out = self.global_attn_fn(q, k, v, kv_mask = kv_mask)
        out.append(global_out)

        attn = paddle.concat(out, axis=1)
        attn = paddle.transpose(attn,(0,2,1,3)).reshape((b, t, -1))
        return self.dropout(self.to_out(attn))

class LinearAttentionTransformer(nn.Layer):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        dim_head = None,
        bucket_size = 64,
        causal = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        reversible = False,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 0,
        receives_context = False,
        attend_axially = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        shift_tokens = False
    ):
        super().__init__()

        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.LayerList()

        for ind, local_heads in zip(range(depth), n_local_attn_heads):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)

            parallel_net = Chunk(ff_chunks, FeedForward(dim), along_dim = 1)

            attn = SelfAttention(dim, heads, causal, dim_head = dim_head, blindspot_size = blindspot_size, n_local_attn_heads = local_heads, local_attn_window_size = local_attn_window_size, dropout = attn_layer_dropout, attn_dropout= attn_dropout)

            if shift_tokens:
                shifts = (1, 0, -1) if not causal else (1, 0)
                attn, parallel_net = map(partial(PreShiftTokens, shifts), (attn, parallel_net))

            layers.append(nn.LayerList([
                PreNorm(dim, attn),
                PreNorm(dim, parallel_net)
            ]))

            if attend_axially:
                layers.append(nn.LayerList([
                    PreNorm(dim, FoldAxially(local_attn_window_size, SelfAttention(dim, heads, causal, dropout = attn_layer_dropout, attn_dropout= attn_dropout))),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

            if receives_context:
                attn = SelfAttention(dim, heads, dim_head = dim_head, dropout = attn_layer_dropout, attn_dropout= attn_dropout, receives_context = True)

                layers.append(nn.LayerList([
                    PreNorm(dim, attn),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

        #TODO: ReversibleSequence
        execute_type = SequentialSequence

        axial_layer = ((True, False),) if attend_axially else tuple()
        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *axial_layer, *attn_context_layer) * depth
        route_context = ((False, False), *axial_layer, *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn, 'pos_emb': route_attn}
        self.layers = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        self.pad_to_multiple = lcm(
            1 if not causal else blindspot_size,
            1 if all([(h == 0) for h in n_local_attn_heads]) else local_attn_window_size
        )

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)


class Attention(nn.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(axis = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias_attr = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias_attr = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x, style=None,context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if style is None:
            style=context
        if kv_include_self:
            context = paddle.concat((x, style), axis = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(style).chunk(2, axis = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = paddle.matmul(q, paddle.transpose(k,(0,1,3,2))) * self.scale

        attn = self.attend(dots)

        out = paddle.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.LayerList()
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, style=None):
        for attn, ff in self.layers:
            x = attn(x,style=style) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Layer):
    def Identity(self,input):
        return input
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else self.Identity
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else self.Identity

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Layer):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, SelfAttention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, SelfAttention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens, sm_style=None,lg_style=None):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_style, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_style, kv_include_self = True) + lg_cls

        sm_tokens = paddle.concat((sm_cls, sm_patch_tokens), axis = 1)
        lg_tokens = paddle.concat((lg_cls, lg_patch_tokens), axis = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Layer):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        receives_context=False,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.LayerList()
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                LinearAttentionTransformer(sm_dim, receives_context=receives_context, **sm_enc_params),
                LinearAttentionTransformer(lg_dim, receives_context=receives_context,**lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens,sm_style=None,lg_style=None):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens,context=sm_style), lg_enc(lg_tokens,context=lg_style)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens,sm_style=sm_style,lg_style=lg_style)

        return sm_tokens, lg_tokens

# patch-based image to token embedder

class ImageEmbedder(nn.Layer):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.to_patch_embedding =nn.Linear(patch_dim, dim)

        self.pos_embedding = paddle.create_parameter(shape=(1, num_patches + 1, dim), dtype='float32')
        self.cls_token = paddle.create_parameter(shape=(1, 1, dim), dtype='float32')
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, img):
        x = self.rearrange(img)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = paddle.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# cross ViT class
@GENERATORS.register()
class LinearCrossViT(nn.Layer):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)
        self.sm_image_embedder_style = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder_style = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                dim_head = sm_enc_dim_head,
                local_attn_window_size = 4
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                dim_head = lg_enc_dim_head,
                local_attn_window_size=2
            ),
            dropout = dropout
        )
        self.multi_scale_encoder_style = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                dim_head = sm_enc_dim_head,
                local_attn_window_size=4
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                dim_head = lg_enc_dim_head,
                local_attn_window_size=2
            ),
            dropout = dropout
        )
        self.multi_scale_decoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            receives_context = True,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                dim_head = sm_enc_dim_head,
                local_attn_window_size=4
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                dim_head = lg_enc_dim_head,
                local_attn_window_size=2
            ),
            dropout = dropout
        )

        #sm_decoder_layer = nn.TransformerDecoderLayer(256, 16, 256, normalize_before=True)

        self.decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=2,d=4,e=4)
        self.sm_decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=8,d=4,e=4)
        self.partial_unfold = Rearrange('b (h w p1) c -> b (h w) (p1 c)', w=2,h=2,
                                        p1=16)
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8)
        self.lg_project = nn.Sequential(nn.LayerNorm(lg_dim),nn.Conv2DTranspose(4,256,1,groups=4))
        #self.sm_decoder_transformer = nn.TransformerDecoder(sm_decoder_layer, 6)
        self.upscale = nn.Upsample(scale_factor=4, mode='nearest')
        self.decoder = nn.Sequential(
            nn.Sigmoid(),
            ResnetBlock(48),
            ConvBlock(48, 24),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResnetBlock(24),
            ConvBlock(24, 12),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResnetBlock(12),
            ConvBlock(12, 6)
        )
        self.final = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                   nn.Conv2D(6, 3, (3, 3)))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img[:,:3,:,:])
        lg_tokens = self.lg_image_embedder(img[:,:3,:,:])
        sm_tokens_style = self.sm_image_embedder_style(img[:,3:,:,:])
        lg_tokens_style = self.lg_image_embedder_style(img[:,3:,:,:])

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)
        sm_tokens_style, lg_tokens_style = self.multi_scale_encoder_style(sm_tokens_style, lg_tokens_style)

        sm_tokens, lg_tokens = self.multi_scale_decoder(sm_tokens, lg_tokens,sm_style=sm_tokens_style,lg_style=lg_tokens_style)
        lg_tokens = paddle.unsqueeze(lg_tokens,axis=2)
        lg_tokens = self.lg_project(lg_tokens[:,1:,:])
        lg_tokens = paddle.squeeze(lg_tokens,axis=2)
        print(lg_tokens.shape)
        print(sm_tokens.shape)
        x = lg_tokens+sm_tokens[:,1:,:]
        print(x.shape)
        x = self.sm_decompose_axis(x)
        x = self.decoder(x)
        print(x.shape)
        return self.final(x)