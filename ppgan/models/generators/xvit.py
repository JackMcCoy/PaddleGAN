import paddle
from paddle import nn
import paddle.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .builder import GENERATORS

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class TransformerDecoderLayer(nn.Layer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = nn.layer.transformer._convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = nn.layer.transformer._convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = nn.layer.transformer.MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.cross_attn = nn.layer.transformer.MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        tgt_mask = nn.layer.transformer._convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = nn.layer.transformer._convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory,
                                                memory_mask, cache[1])
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(
            memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache

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
        self.norm = nn.InstanceNorm1D(dim*2,data_format='NCL')
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

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
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens, sm_style=None,lg_style=None):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, style=sm_style, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, style=lg_style, context = sm_patch_tokens, kv_include_self = True) + lg_cls

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
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.LayerList()
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens,sm_style=None,lg_style=None):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens,style=sm_style), lg_enc(lg_tokens,style=lg_style)
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
class CrossViT(nn.Layer):
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
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
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
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
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
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))
        #sm_decoder_layer = nn.TransformerDecoderLayer(256, 16, 256, normalize_before=True)

        self.decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=2,d=4,e=4)
        self.sm_decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=8,d=4,e=4)
        self.partial_unfold = Rearrange('b (h w p1) c -> b (h w) (p1 c)', w=2,h=2,
                                        p1=16)
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8)
        self.lg_project = nn.Sequential(nn.LayerNorm(lg_dim),nn.Conv2DTranspose(4,64,1))
        self.lg_style_project = nn.Sequential(nn.LayerNorm(lg_dim),nn.Conv2DTranspose(4,48,1,groups=4))
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
        lg_tokens = paddle.concat([paddle.unsqueeze(lg_tokens[:,0,:],axis=2),self.lg_project(lg_tokens[:,1:,:])],axis=1)
        lg_tokens = paddle.squeeze(lg_tokens,axis=2)
        x = lg_tokens+sm_tokens
        x = self.sm_decompose_axis(x[:,1:,:])
        x = self.decoder(x)
        return self.final(x)