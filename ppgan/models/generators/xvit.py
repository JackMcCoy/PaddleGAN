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

# pre-layernorm

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

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = paddle.concat((x, context), axis = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, axis = -1))
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

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
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

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

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

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

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
        patch_dim = 6 * patch_size ** 2

        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.to_patch_embedding =nn.Linear(patch_dim, dim)

        self.pos_embedding = paddle.create_parameter(shape=(1, num_patches*4 + 1, dim), dtype='float32')
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

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))
        #sm_decoder_layer = nn.TransformerDecoderLayer(256, 16, 256, normalize_before=True)
        lg_decoder_layer = nn.TransformerDecoderLayer(768, 16, 768, normalize_before=True)
        self.decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=2,d=16,e=16)
        self.sm_decompose_axis = Rearrange('b (h w) (e d c) -> b c (h e) (w d)',h=8,d=16,e=16)
        self.partial_unfold = Rearrange('b (h w p1) c -> b (h w) (p1 c)', w=2,h=2,
                                        p1=16)
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8)
        self.sm_project = nn.Sequential(nn.LayerNorm(sm_dim),nn.Linear(sm_dim,768))
        self.lg_project = nn.Sequential(nn.LayerNorm(lg_dim),nn.Conv2DTranspose(4,64,1))
        #self.sm_decoder_transformer = nn.TransformerDecoder(sm_decoder_layer, 6)
        self.lg_decoder_transformer = nn.TransformerDecoder(lg_decoder_layer, 6)
        self.upscale = nn.Upsample(scale_factor=4, mode='nearest')
        self.decoder = nn.Sequential(
            nn.Sigmoid()
            ResnetBlock(3),
            ConvBlock(3, 3),
            ResnetBlock(3),
            ConvBlock(3, 3),
            nn.ReLU()
        )
        self.final = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                   nn.Conv2D(3, 3, (3, 3)))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        lg_tokens = paddle.unsqueeze(lg_tokens,axis=2)
        lg_tokens = paddle.concat([self.lg_tokens[:,0,:],self.lg_project(lg_tokens[:,1:,:]],axis=1)
        lg_tokens = paddle.squeeze(lg_tokens,axis=2)
        x = sm_tokens+lg_tokens
        x = self.lg_decoder_transformer(x,x)

        x = self.sm_decompose_axis(x[:,1:,:])
        print(x.shape)
        x = self.decoder(x)
        return self.final(x)