import paddle
from paddle import nn

from einops import rearrange, repeat
from einops.layers.paddle import Rearrange
from .builder import GENERATORS

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Layer):
    def Identity(self,input):
        return input

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(axis = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr = False)

        self.to_out = nn.Sequential(
            [nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)]
        ) if project_out else self.Identity()

    def forward(self, x):
        qkv = paddle.chunk(self.to_qkv(x),3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = paddle.matmul(q, paddle.transpose(k,(0,1,3,2))) * self.scale

        attn = self.attend(dots)

        out = paddle.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class rearrange_tensors(nn.Layer):
    def __init__(self,image_height,patch_height):
        super().__init__()
        self.num_splits = int(image_height/patch_height)
    def forward(self,x):
        patches=[]
        split_x = paddle.split(x,self.num_splits,axis=2)
        for y in split_x:
            patches.extend(paddle.split(y,self.num_splits,axis=3))
        x = paddle.concat(patches,axis=1)
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
class ViT(nn.Layer):
    def Identity(self,input):
        return input

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.decompose_axis=Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=(image_width // patch_width),p1=patch_height,p2=patch_width)
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = paddle.create_parameter(shape=(1, num_patches + 1, dim), dtype='float32')
        self.cls_token = paddle.create_parameter(shape=(1, 1, dim),dtype='float32')
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        decoder_layer = nn.TransformerDecoderLayer(1024, 2, 1024,normalize_before=True)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, 2)

        dec_input = paddle.rand((5, 64, 1024))
        enc_output = paddle.rand((5, 64, 1024))
        self.pool = pool
        self.to_latent = self.Identity

        self.decoder = nn.Sequential(
            ResnetBlock(4),
            ConvBlock(4, 3),
        )
        self.final = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(3, 3, (3, 3)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.rearrange(img)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = paddle.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:,1:,:]
        x = self.decoder_transformer(x,x)
        x = self.decompose_axis(x)
        counter=0
        x = self.decoder(x)
        x = x*self.sigmoid(x)
        return self.final(x)
