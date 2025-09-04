import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Patchembedding(nn.Module):
    def __init__(self, channels, patch_size, dim):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * channels
        
        self.to_patching = nn.Sequential(
            Rearrange('B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        x = self.to_patching(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        return self.attn(*qkv)

class multihead_attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv(x) # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1) # each (B, N, D)
        atten = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        atten = atten.softmax(dim=-1)
        atten = self.dropout(atten)
        out = torch.matmul(atten, v)
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        x = self.norm(x)
        for i, (attn, ff) in enumerate(self.layers):    
            x = attn(x)[0] + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        image_width, image_height = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        assert image_width % patch_size == 0 and image_height % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        N = (image_width // patch_size) * (image_height // patch_size)

        self.patching = Patchembedding(channels, patch_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, N+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        
    def forward(self, img):
        x = self.patching(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> (repeat) 1 d', repeat=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.dropout(x)
        x = self.transformer(x)

        if self.pool == "cls":
            x = x[:, 0]              # (B, dim) take CLS token
        else:
            x = x.mean(dim=1)        # (B, dim) average over tokens

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x