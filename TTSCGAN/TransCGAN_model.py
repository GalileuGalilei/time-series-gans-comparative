import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor 
import math 
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, seq_len=150, channels=3, num_classes=9, latent_dim=100, data_embed_dim=10, 
                label_embed_dim=10 ,depth=4, num_heads=8, 
                forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        self.l1 = nn.Linear(self.latent_dim + self.label_embed_dim, self.seq_len * self.data_embed_dim)
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embed_dim) 
        
        self.blocks = Gen_TransformerEncoder(
                 depth=self.depth,
                 emb_size = self.data_embed_dim,
                 num_heads = self.num_heads,
                 drop_p = self.attn_drop_rate,
                 forward_drop_p=self.forward_drop_rate
                )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.data_embed_dim, self.channels, 1, 1, 0)
        )

        self.batchNorm = nn.BatchNorm1d(self.seq_len * self.data_embed_dim)
        self.batchNorm2d = nn.BatchNorm2d(1)
        
    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        x = self.l1(x)
        x = self.batchNorm(x)
        x = x.view(-1, self.seq_len, self.data_embed_dim)
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.batchNorm2d(x)
        output = self.deconv(x.permute(0, 3, 1, 2))
        return output 


class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.1,
                 forward_expansion=4,
                 forward_drop_p=0.1):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=3, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)]) 
        

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

     

class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=8,
                 drop_p=0.,
                 forward_expansion=2,
                 forward_drop_p=0.1):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, 4, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=4, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, adv_classes=2, cls_classes=10):
        super().__init__()
        self.adv_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            MiniBatch(emb_size, 9, 4),
            nn.Linear(emb_size + 9 + 1, adv_classes) #mini batch(9) + std(1)
        )
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, cls_classes)
        )

    def forward(self, x):
        out_adv = self.adv_head(x)
        out_cls = self.cls_head(x)
        return out_adv, out_cls

    
class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels = 21, patch_size = 16, emb_size = 100, seq_length = 1024):
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
            #nn.Linear(patch_size, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))


    def forward(self, x:Tensor) ->Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x    
        
class MiniBatch(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        """
        in_features: Dimensão da feature de entrada (ex: emb_size)
        out_features: Número de kernels (dimensão de saída do minibatch)
        kernel_dims: Tamanho dos vetores latentes para comparação
        """
        super(MiniBatch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        # Tensor de peso T para projeção: shape (in_features, out_features, kernel_dims)
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # x: (batch, in_features)
        M = x @ self.T.view(self.in_features, -1)  # (batch, out_features * kernel_dims)
        M = M.view(-1, self.out_features, self.kernel_dims)  # (batch, out_features, kernel_dims)

        # Calcula a distância L1 entre os pares de amostras
        batch_size = M.size(0)
        out = []

        for i in range(batch_size):
            diff = M[i].unsqueeze(0) - M  # (batch, out_features, kernel_dims)
            diff = torch.abs(diff).sum(2)  # Soma ao longo de kernel_dims → (batch, out_features)
            exp = torch.exp(-diff)  # Similaridade invertida
            out.append(exp.sum(0) - 1)  # Remove self-similarity

        # Resulta em (batch, out_features)
        similarity_features = torch.stack(out)

        # --------- Adiciona desvio padrão como feature adicional ----------
        std = torch.std(x, dim=0, keepdim=True)  # (1, in_features)
        mean_std = std.mean(dim=1, keepdim=True)  # (1, 1)
        std_feature = mean_std.expand(x.size(0), 1)  # (batch, 1)

        # --------- Concatena tudo ----------
        out = torch.cat([x, similarity_features, std_feature], dim=1)  # (batch, in_features + out_features + 1)
        return out

""" class Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=3,
                 patch_size=42,
                 data_emb_size=50,
                 label_emb_size=10,
                 seq_length = 150,
                 depth=3, 
                 n_classes=9, 
                 **kwargs): 
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, data_emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=data_emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(data_emb_size, 1, n_classes)
        ) """

class Discriminator(nn.Module):
    def __init__(self, in_channels, patch_size, data_emb_size, label_emb_size, seq_length, depth, n_classes, **kwargs):
        super().__init__()
        self.embedding = PatchEmbedding_Linear(in_channels, patch_size, data_emb_size, seq_length)
        self.encoder = Dis_TransformerEncoder(depth, emb_size=data_emb_size, drop_p=0.0, forward_drop_p=0.0, **kwargs)
        self.cls_head = ClassificationHead(data_emb_size, 1, n_classes)

    def forward(self, x):
        x = self.embedding(x)                  # (batch, seq+1, emb_size)
        x = self.encoder(x)                    # (batch, seq+1, emb_size)
        x = self.cls_head(x)                   # (batch, emb_size)
        return x

        