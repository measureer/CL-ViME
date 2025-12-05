import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath
from functools import partial, reduce
from operator import mul
import math
import torch.nn.functional as F
__all__ = [
   'vit_col_112'
]

#change position embedding and onehot
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, dropout, norm_layer=None):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size[1], img_size // patch_size[0])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 正弦位置编码
        #position_embedding = self.get_sinusoid_encoding(self.num_patches + 1, embed_dim)
        #self.position_embedding = nn.Parameter(position_embedding, requires_grad=False)  # 通常正弦编码不需要学习
        self.build_sincos_encoding()
        
        self.dropout = nn.Dropout(p=dropout)
    
        
    def build_sincos_encoding(self):
        assert hasattr(self, 'grid_size') and len(self.grid_size) >= 1
        w = self.grid_size[0]  # 只考虑列数
        assert self.embed_dim > 0 and self.embed_dim % 2 == 0, 'Embed dimension must be positive and divisible by 2'
        
        device = next(self.parameters()).device if hasattr(self, 'parameters') else 'cpu'
        
        grid_w = torch.arange(w, dtype=torch.float32, device=device)  # (W,)
        pos_dim = self.embed_dim // 2
        omega = torch.exp(torch.arange(pos_dim, dtype=torch.float32, device=device) * 
                        (-math.log(10000.0) / pos_dim))
        
        out_w = grid_w[:, None] * omega  # (W, pos_dim)
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (W, D)
        pe_token = torch.zeros([1, self.embed_dim], device=device)  # (1, D)
        
        self.position_embedding = nn.Parameter(torch.cat([pe_token, pos_emb], dim=0)[None, ...])  # (1, W+1, D)
        self.position_embedding.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.position_embedding = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.position_embedding.requires_grad = False        
        
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MoEProcessor(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Routing network
        self.routing = nn.Sequential(
            nn.Linear(dim, num_experts),
            nn.LayerNorm(num_experts)
        )
        
        # Expert projections
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Compute routing weights
        routing_weights = self.routing(x)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Add noise for exploration
        noise = torch.randn_like(routing_weights) * (1.0 / self.num_experts)
        routing_weights = routing_weights + noise
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # Process with expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        # Gather outputs
        batch_indices = torch.arange(expert_outputs.size(0), device=expert_outputs.device)[:, None, None]
        token_indices = torch.arange(expert_outputs.size(1), device=expert_outputs.device)[None, :, None]
        
        selected_outputs = expert_outputs[batch_indices, token_indices, topk_indices]
        moe_output = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)
        
        return moe_output


class MoEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_experts=8, top_k=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.moe=MoEProcessor(dim, num_experts=num_experts, top_k=top_k)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.moe(self.norm2(x)))
        return x
  
class VisionTransformerMoCo(nn.Module):
    def __init__(self,in_channels,depth,img_size,patch_embed,stop_grad_conv1=True,classifier='None',
                 num_heads=6,mlp_ratio=4,drop_rate=0,num_classes=1000,drop_path_rate=0, qkv_bias=True, 
                 qk_scale=None,attn_drop_rate=0., norm_layer=None,act_layer=None,num_experts=8):
        super(VisionTransformerMoCo,self).__init__()
        act_layer = act_layer or nn.GELU
        self.embed_dim = img_size*2
        self.patch_embed = patch_embed(in_channels=in_channels,img_size=img_size,patch_size=(img_size,2),embed_dim=self.embed_dim,dropout=drop_rate,norm_layer=norm_layer)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MoEBlock(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,num_experts=num_experts)
            for i in range(depth)])
        self.norm = norm_layer(self.embed_dim)

        if classifier == 'None':
            self.head = nn.Identity()
        elif classifier == 'Linear': 
            self.head = nn.Linear(self.embed_dim*2, out_features=num_classes)
        elif classifier == 'MLP':
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim*2, 256),  # 第一个隐藏层
                nn.ReLU(),
                nn.Linear(256, 128),             # 第二个隐藏层
                nn.ReLU(),
                nn.Linear(128, num_classes)       # 输出层
            )
                # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
        nn.init.normal_(self.patch_embed.cls_token, std=1e-6)
        
        if isinstance(self.patch_embed, PatchEmbedding):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(in_channels* reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.patcher.weight, -val, val)
            nn.init.zeros_(self.patch_embed.patcher.bias)

            if stop_grad_conv1:#选择性冻结 PatchEmbed 层（stop_grad_conv1）
                self.patch_embed.patcher.weight.requires_grad = False
                self.patch_embed.patcher.bias.requires_grad = False

    def forward(self,x1,x2):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        x1 = self.blocks(x1)
        x2 = self.blocks(x2)
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        x1 = x1[:,0]
        x2 = x2[:,0]
        x=torch.cat((x1,x2),dim=1)
        x = self.head(x)
        return x
    
def vit_col_112(**kwargs):
    model = VisionTransformerMoCo(in_channels=1,patch_embed=PatchEmbedding,depth=12,num_heads=8,mlp_ratio=4,qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm,eps=1e-6),**kwargs)
    return model
