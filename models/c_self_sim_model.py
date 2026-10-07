import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAttentionPool(nn.Module): # used for CQT Pooling
    def __init__(self, d_model=64):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x): # x: [B, N, D]
        weights = self.score(x) # [B, N, 1]
        weights = torch.softmax(weights, dim=1) # [B, N, 1]
        x = torch.sum(weights * x, dim=1) # [B, D]

        return x


class VitSelfAttentionModel(nn.Module):
    def __init__(self, tokenizer, self_attention, nhead, d_model=64):
        super().__init__()
        self.dim = d_model
        self.head = nhead # freq
        self.tokenizer = tokenizer
        self.self_attention = self_attention
        self.song_pool = TokenAttentionPool(d_model=self.dim)

    def encode(self, x): # [B, 1, 96, 4375]
        B = x.shape[0]
        x = self.tokenizer(x) # [B*4, 182, 64] | [B, 364, 64]
        x = self.self_attention(x)
        x = self.song_pool(x) # [B*4 64] | # [B*1 64]
        x = x.reshape(B, self.head, self.dim)  # [B, head, 64]
        x = x.mean(dim=1) # [B, 64]
        x = F.normalize(x, p=2, dim=-1,)

        return x

    def forward(self, src, tgt):
        src_feature = self.encode(src) # [B, 64]
        tgt_feature = self.encode(tgt) # [B, 64]

        cosine_sim = F.cosine_similarity(src_feature, tgt_feature, dim=-1) # [B]

        return {
            "src_feature": src_feature,
            "tgt_feature": tgt_feature,
            "cosine_similarity": cosine_sim,
        }
    

class VitSelfAttentionModel(nn.Module): # used for CQT
    def __init__(self, tokenizer, self_attention, nfreq=4, d_model=64):
        super().__init__()
        self.dim = d_model
        self.nfreq = nfreq
        self.tokenizer = tokenizer
        self.self_attention = self_attention
        self.song_pool = TokenAttentionPool(d_model=self.dim)

    def encode_tokens(self, x):  # [B, 1, 96, 4375]
        B = x.shape[0]
        x = self.tokenizer(x)  # [B*4, 182, 64]
        x = self.self_attention(x)  # [B*4, 182, 64]
        x = x.reshape(B, self.nfreq, x.shape[1], self.dim)  # [B, 4, 182, 64]
        return x

    def global_embedding(self, x):
        B, H, N, D = x.shape
        x = x.reshape(B * H, N, D)  # [B*4, 182, 64]
        x = self.song_pool(x)  # [B*4, 64]
        x = x.reshape(B, H, D)  # [B, 4, 64]
        x = x.mean(dim=1)  # [B, 64]
        x = F.normalize(x, p=2, dim=-1)
        return x

    def local_embedding(self, x):
        x = x.mean(dim=1)  # [B, 182, 64]
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, src, tgt):
        src_token = self.encode_tokens(src)
        tgt_token = self.encode_tokens(tgt)

        src_feature = self.global_embedding(src_token)
        tgt_feature = self.global_embedding(tgt_token)

        src_local = self.local_embedding(src_token)
        tgt_local = self.local_embedding(tgt_token)

        cosine_sim = F.cosine_similarity(src_feature, tgt_feature, dim=-1)

        return {
            "src_feature": src_feature,
            "tgt_feature": tgt_feature,
            "src_local": src_local,
            "tgt_local": tgt_local,
            "cosine_similarity": cosine_sim,
        }
    

# --- HPCP related models --- #
class MultiTokenAttentionPool(nn.Module): # used in HPCP pooling
    def __init__(self, d_model=64, num_pool=4):
        super().__init__()
        self.num_pool = num_pool
        self.score = nn.Linear(d_model, num_pool)

    def forward(self, x): # [B, N, D]
        weights = self.score(x) # [B, N, K]
        weights = torch.softmax(weights, dim=1)

        pooled = torch.einsum(
            "bnk,bnd->bkd",
            weights,
            x,
        ) # [B, K, D]

        return pooled
    

# class PairScorer(nn.Module):
#     def __init__(self, d_model=64):
#         super().__init__()

#         self.scorer = nn.Sequential(
#             nn.Linear(d_model * 2, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, 1)
#         )

#     def forward(self, src, tgt): # src: [B,N,D] | tgt: [B,M,D]

#         B, N, D = src.shape
#         M = tgt.shape[1]

#         src = src.unsqueeze(2).expand(-1, -1, M, -1) # [B,N,M,D]
#         tgt = tgt.unsqueeze(1).expand(-1, N, -1, -1) # [B,N,M,D]

#         pair = torch.cat([src, tgt], dim=-1) # [B,N,M,2D]

#         return self.scorer(pair).squeeze(-1) # [B,N,M]


class PairProjection(nn.Module): # this finds local match between 
    def __init__(self, d_model=64, pair_dim=64):
        super().__init__()
        self.src_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pair_dim)
        )

        self.tgt_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pair_dim)
        )

    def forward(self, src, tgt): # src: [B,N,D] | tgt: [B,M,D]
        # src = self.src_proj(src) # [B,N,pair_dim]
        # tgt = self.tgt_proj(tgt) # [B,M,pair_dim]

        # src = F.normalize(src, p=2, dim=-1)
        # tgt = F.normalize(tgt, p=2, dim=-1)

        sim = torch.bmm(src, tgt.transpose(1, 2)) # [B,N,M]

        return src, tgt, sim
    

class CosPoolSelfAttentionModel(nn.Module):
    def __init__(self, tokenizer, self_attention, nhead, d_model=64, num_pool=4):
        super().__init__()
        self.dim = d_model
        self.head = nhead
        self.num_pool = num_pool
        self.tokenizer = tokenizer
        self.self_attention = self_attention

        self.song_pool = MultiTokenAttentionPool(d_model=self.dim, num_pool=self.num_pool)
        self.pool_proj = nn.Linear(self.head * self.num_pool * self.dim, self.dim)

    # def encode(self, x):
    #     B = x.shape[0]

    #     x = self.tokenizer(x) # [B*head, N, D]
    #     x = self.self_attention(x) # [B*head, N, D]
    #     x = self.song_pool(x) # [B*head, K, D]

    #     x = x.reshape(B, self.head, self.num_pool, self.dim) # [B, head, K, D]
    #     x = x.flatten(1) # [B, head*K*D]
    #     x = self.pool_proj(x) # [B, D]
    #     x = F.normalize(x, p=2, dim=-1)

    #     return x

    def encode(self, x):
        B = x.shape[0]

        x = self.tokenizer(x) # [B*head, N, D]
        x = self.self_attention(x) # [B*head, N, D]
        # x = self.song_pool(x) # [B*head, K, D]
        x = x.mean(dim=1) # [B*head, D]

        x = x.reshape(B, self.head, self.dim) # [B, head, K, D]
        x = x.mean(dim=1)
        x = F.normalize(x, p=2, dim=-1)

        return x
    

    def forward(self, src, tgt):
        src_feature = self.encode(src)
        tgt_feature = self.encode(tgt)

        cosine_sim = F.cosine_similarity(src_feature, tgt_feature, dim=-1)

        return {
            "src_feature": src_feature,
            "tgt_feature": tgt_feature,
            "cosine_similarity": cosine_sim,
        }
    

class LGSelfAttentionModel(nn.Module): # used for HPCP
    def __init__(self, tokenizer, self_attention, nhead, d_model=64, num_pool=4):
        super().__init__()
        self.dim = d_model
        self.head = nhead
        self.num_pool = num_pool
        self.tokenizer = tokenizer
        self.self_attention = self_attention

        self.song_pool = MultiTokenAttentionPool(d_model=self.dim, num_pool=self.num_pool)
        self.pool_proj = nn.Linear(self.head * self.num_pool * self.dim, self.dim)

    def encode_tokens(self, x):
        B = x.shape[0]
        x = self.tokenizer(x)
        x = self.self_attention(x) # [B * H, N, D]

        x = x.reshape(B, self.head, x.shape[1], self.dim)
        return x
    
    def global_embedding(self, x):
        B, H, N, D = x.shape

        x = x.reshape(B * H, N, D) # [B*H, N, D]
        x = self.song_pool(x) # [B*H, K, D]
        x = x.reshape(B, H, self.num_pool, D) # [B, H, K, D]
        x = x.flatten(1) # [B, H*K*D]
        x = self.pool_proj(x) # [B, D]
        x = F.normalize(x, p=2, dim=-1)
        return x
    
    def local_embedding(self, x):
        x = x.mean(dim=1) # [B, N, D]
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, src, tgt):
        src_token = self.encode_tokens(src)
        tgt_token = self.encode_tokens(tgt)

        src_feature = self.global_embedding(src_token)
        tgt_feature = self.global_embedding(tgt_token)

        src_local = self.local_embedding(src_token)
        tgt_local = self.local_embedding(tgt_token)

        cosine_sim = F.cosine_similarity(src_feature, tgt_feature, dim=-1)

        return {
            "src_feature": src_feature,
            "tgt_feature": tgt_feature,
            "src_local": src_local,
            "tgt_local": tgt_local,
            "cosine_similarity": cosine_sim,
        }
    

class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, N, D]

        weight = self.score(x) # [B, N, 1]
        weight = torch.softmax(weight, dim=1)

        x = (x * weight).sum(dim=1) # [B, D]

        return x
    

class HPCPSelfAttentionModel(nn.Module): # used for HPCP
    def __init__(self, tokenizer, self_attention, nhead, num_token=175, d_model=64):
        super().__init__()
        self.dim = d_model
        self.head = nhead
        self.num_token = num_token
        self.tokenizer = tokenizer
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token, self.dim))
        self.self_attention = self_attention
        self.global_pool = AttentionPool(d_model=self.dim)

        # self.pair_projection = PairProjection(d_model=self.dim, pair_dim=self.dim)
        # self.song_pool = MultiTokenAttentionPool(d_model=self.dim, num_pool=self.num_pool)
        # self.pool_proj = nn.Linear(self.head * self.num_pool * self.dim, self.dim)

    def encode_tokens(self, x):
        B = x.shape[0]
        x = self.tokenizer(x)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.self_attention(x) # [B, N, D]

        # x = x.reshape(B, self.head, x.shape[1], self.dim) # [B*H, N, D]
        return x
    
    def global_embedding(self, x):
        # B, H, N, D = x.shape
        B, N, D = x.shape

        # x = x.mean(dim=1)
        x = self.global_pool(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
    
    def local_embedding(self, x):
        # x = x.mean(dim=1) # [B, N, D]
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, src, tgt):
        src_token = self.encode_tokens(src)
        tgt_token = self.encode_tokens(tgt)

        src_feature = self.global_embedding(src_token)
        tgt_feature = self.global_embedding(tgt_token)

        src_local = self.local_embedding(src_token)
        tgt_local = self.local_embedding(tgt_token)

        pair_sim = torch.bmm(src_local, tgt_local.transpose(1, 2))

        cosine_sim = F.cosine_similarity(src_feature, tgt_feature, dim=-1)

        return {
            "src_feature": src_feature,
            "tgt_feature": tgt_feature,
            "src_local": src_local,
            "tgt_local": tgt_local,
            # "src_pair": src_pair,
            # "tgt_pair": tgt_pair,
            "pair_similarity": pair_sim,
            "cosine_similarity": cosine_sim
        }
