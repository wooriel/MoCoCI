import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# Tokenizer
class RawTokenizer(nn.Module):
    def __init__(self, d_model=64, patch_h=23, patch_w=25):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w

        patch_dim = 1 * patch_h * patch_w
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        B, C, H, W = x.shape # x: [B, 1, 23, 4375]

        usable_h = (H // self.patch_h) * self.patch_h # patch_h로 나눠지는 길이
        usable_w = (W // self.patch_w) * self.patch_w # patch_w로 나눠지는 길이

        x = x[:, :, :usable_h, :usable_w] # x: [B, 1, 23, 4375]

        x = x.unfold(2, self.patch_h, self.patch_h)
        x = x.unfold(3, self.patch_w, self.patch_w) # [B, 1, 1, 23, 175, 25]

        x = x.permute(0, 2, 3, 1, 4, 5) # [B, 1, 175, 1, 23, 25] = [B, H_patch, W_patch, C, patch_h, patch_w]
        x = x.reshape(B, -1, C * self.patch_h * self.patch_w) # [B, 1*175, 575]
        x = self.proj(x) # [B, N, d_model]

        return x
    

# Transformer Encoder
class SelfAttentionEncoder(nn.Module):
    def __init__(self, nhead, d_model=64, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.dim = d_model
        self.head = nhead

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True, # [B, N, D] order
            norm_first=True, # apply layernorm before attention layer
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x): # [B*nhead, N, self.dim]
        return self.encoder(x)


# HPCP Attention Model
class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.score = nn.Linear(d_model, 1)

    def forward(self, x): # x: [B, N, D]
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

    def encode_tokens(self, x):
        B = x.shape[0]
        x = self.tokenizer(x)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.self_attention(x) # [B, N, D]

        # x = x.reshape(B, self.head, x.shape[1], self.dim) # [B*H, N, D]
        return x
    
    def global_embedding(self, x): # B, H, N, D = x.shape
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
            "pair_similarity": pair_sim,
            "cosine_similarity": cosine_sim
        }
    

# MoCo Attention Model
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=64,):
        super().__init__()

        hidden_dim = in_dim*4
        out_dim = in_dim*2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class PredictionMLP(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()

        hidden_dim = in_dim*2
        out_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class HPCPSelfAttentionEncoder(nn.Module):
    def __init__(self, tokenizer, self_attention, nhead, num_tokens=175, d_model=64):
        super().__init__()

        self.dim = d_model
        self.head = nhead
        self.tokenizer = tokenizer
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, self.dim))
        self.self_attention = self_attention
        # self.pair_projection = PairProjection(d_model=d_model, pair_dim=d_model)

    def encode_tokens(self, x):  # [B, 1, 96, 4375]
        x = self.tokenizer(x)  # [B, N, D]
        x = x + self.pos[:, :x.shape[1], :]
        x = self.self_attention(x)  # [B, N, D]

        return x

    def global_embedding(self, x):  # [B, N, D]
        x = x.mean(dim=1)  # [B, D]
        x = F.normalize(x, p=2, dim=-1)

        return x

    def local_embedding(self, x):  # [B, N, D]
        x = F.normalize(x, p=2, dim=-1)

        return x

    def encode_one(self, x):
        tokens = self.encode_tokens(x)

        global_feature = self.global_embedding(tokens)
        local_feature = self.local_embedding(tokens)

        return {
            "global_feature": global_feature,
            "local_feature": local_feature,
        }


    def forward(self, src, tgt):
        src_out = self.encode_one(src)
        tgt_out = self.encode_one(tgt)

        src_feature = src_out["global_feature"]
        tgt_feature = tgt_out["global_feature"]

        src_local = src_out["local_feature"]
        tgt_local = tgt_out["local_feature"]

        pair_sim = torch.bmm(src_local, tgt_local.transpose(1, 2))

        cosine_sim = F.cosine_similarity(src_feature, tgt_feature, dim=-1)

        return {
            "src_feature": src_feature,
            "tgt_feature": tgt_feature,
            "src_local": src_local,
            "tgt_local": tgt_local,
            "pair_similarity": pair_sim,
            "cosine_similarity": cosine_sim,
        }
    

class HPCPMoCo(nn.Module):
    def __init__(self, encoder, proj_dim=128, proj_hidden=256, momentum=0.99, temperature=0.2):
        super().__init__()

        self.m = momentum
        self.temperature = temperature

        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)

        self.projector_q = ProjectionMLP(encoder.dim) # , proj_hidden, proj_dim
        self.projector_k = copy.deepcopy(self.projector_q)
        self.predictor = PredictionMLP(proj_dim) # , proj_hidden, proj_dim,

        self.encoder_k = copy.deepcopy(encoder)
        self.projector_k = copy.deepcopy(self.projector_q)

        # key network gets no normal gradient
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        for p in self.projector_k.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_key(self, x): # encode key without predictor
        enc = self.encoder_k.encode_one(x)

        h = enc["global_feature"]

        k = self.projector_k(h)
        k = F.normalize(k, dim=-1)

        return {
            "global_feature": h,
            "projected_feature": k,
            "local_feature": enc["local_feature"],
        }

    def encode_query(self, x):
        enc = self.encoder_q.encode_one(x)

        h = enc["global_feature"]

        z = self.projector_q(h)
        z = F.normalize(z, dim=-1)

        q = self.predictor(z)
        q = F.normalize(q, dim=-1)

        return {
            "global_feature": h,
            "projected_feature": z,
            "predicted_feature": q,
            "local_feature": enc["local_feature"],
        }

    @torch.no_grad()
    def momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.mul_(self.m).add_(q.data, alpha=1.0 - self.m)

        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data.mul_(self.m).add_(q.data, alpha=1.0 - self.m)

    def forward(self, x1, x2): # query input
        out_q1 = self.encoder_q.encode_one(x1)
        out_q2 = self.encoder_q.encode_one(x2)

        global_q1 = out_q1["global_feature"]
        global_q2 = out_q2["global_feature"]

        local_q1 = out_q1["local_feature"]
        local_q2 = out_q2["local_feature"]

        z_q1 = self.projector_q(global_q1)
        z_q2 = self.projector_q(global_q2)

        z_q1 = F.normalize(z_q1, dim=-1)
        z_q2 = F.normalize(z_q2, dim=-1)

        q1 = self.predictor(z_q1)
        q2 = self.predictor(z_q2)

        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)

        # local
        pair_sim = torch.bmm(local_q1, local_q2.transpose(1, 2))

        with torch.no_grad():
            self.momentum_update()

            out_k1 = self.encoder_k.encode_one(x1)
            out_k2 = self.encoder_k.encode_one(x2)

            h_k1 = out_k1["global_feature"]
            h_k2 = out_k2["global_feature"]

            k1 = self.projector_k(h_k1)
            k2 = self.projector_k(h_k2)

            k1 = F.normalize(k1, dim=-1)
            k2 = F.normalize(k2, dim=-1)

        return {
            "x1_glob_feature": global_q1,
            "x2_glob_feature": global_q2,

            "x1_local_feature": local_q1,
            "x2_local_feature": local_q2,

            "x1_query_proj": z_q1,
            "x2_query_proj": z_q2,

            "x1_query_pred": q1,
            "x2_query_pred": q2,

            "x1_key_feature": k1,
            "x2_key_feature": k2,

            "pair_sim": pair_sim
        }