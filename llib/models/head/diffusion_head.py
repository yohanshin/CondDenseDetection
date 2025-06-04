import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    def initialize_last_layer(self):
        # Extract the last layer
        last_layer = self.layers[-1]
        nn.init.constant_(last_layer.weight, 0.0)  # Optional: Initialize the first weight to 0
        nn.init.uniform_(last_layer.bias[:2], a=-2, b=2)  # Optional: Initialize the first bias to 0
        last_layer.bias.data[:2].clamp_(-1, 1)  # Optional: Clamp the first bias to -1 and 1

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]: 0, 1, 2, ..., 4999
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # torch.arange(0, d_model, 2): [256]: 0, 2, 4, 6, 8, ..., 510  div_term: [256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)   # [5000, 1, 512]

        self.register_buffer('pe', pe.unsqueeze(1))
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)



class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = rearrange(x, "b n d -> n b d")
        x = self.poseEmbedding(x)  # [L, bs, dim]
        return x


class MaskBitEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

    def forward(self, mask_bits):
        if mask_bits.dtype != torch.long:
            mask_bits = mask_bits.long()

        mask_emb = self.embedding(mask_bits)
        return mask_emb


class EncDecPerLandmark(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_enc_layers=6,
                 n_dec_layers=6,
                 n_landmarks=512,
                 transformer_dim_feedforward=2048,
                 ldmks_dim=2,
                 dropout=0.1,
                 uncertainty=True,
                 condition=True,
                 inpainting=False):

        super(EncDecPerLandmark, self).__init__()

        self.condition = condition
        self.inpainting = inpainting
        assert self.condition, "This architecture always assumes the conditioning!"
        
        self.output_dim = ldmks_dim+1 if uncertainty else ldmks_dim

        activation = 'gelu'
        self.return_intermediate_dec = True
        normalize_before = False
        self.query_embed = nn.Embedding(n_landmarks, d_model)

        # Diffusion
        self.time_embed = TimestepEmbedder(d_model, dropout)
        self.xt_embed = InputProcess(ldmks_dim, d_model)
        
        # Make encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=n_heads,
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                norm_first=not normalize_before,
                                                batch_first=False)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, 
                                          num_layers=n_enc_layers,
                                          norm=encoder_norm)
        
        # Make decoder
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                activation=activation, 
                                                norm_first=not normalize_before,
                                                batch_first=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, 
                                          num_layers=n_dec_layers, 
                                          norm=decoder_norm)
        self.landmarks = MLP(d_model, d_model, self.output_dim, 3)

        if self.condition:
            self.cond_emb = InputProcess(2, d_model)
            self.cond_pos_emb = nn.Parameter(torch.zeros(23, 1, d_model))

        if self.inpainting:
            self.mask_embed = MaskBitEmbedding(d_model)


    def forward(self, batch, timesteps, cond=None, cond_mask=None, mask_bits=None):
        
        # Assign variables
        src = batch["features"]
        pos_embed = batch["pos_embed"]
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        bs = src.shape[0]
        
        # Condition encoding
        patch_pos_embed = pos_embed[:, 1:]
        patch_pos_embed = patch_pos_embed.permute(1, 0, 2)
        
        src = rearrange(src, 'B D H W -> (H W) B D') + patch_pos_embed
        cond_emb = self.cond_emb(cond)
        cond_emb = cond_emb + self.cond_pos_emb
        src = torch.cat((cond_emb, src), dim=0)
        padding_mask = torch.zeros(src.shape[1], src.shape[0]).bool().to(src.device)
        padding_mask[:, :cond_mask.shape[1]] = cond_mask
        src_hs = self.encoder(src, src_key_padding_mask=padding_mask)
        
        # Process diffusion variables
        time_emb = self.time_embed(timesteps)
        xt_emb = self.xt_embed(batch["x_t"])
        query_emb = self.query_embed.weight
        query_emb = query_emb.unsqueeze(1).repeat(1, bs, 1)

        if self.inpainting:
            mask_bits = batch["mask_bits"]
            mask_emb = self.mask_embed(mask_bits)
            mask_emb = rearrange(mask_emb, "B J D -> J B D")
            query_emb = query_emb + mask_emb
        
        # Decode
        query = xt_emb + query_emb + time_emb
        hs = self.decoder(query, src_hs, memory_key_padding_mask=padding_mask)
        hs = rearrange(hs, 'J B D -> B J D')

        # Construct predictions
        prediction = self.landmarks(hs)
        pred = dict(
            joints2d=prediction[..., :2],
            uncertainty=prediction[..., -1:],
        )

        return pred