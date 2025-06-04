import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DecoderEntireLandmark(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_layers=6,
                 n_landmarks=512,
                 transformer_dim_feedforward=2048,
                 ldmks_dim=2,
                 dropout=0.1,
                 uncertainty=True,
                 visibility=True,):

        super(DecoderEntireLandmark, self).__init__()

        self.output_dim = ldmks_dim+1 if uncertainty else ldmks_dim

        activation = 'gelu'
        self.return_intermediate_dec = True
        normalize_before = False
        self.query_embed = nn.Embedding(n_landmarks, d_model)

        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                activation=activation, 
                                                norm_first=not normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, 
                                          num_layers=n_layers, 
                                          norm=decoder_norm)
        self.landmarks = MLP(d_model, d_model, self.output_dim, 3)

    def forward(self, src, pos_embed):
        patch_pos_embed = pos_embed[:, 1:]
        bs = src.shape[0]
        patch_pos_embed = patch_pos_embed.permute(1, 0, 2)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        src = rearrange(src, 'B D H W -> (H W) B D')

        hs = self.decoder(tgt, src, memory_key_padding_mask=None, pos=patch_pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2) #, memory.permute(1, 2, 0).view(bs, c, h, w)

        # Construct predictions
        pred = dict(
            joints2d=self.landmarks(hs)[-1],
        )

        return pred


class DecoderPerLandmark(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_layers=6,
                 n_landmarks=512,
                 transformer_dim_feedforward=2048,
                 ldmks_dim=2,
                 dropout=0.1,
                 uncertainty=True,
                 condition=True,):

        super(DecoderPerLandmark, self).__init__()

        self.condition = condition
        self.output_dim = ldmks_dim+1 if uncertainty else ldmks_dim

        activation = 'gelu'
        self.return_intermediate_dec = True
        normalize_before = False
        self.query_embed = nn.Embedding(n_landmarks, d_model)

        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                activation=activation, 
                                                norm_first=not normalize_before,
                                                batch_first=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, 
                                          num_layers=n_layers, 
                                          norm=decoder_norm)
        self.landmarks = MLP(d_model, d_model, self.output_dim, 3)

        if self.condition:
            self.cond_emb = nn.Linear(2, d_model)
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, 23, d_model))

    def forward(self, src, pos_embed, cond=None, cond_mask=None):
        patch_pos_embed = pos_embed[:, 1:]
        bs = src.shape[0]
        patch_pos_embed = patch_pos_embed.permute(1, 0, 2)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        src = rearrange(src, 'B D H W -> (H W) B D') + patch_pos_embed
        
        if self.condition:
            cond_emb = self.cond_emb(cond)
            cond_emb = cond_emb + self.cond_pos_emb
            cond_emb = rearrange(cond_emb, 'B J D -> J B D')

            src = torch.cat((cond_emb, src), dim=0)
            memory_key_padding_mask = torch.zeros(src.shape[1], src.shape[0]).bool().to(src.device)
            memory_key_padding_mask[:, :cond_mask.shape[1]] = cond_mask
        else:
            memory_key_padding_mask = None

        hs = self.decoder(query_embed, src, memory_key_padding_mask=memory_key_padding_mask)
        hs = rearrange(hs, 'J B D -> B J D')
        
        # Construct predictions
        pred = dict(
            joints2d=self.landmarks(hs),
        )

        return pred


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
                 condition=True,):

        super(EncDecPerLandmark, self).__init__()

        self.condition = condition
        assert self.condition, "This architecture always assumes the conditioning!"
        
        self.output_dim = ldmks_dim+1 if uncertainty else ldmks_dim

        activation = 'gelu'
        self.return_intermediate_dec = True
        normalize_before = False
        self.query_embed = nn.Embedding(n_landmarks, d_model)

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
            self.cond_emb = nn.Linear(2, d_model)
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, 23, d_model))

    def forward(self, src, pos_embed, cond=None, cond_mask=None):
        patch_pos_embed = pos_embed[:, 1:]
        bs = src.shape[0]
        patch_pos_embed = patch_pos_embed.permute(1, 0, 2)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        src = rearrange(src, 'B D H W -> (H W) B D') + patch_pos_embed
        
        # Process condition
        cond_emb = self.cond_emb(cond)
        cond_emb = cond_emb + self.cond_pos_emb
        cond_emb = rearrange(cond_emb, 'B J D -> J B D')

        # Encode
        src = torch.cat((cond_emb, src), dim=0)
        padding_mask = torch.zeros(src.shape[1], src.shape[0]).bool().to(src.device)
        padding_mask[:, :cond_mask.shape[1]] = cond_mask
        src_hs = self.encoder(src, src_key_padding_mask=padding_mask)
        
        # Decode
        hs = self.decoder(query_embed, src_hs, memory_key_padding_mask=padding_mask)
        hs = rearrange(hs, 'J B D -> B J D')
        
        # Construct predictions
        pred = dict(
            joints2d=self.landmarks(hs),
        )

        return pred