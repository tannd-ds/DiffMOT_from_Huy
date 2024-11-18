
import torch
from torch import nn
from einops import rearrange, repeat
import math

class ResidualConnection(nn.Module):
    def __init__(self, _size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(_size)

    def forward(self, x, att_features):
        return x + self.dropout(self.norm(att_features))

class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.1):
      super(MLP, self).__init__()
      self.layer_norm = nn.LayerNorm(in_features)
      self.dropout = nn.Dropout(dropout)
      self.dense_layer = nn.Sequential(
          nn.Linear(in_features, in_features * 2),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(in_features * 2, in_features),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(in_features, out_features)
      )

    def forward(self, x):
      normalized_x = self.layer_norm(x)
      ffn_x = self.dense_layer(normalized_x)
      output_x = self.dropout(ffn_x)
      return output_x

class TransAoA(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size = 256):
      super(TransAoA, self).__init__()
      layer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                          nhead=1,
                                          dim_feedforward=4*hidden_size)
      layer_norm = nn.LayerNorm(hidden_size)
      self.mlp_input = MLP(in_features = input_size, out_features = hidden_size)
      self.transformer_core = nn.TransformerDecoder(decoder_layer = layer,
                                                    num_layers=num_layers,
                                                    norm = layer_norm)
      self.aoa = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GLU(),
      ) # AoA Layer

      self.residual_fn = ResidualConnection(hidden_size)
      self.head = nn.Linear(hidden_size, output_size)

    def forward(self, input, ctx):
      batch_size = input.size(0)
      input = self.mlp_input(input)

      encoded_input = self.transformer_core(tgt = input,
                                            memory = ctx)
      aoa_output = self.aoa(torch.cat([encoded_input, input], dim = -1))
      res_connection = self.residual_fn(input, aoa_output)
      return self.head(res_connection)

class ReUNet3Plus(nn.Module):
    def __init__(self, noise_dim = 4, num_layers = 1, hidden_size = 256, filters = [16, 64, 128, 256], mid = True):
        super(ReUNet3Plus, self).__init__()
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.filters = filters
        self.reversed_filters = filters[::-1]
        self.mid = mid
        self.shared_ctx_mlp = MLP(in_features = hidden_size + 3,
                                  out_features = hidden_size)
        self.up_blocks, self.down_blocks = nn.ModuleList(), nn.ModuleList()
        self.prediction = MLP(in_features = self.filters[0],
                              out_features = noise_dim)

        ## -------------UP--------------
        input_size = noise_dim
        for filter in self.filters:
          block = TransAoA(input_size = input_size,
                           output_size = filter,
                           num_layers = num_layers,)
          self.up_blocks.append(block)
          input_size = filter

        ## -------------DOWN--------------
        for i in range(len(self.reversed_filters) - 1):
          block = TransAoA(input_size = self.reversed_filters[i],
                           output_size = self.reversed_filters[i+1],
                           num_layers = num_layers,)
          self.down_blocks.append(block)

        ## -------------MID--------------

        # if self.mid
        # '''stage 4d'''

    def forward(self, x, beta, context):
      batch_size = x.size(0)
      beta = beta.view(batch_size, 1) # (B, 1)
      context = context.view(batch_size, -1)   # (B, F)
      time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
      ctx_emb = self.shared_ctx_mlp(torch.cat([time_emb, context], dim=-1)) # (B, 256)

      output = x # 16, 4
      ## -------------UP--------------
      for i, block in enumerate(self.up_blocks):
        output = block(input = output,
                       ctx = ctx_emb) # 16, 4^i

      ## -------------MID--------------

      # if self.mid

      ## -------------DOWN--------------
      for i, block in enumerate(self.down_blocks):
        output = block(input = output,
                       ctx = ctx_emb)

      output = self.prediction(output)
      return output
