# model.py
import numpy as np
import torch.nn as nn
import torch


def create_pos_embeddings(n_pos, dim, out):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(pos_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(pos_enc[:, 1::2]))
    out.detach_()

class XVir(nn.Module):
    def __init__(self, read_len, ngram, model_dim, num_layers, dropout_prob):
        super(XVir, self).__init__()
        self.n = ngram
        self.input_dim = read_len - self.n + 1
        self.model_dim = model_dim
        self.num_layers = num_layers

        ff_dim = 4 * model_dim
        num_head = 4

        self.ngram_embed = nn.Embedding(4**self.n, self.model_dim, max_norm=1.0)

        # Configure position embeddings
        self.pos_embed = nn.Embedding(self.input_dim, self.model_dim, max_norm=1.0)
        self.pos_embed.weight.requires_grad = False
        create_pos_embeddings(self.input_dim, self.model_dim, self.pos_embed.weight)

        self.LayerNorm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout_prob)

        xformLayer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_head,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout_prob
        )
        self.xformEncoder = nn.TransformerEncoder(xformLayer, self.num_layers)

        self.prob = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.model_dim * self.input_dim, 1),
        )

    def forward(self, x):
        """
        Returns:
            x_enc: The penultimate-layer embeddings, shape (batch_size, input_dim, model_dim)
            x_prob: The final output logits, shape (batch_size, 1)
        """
        x_ngram = self.ngram_embed(x)
        pos_ids = torch.arange(self.input_dim, dtype=torch.long, device=x.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(x)  # (batch_size, input_dim)
        x_pos = self.pos_embed(pos_ids)
        x_embed = x_ngram + x_pos

        x_model = self.dropout(self.LayerNorm(x_embed))
        x_enc = self.xformEncoder(x_model)          # penultimate-layer embeddings
        x_prob = self.prob(x_enc)                   # final classification logits
        return x_enc, x_prob
