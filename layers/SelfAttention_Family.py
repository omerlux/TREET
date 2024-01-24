import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask, DoubleTriangularCausalMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, *args, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # bh.le @ bh.es = bhls - each head perform multiplication separately
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullFixedTimeCausalConstructiveAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, history_len=None):
        super(FullFixedTimeCausalConstructiveAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.queries_origin = None
        self.keys_origin = None
        self.values_origin = None
        self.history_len = history_len

    def forward(self, queries, keys, values, attn_mask, drawn_y=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        if self.history_len is not None and not drawn_y:
            self.queries_origin = queries
            self.keys_origin = keys
            self.values_origin = values
        elif self.history_len is not None and drawn_y:
            # most of the operation is with origin. only queries is half origin half drawn
            queries_drawn = queries
            queries = self.queries_origin.clone()
            keys_drawn = keys
            keys = self.keys_origin.clone()
            values_drawn = values
            values = self.values_origin.clone()
            # the history is from queries_origin
            queries[:, self.history_len:] = queries_drawn[:, self.history_len:]

        # bh.le @ bh.es = bhls - each head perform multiplication separately
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.history_len is not None and drawn_y:
            future_scores_with_drawn_keys = torch.einsum("blhe,blhe->bhl", queries[:, self.history_len:], keys_drawn[:, self.history_len:])
            scores_with_drawn_keys = torch.diag_embed(
                torch.cat([
                    torch.zeros((B, H, self.history_len)).to(scores.device),
                    future_scores_with_drawn_keys
                ], dim=-1)
            )
            partial_eye_mask = torch.diag(
                torch.cat([torch.zeros(self.history_len),
                           torch.ones(L - self.history_len)])
            ).to(scores.device)[:, :S]
            # the partial eye mask is for the drawn values
            scores = scores_with_drawn_keys * partial_eye_mask + (1 - partial_eye_mask) * scores

        if self.mask_flag:
            if attn_mask is None:
                # attn_mask = TriangularCausalMask(B, L, device=queries.device)
                attn_mask = DoubleTriangularCausalMask(B, L, diagonal=1, history=self.history_len, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        if self.history_len is not None and drawn_y:
            # setting the values as matrix
            values = values.permute(0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, L, 1)
            values_drawn = values_drawn.permute(0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, L, 1)
            # the partial eye mask is for the drawn values
            values = values_drawn * partial_eye_mask + (1 - partial_eye_mask) * values
            # scores multiplied by values matrix
            # V = (A.unsqueeze(2).repeat(1,1,32,1,1) * values).sum(-1).permute(0, 3, 1, 2)
            V = torch.einsum("bhls,bhdls->blhd", A, values)
        else:
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, drawn_y=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            drawn_y=drawn_y
        )
        # out = out.view(B, L, -1)
        out = out.view(B, out.size(1), -1)

        return self.out_projection(out), attn
