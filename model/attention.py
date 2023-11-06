import torch

from .feedforward import FeedforwardLayer


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, is_gelu, norm_eps):
        super(SelfAttentionLayer, self).__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, num_heads, dropout_rate)
        self.attn_norm = torch.nn.LayerNorm(d_model, norm_eps)
        self.ffn = FeedforwardLayer(d_model, 4 * d_model, d_model, dropout_rate, is_gelu)
        self.ffn_norm = torch.nn.LayerNorm(d_model, norm_eps)

    def forward(self, x, mask=None):
        x = torch.transpose(x, 0, 1)
        residual = x
        x, _ = self.attn.forward(x, x, x, attn_mask=mask)
        x += residual
        x = self.attn_norm.forward(x)
        x = torch.transpose(x, 0, 1)

        residual = x
        x = self.ffn.forward(x)
        x += residual
        x = self.ffn_norm.forward(x)

        return x
