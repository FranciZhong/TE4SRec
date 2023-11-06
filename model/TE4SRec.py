import numpy as np
import torch

from .attention import SelfAttentionLayer
from .feedforward import FeedforwardLayer

NORM_EPS = 1e-8


class TE4SRec(torch.nn.Module):
    def __init__(self, config):
        super(TE4SRec, self).__init__()

        self.is_bert = config.is_bert
        self.device = config.device
        self.num_item = config.num_item

        # item embedding with padding 0 and masked value (num_item+1)
        self.item_embedding = torch.nn.Embedding(config.num_item + 2, config.d_model)

        # position embedding from 0 to (max_seq_len-1)
        self.position_embedding = torch.nn.Embedding(config.max_seq_len, config.d_model)

        # positional sequence constant
        self.POSITION_SEQ = torch.LongTensor(np.arange(config.max_seq_len))

        # masks for left-wise attention
        self.ATTN_MASK = ~torch.tril(torch.ones((config.max_seq_len, config.max_seq_len),
                                                dtype=torch.bool, device=config.device))

        self.input_norm = torch.nn.LayerNorm(config.d_model, NORM_EPS)

        # temporal encoding layer
        self.temporal_ffn = FeedforwardLayer(config.d_temporal,
                                             config.d_model * 4,
                                             config.d_model,
                                             config.dropout_rate)

        # self attention blocks
        self.num_block = config.num_block
        self.attention_layers = torch.nn.ModuleList()
        for _ in range(config.num_block):
            self.attention_layers.append(SelfAttentionLayer(config.d_model,
                                                            config.num_heads,
                                                            config.dropout_rate,
                                                            config.is_gelu,
                                                            NORM_EPS))

        # output layer components
        self.output_linear = torch.nn.Linear(config.d_model, config.d_model)
        self.output_act = torch.nn.GELU() if config.is_gelu else torch.nn.ReLU()
        self.output_bias = torch.nn.Parameter(torch.randn((config.num_item + 1,)))
        self.output_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        item_seqs, ts_seqs, seq_masks = inputs

        outputs = self.encode(item_seqs, seq_masks)

        temporal_encodings = self.temporal_ffn.forward(ts_seqs)
        outputs += temporal_encodings

        if self.is_bert:
            outputs = self.output_act(self.output_linear(outputs))
            item_embeddings = self.item_embedding(torch.arange(0, self.num_item + 1, device=self.device))

            # padding 0 is included for classification output
            outputs = torch.matmul(outputs, torch.transpose(item_embeddings, 0, -1))
            outputs += self.output_bias
        else:
            item_embeddings = self.item_embedding(torch.arange(0, self.num_item + 1, device=self.device))
            outputs = torch.matmul(outputs, torch.transpose(item_embeddings, 0, -1))

        return outputs

    def predict(self, inputs):
        outputs = self.output_softmax(self.forward(inputs))

        return outputs

    def encode(self, item_seqs, seq_masks):
        outputs = self.item_embedding.forward(item_seqs)

        # positional embeddings
        shape = list(item_seqs.shape[:-1])
        shape.append(1)
        with torch.no_grad():
            position_seqs = self.POSITION_SEQ.repeat(shape).to(self.device)

        outputs += self.position_embedding.forward(position_seqs)
        outputs = self.input_norm.forward(outputs)

        outputs *= ~seq_masks.unsqueeze(-1)

        # self attention feed forward
        for i in range(self.num_block):
            # left-wise masking
            attn_mask = None if self.is_bert else self.ATTN_MASK
            outputs = self.attention_layers[i].forward(outputs, attn_mask)

            outputs *= ~seq_masks.unsqueeze(-1)

        return outputs
