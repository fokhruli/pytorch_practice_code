import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "embed size need to divisible by head"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        values_len, keys_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # split data into multiple head
        values = values.reshape(N, values_len, self.head, self.head_dim)
        keys = keys.reshape(N, keys_len, self.head, self.head_dim)
        query = query.reshape(N, query_len, self.head, self.head_dim)

        energy = torch.einsum("nqhd, nkhd-->nhqk", [query, keys])
        # query shape: (N, query_len, head, head_dim)
        # keys shape: (N, keys_len, head, head_dim)
        # energy shape: (N, head, quary_len, keys_len)
        if mask is not None:
            energy = energy.masked_fill(mask = 0, float("-1e20"))
        attention = torch.softmax(energy/(self.embed_size **(1/2)), dim = 3)

        out = torch.einsum("nhql, nlhd-->nqhd", [attention, values]).reshape(
            N, query_len, self.head*self.head_dim
        )

        # attention shape: (N, heads, query_len, keys_len)
        # values shape: (N, values_len, head, head_dim)
        # out shape : (N, query_len, head, head_dim) --> keys_len == values_len
        out = self.fc_out(out)
        return out
