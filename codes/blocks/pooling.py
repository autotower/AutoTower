import torch
import torch.nn as nn
import math


POOLING_LIB = {
    'SumPooling':
        lambda params: SumPooling('SumPooling'),
    'AveragePooling':
        lambda params: AveragePooling('AveragePooling'),
    'SelfAttentivePooling':
        lambda params: SelfAttentivePooling('SelfAttentivePooling',
                                            params['embedding_dim'] * params['seq_fields'], params['seq_len']),
    'SelfAttentionPooling':
        lambda params: SelfAttentionPooling('SelfAttentionPooling',
                                            params['embedding_dim'] * params['seq_fields'], params['seq_len'])
}

POOLING2ID = {p: i for i, p in enumerate(sorted(POOLING_LIB.keys()))}
ID2POOLING = {i: p for i, p in enumerate(sorted(POOLING_LIB.keys()))}

POOLING_PARAMS = {}


class SumPooling(nn.Module):
    def __init__(self, block_name):
        super().__init__()
        self.pooling_name = block_name
        
    def forward(self, histories, mask):
        """
        :param histories (tensor): [batch_size, L, nfields * E]
            the histories here have been concated
        :param mask: [B, L]
        """
        float_mask = mask.float()
        float_mask = float_mask.unsqueeze(-1)
        histories = histories * float_mask
        out = torch.sum(histories, dim=1)
        return out


class AveragePooling(nn.Module):
    def __init__(self, block_name):
        super().__init__()
        self.pooling_name = block_name
        
    def forward(self, histories, mask):
        """
        :param histories (tensor): [batch_size, L, nfields * E]
            the histories here have been concated
        :param mask: [B, L]
        """
        float_mask = mask.float()
        float_mask = float_mask.unsqueeze(-1)
        history_lens = torch.sum(float_mask, dim=1)
        histories = histories * float_mask
        histories = torch.sum(histories, dim=1)
        histories = histories / (history_lens + 1e-8)
        return histories

    
class SelfAttentivePooling(nn.Module):
    def __init__(self, block_name, in_dim, seq_len, attention_dim=None, num_heads=1, add_position=True):
        super().__init__()
        
        self.pooling_name = block_name
        
        if attention_dim is None:
            attention_dim = in_dim
        
        self.num_heads = num_heads
        self.add_position = add_position
        if add_position:
            self.position_encoding = nn.parameter.Parameter(torch.randn(1, seq_len, in_dim))
        
        self.W1 = nn.Sequential(
            nn.Linear(in_dim, attention_dim, bias=False),
            nn.Tanh()
        )
        
        self.W2 = nn.Linear(attention_dim, num_heads, bias=False)
        
        if num_heads > 1:
            self.out_project = nn.Linear(num_heads * in_dim, in_dim, bias=False)
        
    def forward(self, input, mask):
        """
        :param input: [batch_size, L, E]
        """
        float_mask = mask.float()
        float_mask = float_mask.unsqueeze(-1)  # [B, L, 1]

        x = input * float_mask
        
        if self.add_position:
            x_add_position = x + self.position_encoding.repeat([x.shape[0], 1, 1])
        else:
            x_add_position = x
            
        hidden = self.W1(x_add_position)
        att_w = self.W2(hidden)  # [B, L, num_heads]
        att_w = torch.transpose(att_w, 2, 1).contiguous()  # [B, num_heads, L]
        
        atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.num_heads, 1)  # [B, num_heads, L]
        padding = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)
        
        att_w = torch.where(atten_mask, att_w, padding)
        att_w = torch.softmax(att_w, dim=-1)  # [B, num_heads, L]
        
        x = torch.bmm(att_w, x)  # [B, num_heads, E]
        x = x.view((x.shape[0], -1))
        
        if self.num_heads > 1:
            x = self.out_project(x)
            
        return x
    
    
class SelfAttentionPooling(nn.Module):
    def __init__(self, block_name, in_dim, seq_len, num_heads=1, add_position=True, dropout=0.2):
        assert in_dim % num_heads == 0, f'attention_dim: {in_dim}, num_heads: {num_heads}'
        super().__init__()
        
        self.pooling_name = block_name
        
        self.num_heads = num_heads
        self.add_position = add_position
        self.model_dim = in_dim
        self.head_dim = in_dim // num_heads
        
        if add_position:
            self.position_encoding = nn.parameter.Parameter(torch.randn(1, seq_len, in_dim))
            
        self.linear_keys = nn.Linear(in_dim, in_dim)
        self.linear_values = nn.Linear(in_dim, in_dim)
        self.linear_query = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(in_dim, in_dim)
        
    def forward(self, input, mask):
        """
        :param input: [batch_size, L, E]
        :param mask: [batch_size, L]
        """
        batch_size = input.size(0)
        head_dim = self.head_dim
        num_heads = self.num_heads

        def shape(x):
            """projection"""
            return x.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_dim * num_heads)  # [B, L, E]

        if self.add_position:
            x = input + self.position_encoding.repeat([input.shape[0], 1, 1])
        else:
            x = input
            
        # 1. project key, value, and query
        key = self.linear_keys(x)
        value = self.linear_values(x)
        query = self.linear_query(x)
        
        key, value, query = shape(key), shape(value), shape(query)  # [B, num_heads, L, head_dim]
        
        # 2. calculate and scale scores.
        query = query / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(2, 3))  # [B, num_heads, L, L]
        
        key_mask = mask.unsqueeze(1).unsqueeze(-1).expand_as(scores)
        query_mask = torch.transpose(key_mask, 2, 3)
        
        padding = torch.ones_like(key_mask, dtype=torch.float) * (-2 ** 32 + 1)
        
        scores = torch.where(key_mask, scores, padding)
        scores = torch.where(query_mask, scores, padding)
        
        attn = torch.softmax(scores, dim=-1)  # [B, num_heads, L, L]
        drop_attn = self.dropout(attn)
        
        out = torch.matmul(drop_attn, value)
        out = unshape(out)
        out = self.final_linear(out)
        
        float_mask = mask.float()
        float_mask = float_mask.unsqueeze(-1)  # [B, L, 1]
        
        history_lens = torch.sum(float_mask, dim=1)
        out = out * float_mask
        out = torch.sum(out, dim=1)
        out = out / (history_lens + 1e-8)
        
        return out