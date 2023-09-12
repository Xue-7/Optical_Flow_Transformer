# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

# ========================= Masked辅助函数 ==============================
def masked_softmax(X, valid_length, value=-1e6):
    if valid_length is None:
        return F.softmax(X, dim=-1)
    else:
        X_size = X.size()
        device = valid_length.device
        if valid_length.dim() == 1:
            valid_length = torch.tensor(valid_length.cpu().numpy().repeat(X_size[1], axis=0),
                                        dtype=torch.float, device=device) if valid_length.is_cuda \
                else torch.tensor(valid_length.numpy().repeat(X_size[1], axis=0),
                                  dtype=torch.float, device=device)
        else:
            valid_length = valid_length.view([-1])
        X = X.view([-1, X_size[-1]])
        max_seq_length = X_size[-1]
        valid_length = valid_length.to(torch.device('cpu'))
        mask = torch.arange(max_seq_length, dtype=torch.float)[None, :] >= valid_length[:, None]
        X[mask] = value
        X = X.view(X_size)
        return F.softmax(X, dim=-1)
# ============================ 编码器实现 =================================
class DotProductAttention(nn.Module):
    # 经过DotProductAttention之后，输入输出的维度是不变的，都是[batch_size*h, seq_len, d_model//h]
    def __init__(self, dropout,):
        super(DotProductAttention, self).__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, valid_length):
        # Q, K, V shape:[batch_size*h, seq_len, d_model//h]
        d_model = Q.size()[-1]  # int
        # torch.bmm表示批次之间（>2维）的矩阵相乘
        attention_scores = torch.bmm(Q, K.transpose(1, 2))/math.sqrt(d_model)
        # attention_scores shape: [batch_size*h, seq_len, seq_len]
        attention_weights = self.drop(masked_softmax(attention_scores, valid_length))
        return torch.bmm(attention_weights, V)  # [batch_size*h, seq_len, d_model//h]

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout,):
        super(MultiHeadAttention, self).__init__()
        # 保证MultiHeadAttention的输入输出tensor的维度一样
        assert hidden_size % num_heads == 0
        # hidden_size => d_model
        self.num_heads = num_heads
        # num_heads => h
        self.hidden_size = hidden_size
        # 这里的d_model为中间隐层单元的神经元数目,d_model=h*d_v=h*d_k=h*d_q
        self.Wq = nn.Linear(input_size, hidden_size, bias=False)
        self.Wk = nn.Linear(input_size, hidden_size, bias=False)
        self.Wv = nn.Linear(input_size, hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention = DotProductAttention(dropout)

    def _transpose_qkv(self, X):
        # X的输入维度为[batch_size, seq_len, d_model]
        # 通过该函数将X的维度改变成[batch_size*num_heads, seq_len, d_model//num_heads]
        self._batch, self._seq_len = X.size()[0], X.size()[1] #[8,]
        X = X.view([self._batch, self._seq_len, self.num_heads, self.hidden_size//self.num_heads])  # [batch_size, seq_len, num_heads, d_model//num_heads]
        X = X.permute([0, 2, 1, 3])  # [batch_size, num_heads, seq_len, d_model//num_heads]
        return X.contiguous().view([self._batch*self.num_heads, self._seq_len, self.hidden_size//self.num_heads])

    def _transpose_output(self, X):
        X = X.view([self._batch, self.num_heads, -1, self.hidden_size//self.num_heads])
        X = X.permute([0, 2, 1, 3])
        return X.contiguous().view([self._batch, -1, self.hidden_size])

    def forward(self, query, key, value, valid_length):
        #b, ,  = key.size()[0]
        Q = self._transpose_qkv(self.Wq(query))
        K = self._transpose_qkv(self.Wk(key))
        V = self._transpose_qkv(self.Wv(value))
        # 由于输入的valid_length是相对batch输入的，而经过_transpose_qkv之后,
        # batch的大小发生了改变,Q的第一维度由原来的batch改为batch*num_heads
        # 因此,需要对valid_length进行复制,也就是进行np.title的操作
        if valid_length is not None:
            device = valid_length.device
            valid_length = valid_length.cpu().numpy() if valid_length.is_cuda else valid_length.numpy()
            if valid_length.ndim == 1:
                valid_length = np.tile(valid_length, self.num_heads)
            else:
                valid_length = np.tile(valid_length, [self.num_heads, 1])
            valid_length = torch.tensor(valid_length, dtype=torch.float, device=device)
        output = self.attention(Q, K, V, valid_length)
        output_concat = self._transpose_output(output)
        return self.Wo(output_concat)

class PositionWiseFFN(nn.Module):
    # y = w*[max(0, wx+b)]x+b
    def __init__(self, input_size, fft_hidden_size, output_size,):
        super(PositionWiseFFN, self).__init__()
        self.FFN1 = nn.Linear(input_size, fft_hidden_size)
        self.FFN2 = nn.Linear(fft_hidden_size, output_size)

    def forward(self, X):
        return self.FFN2(F.relu(self.FFN1(X)))

class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout,):
        super(AddNorm, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(hidden_size)

    def forward(self, X, Y):
        assert X.size() == Y.size()
        # return self.LN(self.drop(Y) + X)
        return  self.drop(Y)+X

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



class EncoderBlock(nn.Module):
    # 编码块由四部分构成,即多头注意力,addnorm,前馈神经网络,addnorm
    def __init__(self, input_size, ffn_hidden_size, num_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(input_size=input_size,
                                             hidden_size=input_size,
                                             num_heads=num_heads,
                                             dropout=dropout)

        self.addnorm1 = AddNorm(hidden_size=input_size, dropout=dropout, )
        self.ffn = PositionWiseFFN(input_size=input_size,
                                   fft_hidden_size=ffn_hidden_size,
                                   output_size=input_size, )
        self.addnorm2 = AddNorm(hidden_size=input_size, dropout=dropout, )
        self.power = PositionWiseFFN(input_size=input_size,
                                     fft_hidden_size=input_size,
                                     output_size=input_size, )

    def forward(self, X,X_FM,valid_length=None):
        atten_out = self.attention(query=X_FM, key=X_FM, value=X_FM, valid_length=valid_length)
        addnorm_out = self.addnorm1(X_FM, atten_out)
        ffn_out = self.ffn(addnorm_out)
        enc_out = self.addnorm2(addnorm_out,ffn_out)
        return enc_out
        # X_FM = self.addnorm2(addnorm_out, ffn_out)
        # X = self.power(X_FM.permute(0,2,1)).permute(0,2,1)
        # return X,X_FM

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, n_layers, ffn_hidden_size, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.ffn_hidden_size = ffn_hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        self.encoders = nn.ModuleList()
        for _ in range(self.n_layers):
            self.encoders.append(EncoderBlock(input_size=self.input_size,
                                              ffn_hidden_size=self.ffn_hidden_size,
                                              num_heads=self.num_heads,
                                              dropout=self.dropout, ))
        self.power = nn.Linear(input_size, input_size)

    def forward(self, X, X_FM,attn_mask=None):
        #word_embedding = self.word_embed(X)
        for i in range(self.n_layers):
            enc_out = self.encoders[i](X,X_FM, valid_length=attn_mask)
        return self.power(enc_out)
# ============================ 解码器实现 =================================
class DecoderBlock(nn.Module):
    def __init__(self, input_size, ffn_hidden_size, num_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadAttention(input_size=input_size,
                                             hidden_size=input_size,
                                             num_heads=num_heads,
                                             dropout=dropout)
        self.addnorm1 = AddNorm(hidden_size=input_size, dropout=dropout, )
        self.attention2 = MultiHeadAttention(input_size=input_size,
                                             hidden_size=input_size,
                                             num_heads=num_heads,
                                             dropout=dropout, )
        self.addnorm2 = AddNorm(hidden_size=input_size, dropout=dropout, )
        self.ffn = PositionWiseFFN(input_size=input_size,
                                   fft_hidden_size=ffn_hidden_size,
                                   output_size=input_size, )
        self.addnorm3 = AddNorm(hidden_size=input_size, dropout=dropout, )
        # self.addnorm4 = AddNorm(hidden_size=input_size, dropout=dropout, )

    def forward(self, X, state):
        enc_output, enc_valid_length = state[0], state[1]
        #batch_size, seq_len = X.size()[0], X.size()[1]
        #dec_valid_length = torch.tensor(np.tile(np.arange(1, seq_len+1), [batch_size, 1]),
        #                            dtype = torch.float, device = X.device)
        if self.training:  # 参数self自带
            batch_size, seq_len = X.size()[0], X.size()[1]
            dec_valid_length = torch.tensor(np.tile(np.arange(1, seq_len+1), [batch_size, 1]),
                                            dtype=torch.float, device=X.device)
        else:
           dec_valid_length = None

        attention_1_out = self.attention1(X, X, X, dec_valid_length)
        addnorm_1_out = self.addnorm1(X, attention_1_out)
        attention_2_out = self.attention2(addnorm_1_out, enc_output, enc_output, enc_valid_length)
        addnorm_2_out = self.addnorm2(addnorm_1_out, attention_2_out)
        ffn_out = self.ffn(addnorm_2_out)
        addnorm_3_out = self.addnorm3(addnorm_2_out, ffn_out)
        return addnorm_3_out, state

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, n_layers, ffn_hidden_size,
                 num_heads, dropout, output_size):
        super(TransformerDecoder, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.ffn_hidden_size = ffn_hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        self.decoders = nn.ModuleList()
        for _ in range(self.n_layers):
            self.decoders.append(DecoderBlock(input_size=self.input_size,
                                              ffn_hidden_size=self.ffn_hidden_size,
                                              num_heads=self.num_heads,
                                              dropout=self.dropout))
        self.dense = nn.Linear(input_size, output_size)

    def forward(self, dec_in, enc_out, dec_mask):
        for i in range(self.n_layers):
            dec_out = self.decoders[i](dec_in, enc_out, dec_mask)
        return self.dense(dec_out)

