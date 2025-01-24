import csv
import math
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
import ipdb
import matplotlib.pyplot as plt
import os
from datetime import datetime
INF = 1e10

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        self.inv_freq=self.inv_freq.to(device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class ScaleOffset(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        scale=True,
        offset=True,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset

        if self.scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if self.offset:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs):
        if self.scale:
            inputs = inputs * self.weight
        if self.offset:
            inputs = inputs + self.bias

        return inputs


class Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


class GatedAttentionUnit(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        attention_key_size=128,
        activation="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
        attention_dropout=0.1,
        my_gau_layer_num=-1,
        my_info_dict={},
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.intermediate_size = intermediate_size
        self.attention_key_size = attention_key_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout

        self.i_dense = nn.Linear(
            hidden_size, 2 * intermediate_size + attention_key_size, bias=self.use_bias
        )
        self.o_dense = nn.Linear(intermediate_size, hidden_size, bias=self.use_bias)

        self.q_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)


        self.my_gau_layer_num=my_gau_layer_num
        self.my_info_dict=my_info_dict
        self.task=my_info_dict["task"]
        self.dataset=my_info_dict["dataset"]
        self.algorithm=my_info_dict.get("algorithm",None)

        if self.task=="eval":
            self.model_name=my_info_dict["model_name"]
            self.train_len=my_info_dict["train_len"]
            self.seq_len=my_info_dict["seq_len"]
            self.train_norm_method=my_info_dict["train_norm_method"]
            self.train_normalization=my_info_dict["train_normalization"]
            
            self.norm_method=my_info_dict["eval_norm_method"]["name"]
            self.folder_name=f"[seq_len]{self.seq_len}_[{self.norm_method}]"
            
            if self.norm_method == "CosScale":
                self.norm_scale=my_info_dict["eval_norm_method"]["CosScale_value"]
                self.folder_name+=f"-s={self.norm_scale}"

            self.normalization=my_info_dict["eval_normalization"]["name"]
            self.folder_name+=f"_[{self.normalization}]"
            
            self.folder_name+=f"_[{self.algorithm}]"

            if self.algorithm=="ReRoPE":
                self.rerope_window_size=my_info_dict["rerope_window_size"]
                self.folder_name+=f"-w={self.rerope_window_size}"
            if self.algorithm=="StreamingLLM":
                self.streamingllm_mask=self.create_arrowlike_mask(self.seq_len, self.train_len-1,4, mask_left=True,mask_top=True)
            if self.algorithm=="LM-Infinite":
                self.lminfinite_mask=self.create_arrowlike_mask(self.seq_len, self.train_len-1,5, mask_left=True,mask_top=True)
            if self.model_name!="../junnyu/chinese_GAU-alpha-char_L-24_H-768":
                self.folder_name+=f"_[model]{re.search(r'^[^/]+/[^/]+/([^/]+)/', self.model_name).group(1)}]"

        elif self.task=="train":
            self.train_len=512
            self.seq_len=my_info_dict["train_len"]
            self.norm_method=my_info_dict["train_norm_method"]["name"]
            self.normalization=my_info_dict["train_normalization"]["name"]
            if self.norm_method == "CosScale":
                self.norm_scale=my_info_dict["train_norm_method"]["CosScale_value"]
        
        self.position_ids=torch.arange(self.seq_len).unsqueeze(0)
        self.cot=0

    @staticmethod
    def create_windows_mask(seq_len,extend_size=64):
        mask = torch.zeros((seq_len, seq_len), dtype=torch.int)
        for i in range(seq_len):
            lower_bound = max(0, i - extend_size)
            upper_bound = min(seq_len, i + extend_size + 1)
            mask[i, lower_bound:upper_bound] = 1
        return mask

    @staticmethod    
    def create_arrowlike_mask(seq_len, window_size,sinksize,mask_left,mask_top):
        mask = torch.zeros((seq_len, seq_len), dtype=torch.uint8)
        row_indices = torch.arange(seq_len).view(-1, 1)
        col_indices = torch.arange(seq_len).view(1, -1)
        diff = (row_indices - col_indices).abs() 
        mask[diff <= window_size] = 1
        if mask_left:
            mask[:, :sinksize] = 1
        if mask_top:
            mask[:sinksize, :] = 1
        return mask

    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    
    @staticmethod
    def apply_rotary2(x, sinusoidal_pos=None,w=0):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        sin = sin[:, w, :].expand(sin.shape)
        cos = cos[:, w, :].expand(cos.shape)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):  
        x = self.i_dense(hidden_states)
        u, v, qk = torch.split(
            self.activation(x),
            [self.intermediate_size, self.intermediate_size, self.attention_key_size],
            dim=-1,
        )
        q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)


        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓norm_method↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        if self.norm_method =="CosScale":
            q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)
            k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)
            q = q / (q_norm + 1e-9)
            k = k / (k_norm + 1e-9)
            q*=self.norm_scale
            q*=self.attention_key_size ** 0.5
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑norm_method↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


        if self.algorithm=="ReRoPE":
            rerope_window_size=self.rerope_window_size
            position_ids=torch.arange(self.seq_len).unsqueeze(0)
            q1, k1 = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(k, sinusoidal_pos)
            q2 = self.apply_rotary2(q, sinusoidal_pos, rerope_window_size-1)
            attn_weights1 = torch.einsum("bmd,bnd->bmn", q1, k1)
            attn_weights2 = torch.einsum("bmd,bnd->bmn", q2, k)
            attn_weights1=attn_weights1.to(q.device)
            attn_weights2=attn_weights2.to(q.device)
            rectified_mask = (position_ids[:, -self.seq_len:, None] - position_ids[:, None]).abs() < rerope_window_size
            rectified_mask=rectified_mask.to(q.device)
            a = torch.where(rectified_mask, attn_weights1, attn_weights2)
        elif self.algorithm=="ALiBi":
            def get_slopes(n):
                def get_slopes_power_of_2(n):
                    start = (2**(-2**-(math.log2(n)-3)))
                    ratio = start
                    return [start*ratio**i for i in range(n)]

                if math.log2(n).is_integer():
                    return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
                else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                    closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                    return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
            maxpos=self.seq_len
            attn_heads=1

            context_position = torch.arange(maxpos)[:, None]
            memory_position = torch.arange(maxpos)[None, :]
            relative_position = memory_position - context_position 
            relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1,-1)
            slopes = torch.Tensor(get_slopes(attn_heads))*-1
            alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
            alibi = alibi.view(attn_heads, maxpos, maxpos)

            a = torch.einsum("bmd,bnd->bmn", q, k)
            a += alibi.to(a)
        elif self.algorithm=="StreamingLLM":
            self.streamingllm_mask=self.streamingllm_mask.unsqueeze(0).expand(q.shape[0], -1, -1)
            self.streamingllm_mask=self.streamingllm_mask.to(q.device)
            q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(k, sinusoidal_pos)
            a = torch.einsum("bmd,bnd->bmn", q, k)
            a = torch.where(self.streamingllm_mask, a, -INF)
        elif self.algorithm=="LM-Infinit":
            lminfinite_window_size=self.train_len
            position_ids=torch.arange(self.seq_len).unsqueeze(0)
            q1, k1 = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(k, sinusoidal_pos)
            q2 = self.apply_rotary2(q, sinusoidal_pos, lminfinite_window_size-1)
            attn_weights1 = torch.einsum("bmd,bnd->bmn", q1, k1)
            attn_weights2 = torch.einsum("bmd,bnd->bmn", q2, k)
            attn_weights1=attn_weights1.to(q.device)
            attn_weights2=attn_weights2.to(q.device)
            rectified_mask = (position_ids[:, -self.seq_len:, None] - position_ids[:, None]).abs() < lminfinite_window_size
            rectified_mask=rectified_mask.to(q.device)
            a = torch.where(rectified_mask, attn_weights1, attn_weights2)

            self.lminfinite_mask= self.lminfinite_mask.unsqueeze(0).expand(q.shape[0], -1, -1)
            self.lminfinite_mask=self.lminfinite_mask.to(q.device)
            a = torch.where(self.lminfinite_mask, a, -INF)
        else:
            q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(k, sinusoidal_pos)
            a = torch.einsum("bmd,bnd->bmn", q, k)


        if self.norm_method=="Windowed-Attention":
            self.norm_windows_mask = self.norm_windows_mask.unsqueeze(0).expand(a.shape[0], -1, -1)
            self.norm_windows_mask=self.norm_windows_mask.to(a.device)
            a = torch.where(self.norm_windows_mask, a, -INF)


        if self.attention_scale:
            a = a / self.attention_key_size ** 0.5

        if attention_mask is not None:
            a = a.masked_fill(attention_mask == 0, -INF)
            l = attention_mask.sum(-1, keepdim=True)
        else:
            l = torch.ones_like(a) * x.shape[1]


        # A = attention_normalize(a, l, dim=-1, method=self.normalization)
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓InfoScale↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        assert self.normalization in ["InfoScale", "softmax", "softmax_plus"]
        if self.normalization == "softmax":
            A =torch.softmax(a, dim=-1)
        elif self.normalization == "softmax_plus":
                A =torch.softmax(a * torch.log(l) / np.log(512), dim=-1)
        elif self.normalization == "InfoScale":
                train_len_ori = self.train_len
                train_len = self.seq_len
                head_dim = q.shape[2]
                InfoScale_clip = (1-(train_len_ori**(-2/(head_dim))))**0.5 
                InfoScale = (1-(l**(-2/(head_dim))))**0.5
                InfoScale_tensor = torch.tensor(InfoScale, device=a.device, dtype=a.dtype) / torch.tensor(InfoScale_clip, device=a.device, dtype=a.dtype)
                a = a * InfoScale_tensor
                A = torch.softmax(a, dim=-1)
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑InfoScale↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


        A = F.dropout(A, p=self.attention_dropout, training=self.training)

        o = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))

        outputs = (o, A) if output_attentions else (o,)
        return outputs


class GAULayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        attention_key_size=128,
        activation="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        eps=1e-12,
        my_gau_layer_num=-1,
        my_info_dict={},
    ):
        super().__init__()
        self.gau = GatedAttentionUnit(
            hidden_size,
            intermediate_size,
            attention_key_size,
            activation,
            use_bias,
            normalization,
            attention_scale,
            attention_dropout,
            my_gau_layer_num=my_gau_layer_num,
            my_info_dict=my_info_dict,
        )
        self.norm = Norm(eps=eps)
        self.hidden_dropout = hidden_dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):
        gau_output = self.gau(
            hidden_states, attention_mask, sinusoidal_pos, output_attentions
        )
        o = F.dropout(gau_output[0], p=self.hidden_dropout, training=self.training)
        o = self.norm(hidden_states + o)

        outputs = (o,) + gau_output[1:] 

        return outputs
    