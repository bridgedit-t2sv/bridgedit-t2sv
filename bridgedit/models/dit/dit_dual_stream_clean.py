# --------------------------------------------------------
# References:
# GLIDE:    https://github.com/openai/glide-text2im
# MAE:      https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT:      https://github.com/facebookresearch/DiT
# --------------------------------------------------------
import math
import numpy as np
import collections.abc
from einops import rearrange
from itertools import repeat
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Copied from diffusers.models.embeddings
# https://github.com/huggingface/diffusers/blob/c4a8979f3018fbffee33304c1940561f7a5cf613/src/diffusers/models/embeddings.py
def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
    
        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        # print(x.shape, cos.shape, sin.shape, x_rotated.shape)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    
        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    
        return x_out.type_as(x)


# Modified from perceiver-io implemented by krasserm
# https://github.com/krasserm/perceiver-io/blob/main/perceiver/model/core/modules.py
class DualMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_input_channels_1: int,
        num_input_channels_2: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels_1: Optional[int] = None,
        num_output_channels_2: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()
    
        if num_qk_channels is None:
            num_qk_channels = np.min([num_input_channels_1, num_input_channels_2])
    
        if num_v_channels is None:
            num_v_channels = num_qk_channels
    
        if num_output_channels_1 is None:
            num_output_channels_1 = num_input_channels_1
        if num_output_channels_2 is None:
            num_output_channels_2 = num_input_channels_2
    
        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")
    
        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")
    
        num_qk_channels_per_head = num_qk_channels // num_heads
    
        self.dp_scale = num_qk_channels_per_head**-0.5
        self.num_heads = num_heads
        self.num_qk_channels = num_qk_channels
        self.num_v_channels = num_v_channels
        self.causal_attention = causal_attention
    
        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel
    
        self.q_proj_1 = nn.Linear(num_input_channels_1, num_qk_channels, bias=qkv_bias)
        self.k_proj_1 = nn.Linear(num_input_channels_1, num_qk_channels, bias=qkv_bias)
        self.v_proj_1 = nn.Linear(num_input_channels_1, num_v_channels, bias=qkv_bias)
        self.o_proj_1 = nn.Linear(num_v_channels, num_output_channels_1, bias=out_bias)
        
        self.q_proj_2 = nn.Linear(num_input_channels_2, num_qk_channels, bias=qkv_bias)
        self.k_proj_2 = nn.Linear(num_input_channels_2, num_qk_channels, bias=qkv_bias)
        self.v_proj_2 = nn.Linear(num_input_channels_2, num_v_channels, bias=qkv_bias)
        self.o_proj_2 = nn.Linear(num_v_channels, num_output_channels_2, bias=out_bias)
        
        self.dropout = nn.Dropout(dropout)
            
    def forward(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb_1 = None,
        rot_pos_emb_2 = None,
        independent_attention = None,
        cross_only_attention = None,
    ):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length and D the
                number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence length and C
                are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length and F the
                number of output channels (= `num_output_channels`)
        """
        
        if isinstance(independent_attention, bool):
            independent_attention = torch.tensor([independent_attention] * x_1.size(0), device=x_1.device)
        if isinstance(cross_only_attention, bool):
            cross_only_attention = torch.tensor([cross_only_attention] * x_1.size(0), device=x_1.device)
        assert independent_attention is None or len(independent_attention) == x_1.size(0), "Assert: independent_attention size must be equal to batch size."
        assert cross_only_attention is None or len(cross_only_attention) == x_1.size(0), "Assert: cross_only_attention size must be equal to batch size."
        if independent_attention is not None and cross_only_attention is not None:
            for i in range(len(independent_attention)):
                if independent_attention[i] and cross_only_attention[i]:
                    raise ValueError("Error: independent_attention and cross_only_attention cannot be True at the same time.")
        
        q_1 = self.q_proj_1(x_1)
        k_1 = self.k_proj_1(x_1)
        v_1 = self.v_proj_1(x_1)
        
        q_2 = self.q_proj_2(x_2)
        k_2 = self.k_proj_2(x_2)
        v_2 = self.v_proj_2(x_2)
        
        len_1 = q_1.shape[1]
        q = torch.cat([q_1, q_2], dim=1)
        k = torch.cat([k_1, k_2], dim=1)
        v = torch.cat([v_1, v_2], dim=1)
    
        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale
        if rot_pos_emb_2 is None:
            q = apply_rotary_emb(q, rot_pos_emb_1)
            k = apply_rotary_emb(k, rot_pos_emb_1)
        else:   
        
            q[:, :, :len_1] = apply_rotary_emb(q[:, :, :len_1], rot_pos_emb_1)
            k[:, :, :len_1] = apply_rotary_emb(k[:, :, :len_1], rot_pos_emb_1)
        
            q[:, :, len_1:] = apply_rotary_emb(q[:, :, len_1:], rot_pos_emb_2)
            k[:, :, len_1:] = apply_rotary_emb(k[:, :, len_1:], rot_pos_emb_2)
    
        # if rot_pos_emb_1:
        #     q = apply_rotary_emb(q, rot_pos_emb_1)
        #     k = apply_rotary_emb(k, rot_pos_emb_1)


        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")
    
        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]
    
            # If q and k have different length, causal masking only works if they are right-aligned.
            causal_mask = torch.ones((i, j), device=x_1.device, dtype=torch.bool).triu(j - i + 1)
        
        if independent_attention.all() or cross_only_attention.all():
            attn_mask = torch.zeros((q.shape[2], k.shape[2]), device=q.device, dtype=torch.bool)
            attn_mask[:len_1, :len_1] = 1
            attn_mask[len_1:, len_1:] = 1
    
        o_chunks = []
    
        # Only process a given maximum number of heads in
        # parallel, using several iterations, if necessary.
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(self.max_heads_parallel, dim=1),
            k.split(self.max_heads_parallel, dim=1),
            v.split(self.max_heads_parallel, dim=1),
        ):
            ''' Raw attention computation '''
            # attn = torch.einsum("b h i c, b h j c -> b h i j", q_chunk, k_chunk)
            # attn_max_neg = -torch.finfo(attn.dtype).max
    
            # if pad_mask is not None:
            #     attn.masked_fill_(pad_mask, attn_max_neg)
    
            # if self.causal_attention:
            #     attn.masked_fill_(causal_mask, attn_max_neg)
                
            # if independent_attention:
            #     for i in range(len(independent_attention)):
            #         if independent_attention[i]:
            #             attn[i].masked_fill_(~attn_mask, attn_max_neg)
            # if cross_only_attention:
            #     for i in range(len(cross_only_attention)):
            #         if cross_only_attention[i]:
            #             attn[i].masked_fill_(attn_mask, attn_max_neg)
    
            # attn = attn.softmax(dim=-1)
            # attn = self.dropout(attn)
    
            # o_chunk = torch.einsum("b h i j, b h j c -> b h i c", attn, v_chunk)
    
            ''' Flash attention computation '''
            _q = q_chunk.transpose(1, 2)
            _k = k_chunk.transpose(1, 2)
            _v = v_chunk.transpose(1, 2)
            _o = flash_attn_func(_q, _k, _v, dropout_p=0.0, softmax_scale=None, causal=False)
            o_chunk = _o.transpose(1, 2)
    
            o_chunks.append(o_chunk)

            # o1_fa = flash_attn_func(q1_fa, k1_fa, v1_fa, dropout_p=self.dropout.p, softmax_scale=self.dp_scale, causal=False)
        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)
        o1 = self.o_proj_1(o[:, :len_1])
        o2 = self.o_proj_2(o[:, len_1:])
    
        # return o, kv_cache
        return o1, o2

class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_input_channels_1: int, # e.g., Video
        num_input_channels_2: int, # e.g., Audio
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels_1: Optional[int] = None,
        num_output_channels_2: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False, # Causal not typically used in cross-attn
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        flag: str = "bicross", # bicross, v2a, a2v
    ):
        """
        Multi-head attention implementing the bi-directional cross-attention structure.
        - Stream 1 (e.g., Video) queries Stream 2 (e.g., Audio).
        - Stream 2 (e.g., Audio) queries Stream 1 (e.g., Video).
        """
        super().__init__()
    
        if num_qk_channels is None:
            num_qk_channels = np.min([num_input_channels_1, num_input_channels_2])
    
        if num_v_channels is None:
            num_v_channels = num_qk_channels
    
        if num_output_channels_1 is None:
            num_output_channels_1 = num_input_channels_1
        if num_output_channels_2 is None:
            num_output_channels_2 = num_input_channels_2
    
        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")
    
        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")
    
        num_qk_channels_per_head = num_qk_channels // num_heads
    
        self.dp_scale = num_qk_channels_per_head**-0.5
        self.num_heads = num_heads
        self.flag = flag
        # Projections for Stream 1 (e.g., Video)
        self.q_proj_1 = nn.Linear(num_input_channels_1, num_qk_channels, bias=qkv_bias)
        self.k_proj_1 = nn.Linear(num_input_channels_1, num_qk_channels, bias=qkv_bias)
        self.v_proj_1 = nn.Linear(num_input_channels_1, num_v_channels, bias=qkv_bias)
        self.o_proj_1 = nn.Linear(num_v_channels, num_output_channels_1, bias=out_bias)
        
        # Projections for Stream 2 (e.g., Audio)
        self.q_proj_2 = nn.Linear(num_input_channels_2, num_qk_channels, bias=qkv_bias)
        self.k_proj_2 = nn.Linear(num_input_channels_2, num_qk_channels, bias=qkv_bias)
        self.v_proj_2 = nn.Linear(num_input_channels_2, num_v_channels, bias=qkv_bias)
        self.o_proj_2 = nn.Linear(num_v_channels, num_output_channels_2, bias=out_bias)
        
        self.dropout = nn.Dropout(dropout)
            
    def forward(
        self,
        x_1: torch.Tensor, # Video Latent
        x_2: torch.Tensor, # Audio Latent
        pad_mask: Optional[torch.Tensor] = None, # Note: pad_mask needs careful handling for two seqs
        rot_pos_emb_1 = None, # Positional embedding for x_1
        rot_pos_emb_2 = None, # Positional embedding for x_2
    ):
        # Step 1: Project inputs to Q, K, V for both streams
        # Video projections
        q1 = self.q_proj_1(x_1)
        k1 = self.k_proj_1(x_1)
        v1 = self.v_proj_1(x_1)
        
        # Audio projections
        q2 = self.q_proj_2(x_2)
        k2 = self.k_proj_2(x_2)
        v2 = self.v_proj_2(x_2)
        
        # Step 2: Reshape for multi-head attention
        q1, k1, v1 = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q1, k1, v1])
        q2, k2, v2 = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q2, k2, v2])

        # Step 3: Apply Rotary Position Embeddings if provided
        if rot_pos_emb_1:
            q1 = apply_rotary_emb(q1, rot_pos_emb_1)
            k1 = apply_rotary_emb(k1, rot_pos_emb_1)
        if rot_pos_emb_2:
            q2 = apply_rotary_emb(q2, rot_pos_emb_2)
            k2 = apply_rotary_emb(k2, rot_pos_emb_2)

        # Step 4: Perform the two separate cross-attention calculations using Flash Attention 
        if self.flag == "a2v" or self.flag == "bicross":
            # Attention 1: Audio-to-Video (Video is Query, Audio is Key/Value)
            # Reshape for Flash Attention's expected input (B, S, H, D)
            q1_fa = q1.transpose(1, 2)
            k2_fa = k2.transpose(1, 2)
            v2_fa = v2.transpose(1, 2)
            # Calculate attention
            o1_fa = flash_attn_func(q1_fa, k2_fa, v2_fa, dropout_p=self.dropout.p, softmax_scale=self.dp_scale, causal=False)
            # Reshape back to (B, H, S, D)
            o1 = o1_fa.transpose(1, 2)
        else:
            # Attention 1: Self-attention for Video
            q1_fa = q1.transpose(1, 2)
            k1_fa = k1.transpose(1, 2)
            v1_fa = v1.transpose(1, 2)
            o1_fa = flash_attn_func(q1_fa, k1_fa, v1_fa, dropout_p=self.dropout.p, softmax_scale=self.dp_scale, causal=False)
            o1 = o1_fa.transpose(1, 2)
        
        if self.flag == "v2a" or self.flag == "bicross":
            # Attention 2: Video-to-Audio (Audio is Query, Video is Key/Value)
            # Reshape for Flash Attention
            q2_fa = q2.transpose(1, 2)
            k1_fa = k1.transpose(1, 2)
            v1_fa = v1.transpose(1, 2)
            # Calculate attention
            o2_fa = flash_attn_func(q2_fa, k1_fa, v1_fa, dropout_p=self.dropout.p, softmax_scale=self.dp_scale, causal=False)
            # Reshape back
            o2 = o2_fa.transpose(1, 2)
        else:
            # Attention 2: Self-attention for Audio
            q2_fa = q2.transpose(1, 2)
            k2_fa = k2.transpose(1, 2)
            v2_fa = v2.transpose(1, 2)
            o2_fa = flash_attn_func(q2_fa, k2_fa, v2_fa, dropout_p=self.dropout.p, softmax_scale=self.dp_scale, causal=False)
            o2 = o2_fa.transpose(1, 2)

        # Step 5: Reshape and apply final output projections
        o1 = rearrange(o1, "b h n c -> b n (h c)")
        o2 = rearrange(o2, "b h n c -> b n (h c)")
        
        output_1 = self.o_proj_1(o1)
        output_2 = self.o_proj_2(o2)
        
        return output_1, output_2

class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """
        :param num_heads: Number of attention heads.
        :param num_input_channels: Number of input channels.
        :param num_qk_channels: Number of query and key channels. Default is `num_input_channels`.
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_input_channels`.
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`.
        :param qkv_bias: Whether to use a bias term for query, key, and value projections. Default is `True`.
        :param out_bias: Whether to use a bias term for the output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head**-0.5
        self.num_heads = num_heads
        self.causal_attention = causal_attention

        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel

        # Projections for Query, Key, and Value from the single input tensor
        self.q_proj = nn.Linear(num_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[torch.Tensor] = None,
    ):
        """
        :param x: Input tensor of shape (B, N, D) where B is the batch size, N the sequence length, and D is the
                  number of input channels (`num_input_channels`).
        :param pad_mask: Boolean key padding mask of shape (B, N). `True` values indicate padding tokens.
        :param rot_pos_emb: Rotary position embedding tensor. If defined, it's applied to queries and keys.
        :return: Attention result of shape (B, N, F) where F is the number of output channels (`num_output_channels`).
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = (rearrange(t, "b n (h c) -> b h n c", h=self.num_heads) for t in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb is not None:
            q = apply_rotary_emb(q, rot_pos_emb)
            k = apply_rotary_emb(k, rot_pos_emb)

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")

        
        o_chunks = []
        
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(self.max_heads_parallel, dim=1),
            k.split(self.max_heads_parallel, dim=1),
            v.split(self.max_heads_parallel, dim=1),
        ):
            ''' Flash attention computation '''
            _q = q_chunk.transpose(1, 2)
            _k = k_chunk.transpose(1, 2)
            _v = v_chunk.transpose(1, 2)
            _o = flash_attn_func(_q, _k, _v, dropout_p=0.0, softmax_scale=None, causal=False,
                                window_size=(-1, -1), alibi_slopes=None, deterministic=False)
    
            o_chunk = _o.transpose(1, 2)
    
            o_chunks.append(o_chunk)
        
        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)")
        o = self.o_proj(o)
        
        return o


# Copied from timm.models.vision_transformer
class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Modified from Meta DiT
class DualDiTBlockAdaLN(nn.Module):
    """
    A DiT block with expert modules.
    """
    def __init__(
        self, 
        num_input_channels_1: int, 
        num_input_channels_2: int, 
        num_qk_channels : int,
        num_v_channels : int,
        num_heads: int, 
        t_emb_dim_1: int,
        t_emb_dim_2: int,
        mlp_ratio: float = 4.0, 
        **block_kwargs
    ):
        super().__init__()
        self.norm_attn_1 = nn.LayerNorm(num_input_channels_1, elementwise_affine=False, eps=1e-6)
        self.norm_attn_2 = nn.LayerNorm(num_input_channels_2, elementwise_affine=False, eps=1e-6)
        self.attn = DualMultiHeadAttention(
            num_heads = num_heads,
            num_input_channels_1 = num_input_channels_1,
            num_input_channels_2 = num_input_channels_2,
            num_qk_channels = num_qk_channels,
            num_v_channels = num_v_channels,
            num_output_channels_1 = num_input_channels_1,
            num_output_channels_2 = num_input_channels_2,
            **block_kwargs
        )
        self.norm_mlp_1 = nn.LayerNorm(num_input_channels_1, elementwise_affine=False, eps=1e-6)
        self.norm_mlp_2 = nn.LayerNorm(num_input_channels_2, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim_1 = int(num_input_channels_1 * mlp_ratio)
        mlp_hidden_dim_2 = int(num_input_channels_2 * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_1 = MLP(in_features=num_input_channels_1, hidden_features=mlp_hidden_dim_1, act_layer=approx_gelu, drop=0.)
        self.mlp_2 = MLP(in_features=num_input_channels_2, hidden_features=mlp_hidden_dim_2, act_layer=approx_gelu, drop=0.)
        self.adaln_modulation_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim_1, 6 * num_input_channels_1, bias=True)
        )
        self.adaln_modulation_2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim_2, 6 * num_input_channels_2, bias=True)
        )
        
    def forward(
        self, 
        sample_1: torch.Tensor,
        sample_2: torch.Tensor,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        # Attention parameters.
        rot_pos_emb_1: torch.Tensor,
        rot_pos_emb_2: torch.Tensor=None,
        independent_attention: bool = False,
        cross_only_attention: bool = False,
    ):
        # Modulate stream 1.
        shift_msa_1, scale_msa_1, gate_msa_1, shift_mlp_1, scale_mlp_1, gate_mlp_1 = self.adaln_modulation_1(emb_1).chunk(6, dim=1)
        # Modulate stream 2.
        shift_msa_2, scale_msa_2, gate_msa_2, shift_mlp_2, scale_mlp_2, gate_mlp_2 = self.adaln_modulation_2(emb_2).chunk(6, dim=1)
        
        # Forward full attention.
        sample_1_adaln = modulate(self.norm_attn_1(sample_1), shift_msa_1, scale_msa_1)
        sample_2_adaln = modulate(self.norm_attn_2(sample_2), shift_msa_2, scale_msa_2)
        _attn_1, _attn_2 = self.attn(
            sample_1_adaln, 
            sample_2_adaln, 
            rot_pos_emb_1 = rot_pos_emb_1,
            rot_pos_emb_2 = rot_pos_emb_2,
            independent_attention = independent_attention,
            cross_only_attention = cross_only_attention,
        )
    
        sample_1 = sample_1 + gate_msa_1.unsqueeze(1) * _attn_1
        sample_2 = sample_2 + gate_msa_2.unsqueeze(1) * _attn_2
        # Forward mlp.
        sample_1_adaln = modulate(self.norm_mlp_1(sample_1), shift_mlp_1, scale_mlp_1)
        sample_2_adaln = modulate(self.norm_mlp_2(sample_2), shift_mlp_2, scale_mlp_2)
        sample_1 = sample_1 + gate_mlp_1.unsqueeze(1) * self.mlp_1(sample_1_adaln)
        sample_2 = sample_2 + gate_mlp_2.unsqueeze(1) * self.mlp_2(sample_2_adaln)
        
        return sample_1, sample_2

class CrossDiTBlockAdaLN(nn.Module):
    """
    A DiT block with expert modules.
    """
    def __init__(
        self, 
        num_input_channels_1: int, 
        num_input_channels_2: int, 
        num_qk_channels : int,
        num_v_channels : int,
        num_heads: int, 
        t_emb_dim_1: int,
        t_emb_dim_2: int,
        flag: str = "bicross", # bicross, v2a, a2v
        mlp_ratio: float = 4.0, 
        **block_kwargs
    ):
        super().__init__()
        self.norm_attn_1 = nn.LayerNorm(num_input_channels_1, elementwise_affine=False, eps=1e-6)
        self.norm_attn_2 = nn.LayerNorm(num_input_channels_2, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention(
            num_heads = num_heads,
            num_input_channels_1 = num_input_channels_1,
            num_input_channels_2 = num_input_channels_2,
            num_qk_channels = num_qk_channels,
            num_v_channels = num_v_channels,
            num_output_channels_1 = num_input_channels_1,
            num_output_channels_2 = num_input_channels_2,
            flag = flag,
            **block_kwargs
        )
        self.norm_mlp_1 = nn.LayerNorm(num_input_channels_1, elementwise_affine=False, eps=1e-6)
        self.norm_mlp_2 = nn.LayerNorm(num_input_channels_2, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim_1 = int(num_input_channels_1 * mlp_ratio)
        mlp_hidden_dim_2 = int(num_input_channels_2 * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_1 = MLP(in_features=num_input_channels_1, hidden_features=mlp_hidden_dim_1, act_layer=approx_gelu, drop=0.)
        self.mlp_2 = MLP(in_features=num_input_channels_2, hidden_features=mlp_hidden_dim_2, act_layer=approx_gelu, drop=0.)
        self.adaln_modulation_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim_1, 6 * num_input_channels_1, bias=True)
        )
        self.adaln_modulation_2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim_2, 6 * num_input_channels_2, bias=True)
        )
        
    def forward(
        self, 
        sample_1: torch.Tensor,
        sample_2: torch.Tensor,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        # Attention parameters.
        rot_pos_emb_1: torch.Tensor,
        rot_pos_emb_2: torch.Tensor=None,
        independent_attention: bool = False,
        cross_only_attention: bool = False,
    ):
        # Modulate stream 1.
        shift_msa_1, scale_msa_1, gate_msa_1, shift_mlp_1, scale_mlp_1, gate_mlp_1 = self.adaln_modulation_1(emb_1).chunk(6, dim=1)
        # Modulate stream 2.
        shift_msa_2, scale_msa_2, gate_msa_2, shift_mlp_2, scale_mlp_2, gate_mlp_2 = self.adaln_modulation_2(emb_2).chunk(6, dim=1)
        
        # Forward full attention.
        sample_1_adaln = modulate(self.norm_attn_1(sample_1), shift_msa_1, scale_msa_1)
        sample_2_adaln = modulate(self.norm_attn_2(sample_2), shift_msa_2, scale_msa_2)
        _attn_1, _attn_2 = self.attn(
            sample_1_adaln, 
            sample_2_adaln, 
            rot_pos_emb_1 = rot_pos_emb_1,
            rot_pos_emb_2 = rot_pos_emb_2
        )
    
        sample_1 = sample_1 + gate_msa_1.unsqueeze(1) * _attn_1
        sample_2 = sample_2 + gate_msa_2.unsqueeze(1) * _attn_2
        # Forward mlp.
        sample_1_adaln = modulate(self.norm_mlp_1(sample_1), shift_mlp_1, scale_mlp_1)
        sample_2_adaln = modulate(self.norm_mlp_2(sample_2), shift_mlp_2, scale_mlp_2)
        sample_1 = sample_1 + gate_mlp_1.unsqueeze(1) * self.mlp_1(sample_1_adaln)
        sample_2 = sample_2 + gate_mlp_2.unsqueeze(1) * self.mlp_2(sample_2_adaln)
        
        return sample_1, sample_2

class AddFusionDiTBlockAdaLN(nn.Module):
    def __init__(
        self, 
        num_input_channels_1: int, 
        num_input_channels_2: int,     
        t_emb_dim_1: int,
        t_emb_dim_2: int,
        mlp_ratio: float = 4.0, 
        **block_kwargs
    ):
        super().__init__()
        # Additive fusion
        self.video_to_audio_proj = nn.Linear(num_input_channels_1, num_input_channels_2)
        self.audio_to_video_proj = nn.Linear(num_input_channels_2, num_input_channels_1)

        self.norm_mlp_1 = nn.LayerNorm(num_input_channels_1, elementwise_affine=False, eps=1e-6)
        self.norm_mlp_2 = nn.LayerNorm(num_input_channels_2, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim_1 = int(num_input_channels_1 * mlp_ratio)
        mlp_hidden_dim_2 = int(num_input_channels_2 * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_1 = MLP(in_features=num_input_channels_1, hidden_features=mlp_hidden_dim_1, act_layer=approx_gelu, drop=0.)
        self.mlp_2 = MLP(in_features=num_input_channels_2, hidden_features=mlp_hidden_dim_2, act_layer=approx_gelu, drop=0.)
        self.adaln_modulation_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim_1, 4 * num_input_channels_1, bias=True)
        )
        self.adaln_modulation_2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim_2, 4 * num_input_channels_2, bias=True)
        )
        # self.attn_1 = SelfAttention(
        #     num_heads = num_heads,
        #     num_input_channels = num_input_channels_1,
        #     num_qk_channels = num_qk_channels,
        #     num_v_channels = num_v_channels,
        #     num_output_channels = num_input_channels_1,
        # )
        # self.attn_2 = SelfAttention(
        #     num_heads = num_heads,
        #     num_input_channels = num_input_channels_2,
        #     num_qk_channels = num_qk_channels,
        #     num_v_channels = num_v_channels,
        #     num_output_channels = num_input_channels_2,
        # )
    def forward(
        self, 
        sample_1: torch.Tensor,
        sample_2: torch.Tensor,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        # Attention parameters.
        rot_pos_emb_1: torch.Tensor,
        rot_pos_emb_2: torch.Tensor=None,
    ):
        # 1. 生成 AdaLN 参数
        gate_fusion_1, shift_mlp_1, scale_mlp_1, gate_mlp_1 = self.adaln_modulation_1(emb_1).chunk(4, dim=1)
        gate_fusion_2, shift_mlp_2, scale_mlp_2, gate_mlp_2 = self.adaln_modulation_2(emb_2).chunk(4, dim=1)

        global_audio_info = self.audio_to_video_proj(sample_2.mean(dim=1))
        global_video_info = self.video_to_audio_proj(sample_1.mean(dim=1))
        sample_1 = sample_1 + gate_fusion_1.unsqueeze(1) * global_audio_info.unsqueeze(1)
        sample_2 = sample_2 + gate_fusion_2.unsqueeze(1) * global_video_info.unsqueeze(1)

        # 3. --- MLP 独立处理路径 ---
        # (a) 视频流MLP
        sample_1_norm = self.norm_mlp_1(sample_1)
        sample_1_adaln = modulate(sample_1_norm, shift_mlp_1, scale_mlp_1)
        mlp_output_1 = self.mlp_1(sample_1_adaln)
        sample_1 = sample_1 + gate_mlp_1.unsqueeze(1) * mlp_output_1
        
        # (b) 音频流MLP
        sample_2_norm = self.norm_mlp_2(sample_2)
        sample_2_adaln = modulate(sample_2_norm, shift_mlp_2, scale_mlp_2)
        mlp_output_2 = self.mlp_2(sample_2_adaln)
        sample_2 = sample_2 + gate_mlp_2.unsqueeze(1) * mlp_output_2

        return sample_1, sample_2
        
        
        
        # # Forward mlp.
        # sample_1_adaln = modulate(self.norm_mlp_1(sample_1), shift_mlp_1, scale_mlp_1)
        # sample_2_adaln = modulate(self.norm_mlp_2(sample_2), shift_mlp_2, scale_mlp_2)
        # sample_1 = sample_1 + gate_mlp_1.unsqueeze(1) * self.mlp_1(sample_1_adaln)
        # sample_2 = sample_2 + gate_mlp_2.unsqueeze(1) * self.mlp_2(sample_2_adaln)
        
        # return sample_1, sample_2