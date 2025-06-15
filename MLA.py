import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dataclasses import dataclass
from typing import Tuple, Optional, Literal


world_size = 1




@dataclass
class ModelConfig:
    """
    model arguments and hyperparameters.
    Attributes:
    B (int): Max batch size
    T (int): Max sequence length
    dtype (Literal["bf16", "fp8"]): Data type for computations.
    vocab_size (int):  Vocabulary size
    dim (int): Embedding dimension
    """

    B: int = 8
    T: int = 2048
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 50304
    dim: int = 1024
    n_heads: int = 16
    #MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128      # dim = v_head_dim * n_head
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.dim   
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
    
        self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = F.rms_norm(self.kv_lora_rank, eps = 1e-6)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.register_buffer("k_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for MLA

        config:
            x (torch.Tensor): Input tensor of shape (B, T)
            start_pos (int): Starting position in the sequence of caching.
            freqs_cis (torch.Tensor): Precomputerd complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exculde certain positions from attention.
        
        Returns:
            torch.Tensor: Output tensor with the same shape as the input mask
        """

        B, T, n_embed = x.size()
        end_pos = start_pos + T
        q = self.wq(x)                                              # (B, T, n_embed) -> (B, T, n_heads * qk_head_dim)
        q.view(B, T, self.n_heads, self.qk_head_dim)                # (B, T, n_head * qk_head_dim) -> (B, T, n_heads, qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)   # (B, T, n_heads, qk_head_dim) -> (B, T, n_heads, qk_nope_head_dim) + (B, T, n_heads, qk_rope_head_dim)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)                    # (B, T, n_heads, qk_rope_head_dim) -> (B, T, n_heads, qk_rope_head_dim)
        kv = self.wkv_a(x)                                          # (B, T, n_embed) -> (B, T, kv_lora_rank + qk_rope_head_dim)
        kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)          # (B, T, kv_lora_rank + qk_rope_head_dim) -> (B, T, kv_lora_rank) + (B, T, qk_rope_head_dim)
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)       # (B, T, qk_rope_head_dim) -> (B, T, qk_rope_head_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)                     # (B, T, n_heads, qk_head_dim)
        kv = self.wkv_b(self.kv_norm(kv))                           # (B, T, kv_lora_rank) -> (B, T, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim) # (B, T, n_heads * (qk_nope_head_dim + v_head_dim)) -> (B, T, n_heads, (qk_nope_head_dim + v_head_dim))
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)    # (B, T, n_heads, (qk_nope_head_dim + v_head_dim)) -> (B, T, n_heads, qk_nope_head_dim)  +  (B, T, n_heads, v_head_dim)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_heads, -1)], dim=-1)   # (B, T, n_heads, qk_nope_head_dim) + (B, T, n_heads, qk_rope_head_dim) -> (B, T, n_heads, qk_nope_head_dim + qk_rope_head_dim) = qk_head_dim
        self.k_cache[:B, start_pos:end_pos] = k                     
        self.v_cache[:B, start_pos:end_pos] = v

        x = F.scaled_dot_product_attention(
            q, 
            self.k_cache[:B, :end_pos], 
            self.v_cache[:B, :end_pos],
            attn_mask=mask,
            is_causal=mask is None,
            scale=self.softmax_scale
        )
        x = self.wo(x.flatten(2))                                   # (B, T, n_embed)
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 =  nn.Linear(config.dim, 4 * config.dim)
        self.w2 = nn.Linear(4 * config.dim, config.dim)
        self.w3 = nn.Linear(config.dim, 4 * config.dim)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MLA(config)
        self.ffn = MLP(config)
        self.attn_norm = F.rms_norm(config.dim, eps = 1e-6)
        self.ffn_norm = F.rms_norm(config.dim, eps = 1e-6)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
def precompute_freqs_cis(args: ModelConfig) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

    

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(config))
        self.norm = F.rms_norm(config.dim, eps = 1e-6)
        self.head = nn.Linear(config.dim, config.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        T = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+T]
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits


