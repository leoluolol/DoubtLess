"""Grouped-query attention without materializing repeated KV heads.

For single-query decoding, group the query heads around their shared KV heads and
apply attention directly. Other shapes use the standard attention implementation.
"""
import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

NAME = "gqa_decode"

def gqa_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    B, Hq, q_len, D = query.shape
    Hkv = key.shape[1]
    if q_len != 1 or Hq == Hkv or Hq % Hkv != 0 or dropout or query.device.type != key.device.type:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    g = Hq // Hkv
    scale = scaling if scaling is not None else D ** -0.5
    qg = query.reshape(B, Hkv, g, D)                                   # heads h*g..h*g+g-1 share kv head h (repeat_kv order)
    scores = torch.matmul(qg, key.transpose(-1, -2))                    # [B, Hkv, g, L]
    scores = scores.to(torch.promote_types(scores.dtype, torch.float32)) * scale   # bf16/fp16 -> fp32 softmax, fp64 stays fp64
    if attention_mask is not None:
        m = attention_mask[:, :, -1:, :key.shape[2]] if attention_mask.dim() == 4 else attention_mask   # [B, 1, 1, L]
        if m.dtype == torch.bool: scores = scores.masked_fill(~m, float("-inf"))
        else: scores = scores + m.to(scores.dtype)
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    out = torch.matmul(probs, value)                                    # [B, Hkv, g, D]
    return out.reshape(B, Hq, 1, D).transpose(1, 2).contiguous(), None  # [B, q=1, Hq, D]

def register():
    if NAME not in AttentionInterface._global_mapping:
        AttentionInterface.register(NAME, gqa_decode_attention)
        AttentionMaskInterface.register(NAME, sdpa_mask)   # same boolean mask as sdpa (True = attend)
    return NAME
