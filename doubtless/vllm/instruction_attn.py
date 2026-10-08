"""Instruction-buffer attention in Triton.

Each token attends to the instruction keys of its own request: q (N, H, D) against side[row, 0/1, :, Hkv, D] with
`lens` valid keys.  One program per (token, kv head) handles that head's g query heads (padded to 16 rows for the
tensor-core dot), so the whole instruction attention of one layer is ONE kernel launch instead of ~20 small torch ops.
Returns out (N, H, D) in q's dtype and lse (H, N) float32, -inf where lens == 0 (merge_attn_states treats that as an empty partial).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _instruction_attn_kernel(q_ptr, side_ptr, row_ptr, len_ptr, out_ptr, lse_ptr, scale,
                      N, H, HKV,
                      stride_qn, stride_qh, stride_qd,
                      stride_sr, stride_skv, stride_ss, stride_sh, stride_sd,
                      stride_on, stride_oh, stride_od,
                      G: tl.constexpr, S: tl.constexpr, D: tl.constexpr, GP: tl.constexpr):
    n = tl.program_id(0)
    kvh = tl.program_id(1)
    row = tl.load(row_ptr + n)
    ln = tl.load(len_ptr + n)
    gi = tl.arange(0, GP)                       # padded query-head rows
    si = tl.arange(0, S)
    di = tl.arange(0, D)
    hmask = gi < G
    h = kvh * G + gi
    q = tl.load(q_ptr + n * stride_qn + h[:, None] * stride_qh + di[None, :] * stride_qd,
                mask=hmask[:, None], other=0.0)                                  # (GP, D)
    kbase = side_ptr + row * stride_sr + kvh * stride_sh
    k = tl.load(kbase + si[:, None] * stride_ss + di[None, :] * stride_sd)        # (S, D)  K half
    v = tl.load(kbase + stride_skv + si[:, None] * stride_ss + di[None, :] * stride_sd)   # (S, D)  V half
    s = tl.dot(q.to(tl.bfloat16), tl.trans(k).to(tl.bfloat16)) * scale             # (GP, S) fp32
    valid = si[None, :] < ln
    s = tl.where(valid, s, float("-inf"))
    m = tl.max(s, axis=1)                                                          # (GP,)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    p = tl.exp(s - m_safe[:, None])
    p = tl.where(valid, p, 0.0)
    l = tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))                            # (GP, D)
    l_safe = tl.where(l == 0.0, 1.0, l)
    o = acc / l_safe[:, None]
    lse = tl.where(l == 0.0, float("-inf"), m_safe + tl.log(l_safe))
    tl.store(out_ptr + n * stride_on + h[:, None] * stride_oh + di[None, :] * stride_od,
             o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])
    tl.store(lse_ptr + h * N + n, lse, mask=hmask)


def instruction_attention_triton(q, side, rows, lens, scale):
    """q (N, H, D); side (R, 2, S, Hkv, D); rows (N,) int64; lens (N,) int32.  -> out (N, H, D) q.dtype, lse (H, N) fp32."""
    N, H, D = q.shape
    S, Hkv = side.shape[2], side.shape[3]
    G = H // Hkv
    GP = max(16, triton.next_power_of_2(G))
    out = torch.empty_like(q)
    lse = torch.empty((H, N), dtype=torch.float32, device=q.device)
    if N == 0:
        return out, lse
    _instruction_attn_kernel[(N, Hkv)](
        q, side, rows, lens, out, lse, float(scale), N, H, Hkv,
        q.stride(0), q.stride(1), q.stride(2),
        side.stride(0), side.stride(1), side.stride(2), side.stride(3), side.stride(4),
        out.stride(0), out.stride(1), out.stride(2),
        G=G, S=S, D=D, GP=GP, num_warps=4)
    return out, lse
