"""The brief-verbose contrast as plain tensor arithmetic (no engine here; both backends call this or reproduce it).

    combine(z_base, z_brief, z_verbose, alpha, p_keep, eos_ids, close_ids)

    keep = top-p set of softmax(z_base) at temperature one          (the large model bounds where the direction acts)
    z    = z_base + alpha * (z_brief - z_verbose)
    z[EOS] = z_base[EOS];  z[</think>] = z_base[</think>]             (the closing decision stays the large model's)
    z    = where(keep, z, -1e9)                                       (the mask is applied LAST)

The sampler (temperature, top-k, top-p, seed) then draws from z as it would from z_base.  Outside the thinking
block the large model decodes alone: rows whose block has closed get z_base back unchanged.
"""
import torch

NEG = -1e9


def topp_mask(z, p):
    """Boolean mask of the top-p set of a 1-D or 2-D logit tensor: the smallest set of tokens whose probability at
    temperature one sums to at least p (ties included)."""
    pr = torch.softmax(z, -1)
    sp, _ = torch.sort(pr, descending=True, dim=-1)
    cum = torch.cumsum(sp, -1)
    thr = torch.where((cum - sp) < p, sp, torch.ones_like(sp)).min(dim=-1, keepdim=True).values
    return pr >= thr


def combine(zb, zbrief, zverbose, alpha, p_keep, eos_ids=(), close_ids=(), open_flag=None):
    """The rule on (rows, vocab) or (vocab,) float logits.

    alpha: a float, or a (rows,) tensor of per-row strengths.
    open_flag: (rows,) bool -- rows outside the thinking block get z_base back unchanged.
    """
    zb, zbrief, zverbose = [z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
                           for z in (zb, zbrief, zverbose)]
    squeeze = zb.dim() == 1
    if squeeze:
        zb, zbrief, zverbose = zb.unsqueeze(0), zbrief.unsqueeze(0), zverbose.unsqueeze(0)
    keep = topp_mask(zb, p_keep)
    delta = zbrief - zverbose
    a = torch.as_tensor(alpha, dtype=zb.dtype, device=zb.device).reshape(-1, 1)
    z = zb + a * delta
    eos_ids, close_ids = list(eos_ids), list(close_ids)
    if eos_ids:
        z[:, eos_ids] = zb[:, eos_ids]
    if close_ids:
        z[:, close_ids[:1]] = zb[:, close_ids[:1]]   # the first token of the closing tag decides
    out = torch.where(keep, z, torch.full_like(zb, NEG))
    out = torch.where(a != 0, out, zb)
    if open_flag is not None:
        out = torch.where(open_flag.reshape(-1, 1), out, zb)
    return out.squeeze(0) if squeeze else out

