"""Reference calculations for state and Triton attention tests."""
import torch


class ClosedState:
    """CPU reference for the closing-tag state used by test_state.py."""

    def __init__(self, max_slots, close_ids):
        self.close = torch.tensor(list(close_ids), dtype=torch.long)
        self.nc = len(close_ids)
        self.ring = torch.full((max_slots, self.nc), -1, dtype=torch.long)
        self.closed = torch.zeros(max_slots, dtype=torch.bool)

    def reset(self, slot):
        self.ring[slot] = -1
        self.closed[slot] = False

    def move(self, src, dst):
        self.ring[dst] = self.ring[src]
        self.closed[dst] = self.closed[src]
        self.reset(src)

    def feed(self, tokens):
        tokens = tokens.long()
        has = tokens >= 0
        if self.nc > 1:
            self.ring[has, :-1] = self.ring[has, 1:]
        self.ring[has, -1] = tokens[has]
        self.closed |= (self.ring == self.close).all(dim=1)


def instruction_attention(q, k_side, v_side, lens, scale):
    """Dense attention of each query token over ITS OWN instruction-buffer keys (the instruction tokens of its request).
    q (N, H, D); k_side / v_side (N, S_MAX, Hkv, D) gathered per token; lens (N,) number of valid keys (0 = none).
    Query heads are grouped over kv heads (H = g * Hkv, head h uses kv head h // g) without materialising repeated
    keys.  Returns (out (N, H, D), lse (H, N)) with lse = -inf and out = 0 where lens == 0, which merge_attn_states
    treats as an empty partial."""
    N, H, D = q.shape
    S, Hkv = k_side.shape[1], k_side.shape[2]
    g = H // Hkv
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    qg = q.to(dt).reshape(N, Hkv, g, D)
    s = torch.einsum("nkgd,nskd->nkgs", qg, k_side.to(dt)) * scale                    # (N, Hkv, g, S)
    valid = torch.arange(S, device=q.device).unsqueeze(0) < lens.unsqueeze(1)          # (N, S)
    s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)                                                  # (N, Hkv, g); -inf if no key
    p = torch.nan_to_num(torch.exp(s - lse.unsqueeze(-1)), nan=0.0)
    out = torch.einsum("nkgs,nskd->nkgd", p, v_side.to(dt)).reshape(N, H, D)
    return out, lse.reshape(N, H).transpose(0, 1)
