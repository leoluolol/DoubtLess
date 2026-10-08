"""CPU tests of the rule in doubtless/contrast.py.  Run:  python tests/test_contrast.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doubtless.contrast import NEG, combine, topp_mask

torch.manual_seed(0)
fails = []


def check(name, ok):
    print("PASS" if ok else "FAIL", name)
    if not ok:
        fails.append(name)


# top-p set: smallest set whose mass reaches p, ties included; p=1 keeps everything
z = torch.randn(4, 50); z[:, 3] += 8
keep = topp_mask(z, 0.95)
pr = torch.softmax(z, -1)
check("topp_mask: mass of the set >= p", bool((pr.masked_fill(~keep, 0).sum(-1) >= 0.95 - 1e-6).all()))
check("topp_mask: dropping the least likely kept token would fall below p",
      all(pr[r][keep[r]].sum() - pr[r][keep[r]].min() < 0.95 for r in range(4)))
check("topp_mask: p=1 keeps everything", bool(topp_mask(z, 1.0).all()))

# the rule
zb, zc, zv = torch.randn(3, 40), torch.randn(3, 40), torch.randn(3, 40)
zb[:, 5] += 6
out = combine(zb, zc, zv, 2.0, 0.9, eos_ids=[1], close_ids=[2, 3])
keep = topp_mask(zb, 0.9)
ref = zb + 2.0 * (zc - zv); ref[:, 1] = zb[:, 1]; ref[:, 2] = zb[:, 2]
check("combine: base + alpha * direction inside the set", torch.allclose(out[keep], ref[keep]))
check("combine: -1e9 outside the set", bool((out[~keep] == NEG).all()))
check("combine: EOS and the first token of the closing tag keep the base logit (when kept)",
      all(torch.equal(out[r, [1, 2]][keep[r, [1, 2]]], zb[r, [1, 2]][keep[r, [1, 2]]]) for r in range(3)))
check("combine: 1-D input works and equals the row of the 2-D call", torch.equal(combine(zb[0], zc[0], zv[0], 2.0, 0.9, [1], [2, 3]), out[0]))
per_row = combine(zb, zc, zv, torch.tensor([0.0, 2.0, 1.0]), 0.9, [1], [2, 3])
check("combine: per-row strengths (alpha 0 is the base)", torch.equal(per_row[0], zb[0]) and torch.equal(per_row[1], out[1]))
half = combine(zb.half(), zc.half(), zv.half(), 2.0, 0.9, [1], [2, 3])
full = combine(zb.half().float(), zc.half().float(), zv.half().float(), 2.0, 0.9, [1], [2, 3])
check("combine: half-precision logits use float32 arithmetic", half.dtype == torch.float32 and torch.equal(half, full))
op = combine(zb, zc, zv, 2.0, 0.9, [1], [2, 3], open_flag=torch.tensor([True, False, True]))
check("combine: a closed row gets the base logits back unchanged", torch.equal(op[1], zb[1]) and torch.equal(op[0], out[0]))

# the fast backend's compute_logits calls combine() with alpha already zeroed on closed rows; this is the rule it
# wrote out inline before, and the two must agree bit for bit on float32 logits with per-row strengths
def inline_rule(zb, zbr, zv, alpha, p_keep, eos_ids, close_ids):
    keep = topp_mask(zb, p_keep)
    z = zb + alpha.unsqueeze(1) * (zbr - zv)
    z[:, eos_ids] = zb[:, eos_ids]
    z[:, close_ids[:1]] = zb[:, close_ids[:1]]
    z = torch.where(keep, z, torch.full_like(zb, NEG))
    return torch.where((alpha != 0).unsqueeze(1), z, zb)

g = torch.Generator().manual_seed(1)
zb, zbr, zv = (torch.randn(6, 300, generator=g) * 3 for _ in range(3))
zb[:, 7] += 9
alpha = torch.tensor([5.0, 0.0, 5.0, 2.5, 0.0, 5.0]) * torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])   # row 2: closed
check("combine: equals the fast backend's former inline rule bit for bit",
      torch.equal(combine(zb, zbr, zv, alpha, 0.95, [11, 12], [13, 14]), inline_rule(zb.clone(), zbr, zv, alpha, 0.95, [11, 12], [13, 14])))

# prefix-cache salt of a contrast request (doubtless/vllm/engine.py)
from doubtless.vllm.engine import cache_salt
p1, br, vb = [1, 2, 3, 4], [9, 9], [8, 8, 8]
p2 = p1 + [5, 6, 7, 8]                      # the next turn: the previous prompt, header and reply, then a new message
check("cache_salt: the same prompt and instructions give the same salt (repeated runs share the prompt)",
      cache_salt(p1, br, vb) == cache_salt(list(p1), list(br), list(vb)))
check("cache_salt: two turns of one conversation never share a salt", cache_salt(p1, br, vb) != cache_salt(p2, br, vb))
check("cache_salt: different instructions never share a salt",
      cache_salt(p1, br, vb) != cache_salt(p1, vb, br) and cache_salt(p1, br, vb) != cache_salt(p1, br + [9], vb))
check("cache_salt: part boundaries matter", cache_salt([1, 2], [3], [4]) != cache_salt([1], [2, 3], [4]))

print("\n%d failure(s)" % len(fails)); sys.exit(1 if fails else 0)
