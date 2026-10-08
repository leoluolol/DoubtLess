"""GPU test: the Triton instruction attention against the torch implementation.
Run on a GPU:  python tests/gpu_test_instruction_attn.py"""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from reference import instruction_attention
from doubtless.vllm.instruction_attn import instruction_attention_triton

assert torch.cuda.is_available()
torch.manual_seed(0)
dev = torch.device("cuda")
fails = 0
for (N, H, Hkv, D, S, R) in [(1, 16, 8, 128, 64, 4), (7, 16, 8, 128, 64, 8), (33, 2, 1, 128, 64, 40), (128, 32, 8, 128, 64, 128)]:
    q = torch.randn(N, H, D, device=dev, dtype=torch.bfloat16)
    side = torch.randn(R, 2, S, Hkv, D, device=dev, dtype=torch.bfloat16)
    rows = torch.randint(0, R, (N,), device=dev, dtype=torch.long)
    lens = torch.randint(0, S + 1, (N,), device=dev, dtype=torch.int32)
    lens[0] = 0; lens[-1] = S                                    # an empty row and a full row
    out_t, lse_t = instruction_attention_triton(q, side, rows, lens, D ** -0.5)
    out_r, lse_r = instruction_attention(q, side[rows, 0], side[rows, 1], lens, D ** -0.5)
    ok_inf = torch.equal(torch.isinf(lse_t), torch.isinf(lse_r))
    fin = ~torch.isinf(lse_r)
    err_o = (out_t.float() - out_r.float()).abs().max().item()
    err_l = (lse_t[fin] - lse_r[fin]).abs().max().item() if fin.any() else 0.0
    zero_ok = bool((out_t[lens == 0] == 0).all())
    ok = ok_inf and zero_ok and err_o < 2e-2 and err_l < 2e-2
    fails += 0 if ok else 1
    print("%s N=%d H=%d Hkv=%d: max |out| err %.2e, max lse err %.2e, inf pattern %s, empty rows zero %s"
          % ("PASS" if ok else "FAIL", N, H, Hkv, err_o, err_l, ok_inf, zero_ok))
print("\n%d failure(s)" % fails); sys.exit(1 if fails else 0)
