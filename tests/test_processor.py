"""CPU tests of processor state and logits using tiny float64 models.

Mock vLLM batch updates and compare batched decoding against per-request
recomputation, including slot moves, preemption and thinking-block exits.

Usage: python tests/test_processor.py
"""
import sys, os, random, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doubtless.contrast import combine
from doubtless.processor.processor import ContrastLogitsProcessor, MoveDirectionality
from doubtless.processor.stream import ModelStream

torch.manual_seed(0); random.seed(0)
V, PAD, EOS, CLOSE = 97, 0, 1, 2
ALPHA, PKEEP, NEG = 3.0, 0.9, -1e9


def tiny(seed):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=V, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=1024, tie_word_embeddings=False,
                         pad_token_id=PAD, eos_token_id=EOS)
    return Qwen3ForCausalLM(config).double().eval()


SMALL = tiny(2)


class Params:
    """Stand-in for vllm.SamplingParams."""
    def __init__(self, brief, verbose):
        self.top_p, self.top_k = 1.0, -1
        self.extra_args = {"contrast": {"brief_ids": brief, "verbose_ids": verbose}}


class Update:
    """Stand-in for vllm.v1.sample.logits_processor.BatchUpdate."""
    def __init__(self, batch_size, removed=(), added=(), moved=()):
        self.batch_size, self.removed, self.added, self.moved = batch_size, list(removed), list(added), list(moved)


def reference(brief, verbose, out, zb):
    """Recompute the small model from scratch for this request and apply the rule.

    Cast logits to float32 to match ModelStream and the processor."""
    with torch.no_grad():
        lb = SMALL(input_ids=torch.tensor([brief + list(out)])).logits[0, -1].float()
        lv = SMALL(input_ids=torch.tensor([verbose + list(out)])).logits[0, -1].float()
    return combine(zb.float(), lb, lv, ALPHA, PKEEP, [EOS], [CLOSE])


def new_proc(ndev=1):
    """Create one stream per simulated device."""
    return ContrastLogitsProcessor(small=SMALL, alpha=ALPHA, p_keep=PKEEP,
                                   close_ids=[CLOSE], eos_ids=[EOS], vocab=V, device="cpu",
                                   devices=["cpu"] * ndev)


def check(name, proc, slots, logits, tol=1e-5, dist=()):
    """slots: index -> (brief, verbose, out, in_think) or None.

    `dist`: slots whose top-p set has one token this step.  There the processor skips the small
    model, so the kept logit lacks alpha*delta; what must be identical is the sampling
    distribution (T=0.6, then top-k/top-p inside vLLM), which is one-hot either way."""
    got = proc.apply(logits.clone())
    worst = 0.0
    for i, s in enumerate(slots):
        if s is None:
            assert torch.equal(got[i], logits[i]), "%s: slot %d must be untouched" % (name, i)
            continue
        brief, verbose, out, in_think = s[:4]
        want = reference(brief, verbose, out, logits[i]) if in_think else logits[i]
        if i in dist and in_think:
            pg, pw = torch.softmax(got[i].float() / 0.6, -1), torch.softmax(want.float() / 0.6, -1)
            d = float((pg - pw).abs().max())
            assert d < 1e-9, "%s: slot %d sampling distribution differs by %.2e" % (name, i, d)
            assert int(pg.argmax()) == int(pw.argmax()) and float(pg.max()) > 1 - 1e-9, "%s: slot %d is not one-hot" % (name, i)
            assert torch.equal(got[i] > NEG / 2, want > NEG / 2), "%s: slot %d masks differ" % (name, i)
            continue
        d = float((got[i].float() - want.float()).abs().max())
        worst = max(worst, d)
        assert d < tol, "%s: slot %d differs by %.2e" % (name, i, d)
        assert int(got[i].argmax()) == int(want.argmax()), "%s: slot %d picks a different token" % (name, i)
    print("  ok  %-34s max |diff| = %.1e" % (name, worst))
    return got


def peaked(n, tok=5):
    """Base logits whose top-p set is the single token `tok` in every row."""
    L = torch.randn(n, V, dtype=torch.float64)
    L[:, tok] += 60.0
    return L


def scenario_lazy(ndev=1):
    """Cards skip the small model on one-token steps and catch up exactly afterwards."""
    proc = new_proc(ndev)
    calls = []
    for st in proc.streams:                     # count small-model forwards and the tokens they feed
        orig = st.step
        def wrapped(toks, _o=orig, _st=st):
            calls.append(max(len(v) if isinstance(v, (list, tuple)) else 1 for v in toks.values()))
            return _o(toks)
        st.step = wrapped
    reqs = {}
    def slots():
        return [reqs.get(i) and tuple(reqs[i]) for i in range(len(proc.reqs))]
    a = [gen(5), gen(7), [], True]
    b = [gen(6), gen(4), [], True]
    reqs[0], reqs[1] = a, b
    proc.update_state(Update(2, added=[(0, Params(a[0], a[1]), None, a[2]),
                                       (1, Params(b[0], b[1]), None, b[2])]))
    check("admit 2", proc, slots(), torch.randn(2, V, dtype=torch.float64))
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    check("decode step (both need it)", proc, slots(), torch.randn(2, V, dtype=torch.float64))
    n0 = len(calls)
    for t in range(3):                            # three one-token steps: no small-model forward
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
        check("one-token step %d (skipped)" % (t + 1), proc, slots(), peaked(2), dist={0, 1})
    assert len(calls) == n0, "the small model must not run on one-token steps (ran %d times)" % (len(calls) - n0)
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    L = torch.randn(2, V, dtype=torch.float64); L[0, 5] += 60.0   # slot 0 one-token, slot 1 needs it
    check("mixed step: catch-up of 4 tokens", proc, slots(), L, dist={0})
    assert len(calls) == n0 + 1 and calls[-1] == 4, "one catch-up forward feeding 4 tokens, got %s" % calls[n0:]
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    check("next step (1 token)", proc, slots(), torch.randn(2, V, dtype=torch.float64))
    assert calls[-1] == 1
    # a request admitted while the card has a backlog: the card catches up first
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    check("one-token step again", proc, slots(), peaked(2), dist={0, 1})
    c = [gen(3), gen(9), [], True]
    reqs[2] = c
    proc.update_state(Update(3, added=[(2, Params(c[0], c[1]), None, c[2])]))
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    n1 = len(calls)
    check("admit a 3rd on a one-token step", proc, slots(), peaked(3), dist={0, 1, 2})
    assert len(calls) == n1 + 1 and calls[-1] == 2, "admission forces a 2-token catch-up first, got %s" % calls[n1:]
    for t in range(2):
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0]); c[2].append(gen(1)[0])
        check("all three decode %d" % (t + 1), proc, slots(), torch.randn(3, V, dtype=torch.float64))
    # leaving the thinking block during a skipped stretch still frees the rows
    a[2].append(gen(1)[0]); b[2].append(CLOSE); c[2].append(gen(1)[0])
    reqs[1][3] = False
    check("slot 1 closes on a one-token step", proc, slots(), peaked(3), dist={0, 2})
    assert proc.reqs[1].rows is None, "rows freed on leaving <think> even while lazy"
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0]); c[2].append(gen(1)[0])
    check("remaining two catch up", proc, slots(), torch.randn(3, V, dtype=torch.float64))
    assert calls[-1] == 2, "catch-up of 2 tokens, got %d" % calls[-1]


def scenario_stall():
    """A resumed request keeps its prefix while its neighbour produces new tokens."""
    proc = new_proc()
    a = [gen(5), gen(7), [], True]
    b = [gen(6), gen(4), [], True]
    slots = [a, b]
    proc.update_state(Update(2, added=[(0, Params(a[0], a[1]), None, a[2]),
                                     (1, Params(b[0], b[1]), None, b[2])]))
    check("stall: admit 2", proc, slots, torch.randn(2, V, dtype=torch.float64))
    for t in range(2):
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
        proc.update_state(None)
        check("stall: decode step %d" % t, proc, slots, torch.randn(2, V, dtype=torch.float64))

    proc.update_state(Update(1, removed=[1]))
    a[2].append(gen(1)[0])
    check("stall: b preempted, a alone", proc, [a], torch.randn(1, V, dtype=torch.float64))
    proc.update_state(Update(2, added=[(1, Params(b[0], b[1]), None, b[2])]))
    a[2].append(gen(1)[0])
    check("stall: b re-added", proc, slots, torch.randn(2, V, dtype=torch.float64))
    for t in range(2):
        a[2].append(gen(1)[0])
        proc.update_state(None)
        check("stall: b waits, a moves (%d)" % t, proc, slots, torch.randn(2, V, dtype=torch.float64))
        assert proc.reqs[1].n_fed == len(b[2])
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    check("stall: b resumes", proc, slots, torch.randn(2, V, dtype=torch.float64))
    b[2].append(gen(1)[0])
    check("stall: a waits, b moves", proc, slots, torch.randn(2, V, dtype=torch.float64))
    for t in range(2):
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
        check("stall: both resume (%d)" % t, proc, slots,
              peaked(2) if t == 0 else torch.randn(2, V, dtype=torch.float64), dist={0, 1} if t == 0 else ())
    st = proc.streams[0]
    pos = dict(zip(st.rows, st.nextpos.view(-1).tolist()))
    for r in proc.reqs:
        assert pos[r.rows[0]] == len(r.brief) + r.n_fed
        assert pos[r.rows[1]] == len(r.verbose) + r.n_fed


def gen(n):
    return [random.randrange(3, V) for _ in range(n)]


def scenario(ndev=1):
    """adds -> steps -> a remove -> a swap -> a unidirectional move -> preemption -> think exit."""
    proc = new_proc(ndev)
    reqs = {}                                     # slot -> [brief, verbose, out, in_think]
    def slots():
        return [reqs.get(i) and tuple(reqs[i]) for i in range(len(proc.reqs))]

    # --- two requests admitted, both with empty output
    a = [gen(5), gen(7), [], True]
    b = [gen(6), gen(4), [], True]
    reqs[0], reqs[1] = a, b
    proc.update_state(Update(2, added=[(0, Params(a[0], a[1]), None, a[2]),
                                       (1, Params(b[0], b[1]), None, b[2])]))
    L = torch.randn(2, V, dtype=torch.float64)
    check("admit 2, no output yet", proc, slots(), L)

    # --- three decode steps: vLLM appends the sampled token to output_token_ids
    for t in range(3):
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
        L = torch.randn(2, V, dtype=torch.float64)
        check("decode step %d" % (t + 1), proc, slots(), L)

    # --- a third request joins mid-flight (its streams must be admitted at the same position)
    c = [gen(3), gen(9), [], True]
    reqs[2] = c
    proc.update_state(Update(3, added=[(2, Params(c[0], c[1]), None, c[2])]))
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    L = torch.randn(3, V, dtype=torch.float64)
    check("admit a 3rd mid-flight", proc, slots(), L)

    # --- request 0 finishes; vLLM removes it and moves request 2 into slot 0
    proc.update_state(Update(2, removed=[0], moved=[(2, 0, MoveDirectionality.UNIDIRECTIONAL)]))
    reqs[0], reqs[2] = reqs[2], None
    del reqs[2]
    b[2].append(gen(1)[0]); c[2].append(gen(1)[0])
    L = torch.randn(2, V, dtype=torch.float64)
    check("remove slot 0 + move 2->0", proc, slots(), L)

    # --- a swap of the two live slots
    proc.update_state(Update(2, moved=[(0, 1, MoveDirectionality.SWAP)]))
    reqs[0], reqs[1] = reqs[1], reqs[0]
    for r in (b, c):
        r[2].append(gen(1)[0])
    L = torch.randn(2, V, dtype=torch.float64)
    check("swap slots 0 and 1", proc, slots(), L)

    # --- preemption: vLLM recomputes a sequence, so its output list gets shorter
    victim = reqs[0]
    del victim[2][-2:]
    L = torch.randn(2, V, dtype=torch.float64)
    check("preemption (output rolled back)", proc, slots(), L)

    # --- the other one closes its thinking block: no steering from the next step on
    other = reqs[1]
    other[2].append(CLOSE)
    reqs[0][2].append(gen(1)[0])
    # The closing tag is already in the output; the next token is outside the thinking block.
    other[3] = False
    L = torch.randn(2, V, dtype=torch.float64)
    check("the step right after </think>: untouched", proc, slots(), L)
    reqs[0][2].append(gen(1)[0])
    L = torch.randn(2, V, dtype=torch.float64)
    check("and every step after that", proc, slots(), L)
    assert proc.reqs[1].rows is None, "small rows must be freed once out of the thinking block"
    print("  ok  small rows freed on leaving <think>")
    if ndev > 1:
        assert len(proc.streams) == ndev
        assert sum(proc.nrows) == sum(2 for r in proc.reqs if r is not None and r.rows is not None), \
            "the per-card row counters must match the rows actually held"
        print("  ok  %d streams, rows accounted for per card: %s" % (ndev, proc.nrows))


def scenario_placement():
    """Distribute requests across devices and compare logits with the reference."""
    proc = new_proc(3)
    reqs = []
    added = []
    for k in range(6):
        r = [gen(4 + k), gen(6), [], True]
        reqs.append(r)
        added.append((k, Params(r[0], r[1]), None, r[2]))
    proc.update_state(Update(6, added=added))
    L = torch.randn(6, V, dtype=torch.float64)
    check("6 requests over 3 streams", proc, [tuple(r) for r in reqs], L)
    devs = [proc.reqs[i].dev for i in range(6)]
    assert sorted(devs) == [0, 0, 1, 1, 2, 2], "least-loaded placement should balance: got %s" % devs
    assert sum(proc.nrows) == 12 and max(proc.nrows) == 4, "row counters per stream: %s" % proc.nrows
    print("  ok  placement balanced across 3 streams: %s, rows %s" % (devs, proc.nrows))
    for t in range(2):
        for r in reqs: r[2].append(gen(1)[0])
        L = torch.randn(6, V, dtype=torch.float64)
        check("3-stream decode step %d" % (t + 1), proc, [tuple(r) for r in reqs], L)
    # remove two and check the counters come back down
    proc.update_state(Update(4, removed=[0, 1], moved=[(4, 0, MoveDirectionality.UNIDIRECTIONAL),
                                                       (5, 1, MoveDirectionality.UNIDIRECTIONAL)]))
    assert sum(proc.nrows) == 8, "after removing 2 of 6 requests: %s" % proc.nrows
    print("  ok  row counters drop on removal: %s" % proc.nrows)


def scenario_rank_split():
    """Compare two ranks' summed contrast with a single processor.

    Both ranks see the same batch updates; each owns a subset of requests.
    Summing their local contrast tensors simulates the all-reduce."""
    single = new_proc(1)
    r0 = ContrastLogitsProcessor(small=SMALL, alpha=ALPHA, p_keep=PKEEP, close_ids=[CLOSE], eos_ids=[EOS],
                                 vocab=V, device="cpu", devices=["cpu"], rank=0, world=2, all_reduce=lambda D: D)
    r1 = ContrastLogitsProcessor(small=SMALL, alpha=ALPHA, p_keep=PKEEP, close_ids=[CLOSE], eos_ids=[EOS],
                                 vocab=V, device="cpu", devices=["cpu"], rank=1, world=2, all_reduce=lambda D: D)
    procs = (single, r0, r1)

    def upd(u):
        for pr in procs:
            pr.update_state(u)

    def step(name, nslots, peak=False):
        L = peaked(nslots) if peak else torch.randn(nslots, V, dtype=torch.float64)
        for pr in procs:                          # vLLM calls this once per step before the forward
            pr.update_state(None)
        want = single.apply(L.clone())
        live0, D0 = r0._local(L.clone())
        live1, D1 = r1._local(L.clone())
        assert [i for i, _ in live0] == [i for i, _ in live1], "%s: the ranks disagree on the live set" % name
        # each request's delta is written by exactly one rank
        both = ((D0 != 0) & (D1 != 0)).any().item()
        assert not both, "%s: a row was filled on both ranks" % name
        D = D0 + D1
        got0 = r0._finish(L.clone(), live0, D)
        got1 = r1._finish(L.clone(), live1, D)
        assert torch.equal(got0, want), "%s: rank 0 differs from the single instance" % name
        assert torch.equal(got1, want), "%s: rank 1 differs from the single instance" % name
        held = sum(r0.nrows) + sum(r1.nrows)
        assert held == sum(single.nrows), "%s: the ranks hold %d rows, the single instance %d" % (name, held, sum(single.nrows))
        print("  ok  %-34s exact on both ranks; rows %d + %d" % (name, sum(r0.nrows), sum(r1.nrows)))

    a = [gen(5), gen(7), [], True]
    b = [gen(6), gen(4), [], True]
    upd(Update(2, added=[(0, Params(a[0], a[1]), None, a[2]),
                         (1, Params(b[0], b[1]), None, b[2])]))
    assert r0.reqs[0].owned and not r1.reqs[0].owned, "request 0 belongs to rank 0"
    assert r1.reqs[1].owned and not r0.reqs[1].owned, "request 1 belongs to rank 1"
    step("admit 2 (one per rank)", 2)
    for t_ in range(3):
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
        step("decode step %d" % (t_ + 1), 2)
    for t_ in range(2):                           # one-token steps: both ranks skip, then catch up
        a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
        step("one-token step %d (both skip)" % (t_ + 1), 2, peak=True)
    c = [gen(3), gen(9), [], True]
    upd(Update(3, added=[(2, Params(c[0], c[1]), None, c[2])]))
    assert r0.reqs[2].owned and not r1.reqs[2].owned, "request 2 goes round-robin to rank 0"
    a[2].append(gen(1)[0]); b[2].append(gen(1)[0])
    step("admit a 3rd mid-flight", 3)
    upd(Update(2, removed=[0], moved=[(2, 0, MoveDirectionality.UNIDIRECTIONAL)]))
    b[2].append(gen(1)[0]); c[2].append(gen(1)[0])
    step("remove slot 0 + move 2->0", 2)
    upd(Update(2, moved=[(0, 1, MoveDirectionality.SWAP)]))
    b[2].append(gen(1)[0]); c[2].append(gen(1)[0])
    step("swap slots 0 and 1", 2)
    del c[2][-2:]                                 # preemption of the request rank 0 owns
    step("preemption (output rolled back)", 2)
    b[2].append(CLOSE); c[2].append(gen(1)[0])    # rank 1's request leaves the thinking block
    step("the step right after </think>", 2)
    c[2].append(gen(1)[0])
    step("and every step after that", 2)
    assert r1.reqs[0].rows is None and single.reqs[0].rows is None, "rows freed on leaving <think> on both"
    # apply() itself: the reduction is called exactly when there is something live
    calls = []
    r0.all_reduce = lambda D: (calls.append(D.shape), D)[1]
    r0.apply(torch.randn(2, V, dtype=torch.float64))
    assert calls == [(2, V)], "apply() must reduce once per step while a request is live"


@torch.no_grad()
def trim_case():
    """Evict a long row and compare the trimmed survivor with a full forward."""
    stream = ModelStream(SMALL, PAD, "cpu")
    long_prompt, short_prompt = torch.randint(3, V, (30,)), torch.randint(3, V, (4,))
    [long_row] = stream.admit([long_prompt])
    for _ in range(5):
        stream.step({long_row: 7})
    [short_row] = stream.admit([short_prompt])
    tokens = [5, 6, 7]
    stream.step({long_row: 8, short_row: tokens[0]})
    stream.evict([long_row])
    assert stream.width() == len(short_prompt) + 1
    for token in tokens[1:]:
        stream.step({short_row: token})
    for layer in stream.past.layers:
        assert layer.keys.shape[2] == stream.width()
    sequence = torch.cat([short_prompt, torch.tensor(tokens)]).view(1, -1)
    reference = SMALL(input_ids=sequence, use_cache=False).logits[0, -1].float()
    torch.testing.assert_close(stream.logits_of(short_row), reference, rtol=0, atol=1e-5)
    print("  ok  trimmed survivor matches a full forward")


@torch.no_grad()
def variable_advance_case():
    torch.manual_seed(11)
    ss = ModelStream(SMALL, PAD, "cpu")
    prompts = [torch.randint(3, V, (n,)) for n in (6, 9, 4)]
    ids = ss.admit(prompts)
    P = dict(zip(ids, prompts)); fed = {r: [] for r in ids}
    t = lambda: int(torch.randint(3, V, (1,)))
    def ref(r):
        seq = torch.cat([P[r], torch.tensor(fed[r], dtype=torch.long)])
        return SMALL(input_ids=seq.view(1, -1), use_cache=False).logits[0, -1].float()
    plan = [{ids[0]: [t()], ids[1]: [], ids[2]: [t()]},          # row 1 stalls (recompute, chunk 1)
            {ids[0]: [t()], ids[1]: [], ids[2]: [t()]},          # still stalled (chunk 2)
            {ids[0]: [t()], ids[1]: [t()], ids[2]: [t()]},       # resumed: everyone advances
            {ids[0]: [t(), t()], ids[1]: [t()], ids[2]: []},     # a two-token catch-up beside a stall
            {ids[0]: [t()], ids[1]: [t()], ids[2]: [t(), t()]},  # the stalled row catches up by two
            {ids[0]: [t()], ids[1]: [t()], ids[2]: [t()]}]
    worst = 0.0
    for step, toks in enumerate(plan):
        before = {r: ss.logits_of(r).clone() for r in ids}
        ss.step(toks)
        for r, tokens in toks.items():
            fed[r] += tokens
            got, want = ss.logits_of(r), ref(r)
            worst = max(worst, float((got - want).abs().max()))
            torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
            if not tokens:
                assert torch.equal(got, before[r]), "a stalled row retains its logits"
        pos = dict(zip(ss.rows, ss.nextpos.view(-1).tolist()))
        for r in ids:
            assert pos[r] == len(P[r]) + len(fed[r])
    width = ss.width()
    ss.step({r: [] for r in ids})
    assert ss.width() == width, "an all-stall step leaves the cache unchanged"
    # Evict a row with padding in the middle, then decode its neighbours.
    ss.evict([ids[1]])
    ids = list(ss.rows)
    for _ in range(2):
        toks = {r: t() for r in ids}
        ss.step(toks)
        for r in ids:
            fed[r].append(toks[r])
            torch.testing.assert_close(ss.logits_of(r), ref(r), rtol=0, atol=1e-5)
    print("  ok  variable advance matches full forwards, max |diff| = %.1e" % worst)


def scenario_reused_slots():
    """A slot can be replaced or overwritten by a move without an explicit removal."""
    proc = new_proc()
    a = [gen(5), gen(7), [], True]
    b = [gen(6), gen(4), [], True]
    proc.update_state(Update(2, added=[(0, Params(a[0], a[1]), None, a[2]),
                                       (1, Params(b[0], b[1]), None, b[2])]))
    check("before slot reuse", proc, [a, b], torch.randn(2, V, dtype=torch.float64))

    c = [gen(4), gen(8), [], True]
    proc.update_state(Update(2, added=[(0, Params(c[0], c[1]), None, c[2])]))
    check("replace occupied slot", proc, [c, b], torch.randn(2, V, dtype=torch.float64))
    assert proc.nrows == [4] and len(proc.streams[0].rows) == 4

    proc.update_state(Update(1, moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
    check("move onto occupied slot", proc, [b], torch.randn(1, V, dtype=torch.float64))
    assert proc.nrows == [2] and len(proc.streams[0].rows) == 2


def scenario_resumed_closed_request():
    """A resumed request keeps its completed thinking state."""
    proc = new_proc()
    out = [10, CLOSE, 11]
    proc.update_state(Update(1, added=[(0, Params(gen(5), gen(7)), None, out)]))
    for _ in range(2):
        logits = torch.randn(1, V, dtype=torch.float64)
        assert torch.equal(proc.apply(logits.clone()), logits)
        out.append(12)
    assert not proc.reqs[0].in_think and proc.nrows == [0]
    print("  ok  resumed closed request leaves base logits untouched")


def scenario_zero_strength():
    """Zero strength leaves every base logit unchanged, including candidates outside top-p."""
    proc = new_proc()
    proc.alpha = 0.0
    out = []
    proc.update_state(Update(1, added=[(0, Params(gen(5), gen(7)), None, out)]))
    for _ in range(3):
        logits = torch.randn(1, V + 3, dtype=torch.float64)
        assert torch.equal(proc.apply(logits.clone()), logits)
        out.append(gen(1)[0])
    print("  ok  zero strength leaves base logits untouched")


if __name__ == "__main__":
    print("processor: full-attention Qwen3 donor")
    scenario_zero_strength()
    scenario_resumed_closed_request()
    print("-- one stream (current single-card behaviour) --")
    scenario(1)
    print("-- two streams (rows spread over cards) --")
    scenario(2)
    print("-- placement over three streams --")
    scenario_placement()
    print("-- rank split: two ranks in lockstep vs one instance --")
    scenario_rank_split()
    print("-- lazy steps: skip one-token steps, catch up exactly --")
    scenario_lazy(1)
    scenario_lazy(2)
    print("-- preemption with a stalled request --")
    scenario_stall()
    print("-- cache trimming and variable advancement --")
    trim_case()
    variable_advance_case()
    scenario_reused_slots()
    print("PASS")
