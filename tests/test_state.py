"""Compare request state with the reference closing state under adds, removals and moves.

Run: python tests/test_state.py
"""
import os, sys, random
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from doubtless.vllm.state import ThinkerState, ThinkerStateProcessor, MoveDirectionality
from reference import ClosedState


class P:            # stand-in for SamplingParams
    def __init__(self, extra): self.extra_args = extra


class BU:           # stand-in for BatchUpdate
    def __init__(self, added=(), removed=(), moved=()): self.added, self.removed, self.moved = list(added), list(removed), list(moved)


random.seed(1); torch.manual_seed(1)
CLOSE = [7, 8]
proc = ThinkerStateProcessor(max_slots=6, close_ids=CLOSE, device=torch.device("cpu"))
st = proc.state
ref = ClosedState(6, CLOSE)
fails = 0


def check(name, ok):
    global fails
    print("PASS" if ok else "FAIL", name)
    fails += 0 if ok else 1


# add three requests with different instruction lengths and prompt lengths
adds = []
for slot, (nb, nv, pl) in enumerate([(43, 34, 52), (5, 1, 10), (64, 64, 3)]):
    adds.append((slot, P({"contrast": {"brief_ids": list(range(100, 100 + nb)), "verbose_ids": list(range(200, 200 + nv))}}),
                 list(range(pl)), []))
    ref.reset(slot)
proc.update_state(BU(added=adds))
check("add: lengths", st.instr_len.tolist()[:3] == [[43, 34], [5, 1], [64, 64]] and st.prompt_len.tolist()[:3] == [52, 10, 3])
check("add: ids and padding", st.instr[0, 0, 42].item() == 142 and st.instr[0, 0, 43].item() == 0 and st.instr[1, 1, 0].item() == 200)
check("add: flags", st.pending == {0, 1, 2} and st.active.tolist()[:3] == [True] * 3 and st.alpha[0].item() == 5.0)

# feed generated tokens: positions before prompt_len must not count
rows = torch.tensor([0, 1, 2])
st.feed(rows, torch.tensor([7, 7, 7]), torch.tensor([51, 10, 3]))      # slot 0 at a prompt position: ignored
ref.feed(torch.tensor([-1, 7, 7, -1, -1, -1]))
st.feed(rows, torch.tensor([8, 8, 1]), torch.tensor([52, 11, 4]))
ref.feed(torch.tensor([-1, 8, 1, -1, -1, -1]))
check("feed: closes only slot 1", st.closed.tolist() == ref.closed.tolist() == [False, True, False, False, False, False])
check("open_rows", st.open_rows(rows).tolist() == [True, False, True])

# remove slot 1, move slot 2 -> 1 (one-way), then swap 0 and 1
proc.update_state(BU(removed=[1]))
check("remove", not st.active[1] and not st.closed[1])
proc.update_state(BU(moved=[(2, 1, MoveDirectionality.UNIDIRECTIONAL)]))
ref.move(2, 1)
check("move: state follows", st.instr_len[1].tolist() == [64, 64] and st.prompt_len[1].item() == 3 and not st.active[2]
      and st.kv_row.tolist()[:3] == [0, 2, 1] and proc.slots[2] is None)
proc.update_state(BU(moved=[(0, 1, MoveDirectionality.SWAP)]))
check("swap", st.instr_len[0].tolist() == [64, 64] and st.instr_len[1].tolist() == [43, 34] and st.kv_row.tolist()[:3] == [2, 0, 1])

# random churn against the reference ring
proc2 = ThinkerStateProcessor(max_slots=8, close_ids=[9], device=torch.device("cpu")); s2 = proc2.state; r2 = ClosedState(8, [9])
live = set()
for step in range(300):
    if random.random() < 0.3:
        free = [i for i in range(8) if i not in live]
        if free:
            i = random.choice(free); live.add(i); r2.reset(i)
            proc2.update_state(BU(added=[(i, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(4)), [])]))
    if live and random.random() < 0.15:
        i = random.choice(sorted(live)); live.remove(i); r2.reset(i)
        proc2.update_state(BU(removed=[i]))
    if len(live) >= 2 and random.random() < 0.2:
        a, b = random.sample(sorted(live), 2)
        proc2.update_state(BU(moved=[(a, b, MoveDirectionality.SWAP)]))
        r2.ring[[a, b]] = r2.ring[[b, a]]; r2.closed[[a, b]] = r2.closed[[b, a]]
    if live:
        rows = torch.tensor(sorted(live)); toks = torch.randint(0, 12, (len(rows),))
        s2.feed(rows, toks, torch.full((len(rows),), 10))
        full = torch.full((8,), -1); full[rows] = toks; r2.feed(full)
        if s2.closed.tolist() != r2.closed.tolist():
            check("churn step %d" % step, False); break
else:
    check("churn 300 steps: closed flags match the reference", True)

# plan_step + the host mirror of pending rows through add / move / remove
from doubtless.vllm.state import plan_step
proc3 = ThinkerStateProcessor(max_slots=4, close_ids=[9], device=torch.device("cpu")); s3 = proc3.state
proc3.update_state(BU(added=[(0, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(47)), []),
                          (1, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(30)), [])]))
check("pending after add", s3.pending == {0, 1} and s3.prompt_len_host[:2] == [47, 30])
# step 1: both prompts prefilled in full this step -> nothing due yet, nothing crossing
due, cross = plan_step([47, 30], [0, 0], s3.prompt_len_host, s3.pending)
check("plan: prompt prefill step -> not due", due == [] and cross == [])
# step 2: computed == prompt -> due
due, cross = plan_step([1, 1], [47, 30], s3.prompt_len_host, s3.pending)
check("plan: next step -> both due", due == [0, 1] and cross == [])
# a resumed request whose recompute chunk spans the prompt end -> crossing
due, cross = plan_step([60, 1], [0, 30], s3.prompt_len_host, {0})
check("plan: crossing chunk flagged", due == [] and cross == [0])
# a chunk that ends exactly at the prompt end is neither
due, cross = plan_step([47, 1], [0, 30], s3.prompt_len_host, {0})
check("plan: chunk ending at the prompt end -> wait", due == [] and cross == [])
proc3.update_state(BU(moved=[(1, 3, MoveDirectionality.UNIDIRECTIONAL)]))
check("pending follows a move", s3.pending == {0, 3} and s3.prompt_len_host[3] == 30)
proc3.update_state(BU(removed=[0]))
check("pending drops a removed row", s3.pending == {3})

# a resumed request: the ring and the closed flag come back from its output tokens
proc4 = ThinkerStateProcessor(max_slots=3, close_ids=[7, 8], device=torch.device("cpu")); s4 = proc4.state
proc4.update_state(BU(added=[(0, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(10)), [3, 4, 7, 8, 5, -1]),
                          (1, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(10)), [3, 7]),
                          (2, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(10)), [])]))
check("resume: closed rebuilt from the fed outputs", s4.closed.tolist() == [True, False, False])
check("resume: ring holds the fed tokens (last entry not fed yet)", s4.ring.tolist() == [[8, 5], [-1, 3], [-1, -1]])
s4.feed(torch.tensor([1, 2]), torch.tensor([7, 9]), torch.tensor([11, 11]))          # the coming forward feeds it
s4.feed(torch.tensor([1, 2]), torch.tensor([8, 9]), torch.tensor([12, 12]))
check("resume: row 1 closes once both tokens are fed", s4.closed.tolist() == [True, True, False])
# a 3-token close whose last id is the unfed one must not close at add time
s6 = ThinkerStateProcessor(max_slots=2, close_ids=[7, 8, 9], device=torch.device("cpu")).state
s6.add(0, [1], [2], 5.0, 4, out_ids=[3, 7, 8, 9]); s6.add(1, [1], [2], 5.0, 4, out_ids=[3, 7, 8, 9, 2])
check("resume: 3-token close, last id unfed", s6.closed.tolist() == [False, True] and s6.ring[0].tolist() == [3, 7, 8])
s6.feed(torch.tensor([0]), torch.tensor([9]), torch.tensor([7]))
check("resume: 3-token close completes when fed", bool(s6.closed[0]))
# a vacated slot keeps no state, and a request without contrast settings resets its slot
proc7 = ThinkerStateProcessor(max_slots=3, close_ids=[9], device=torch.device("cpu")); s7 = proc7.state
proc7.update_state(BU(added=[(0, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(10)), []),
                          (2, P({"contrast": {"brief_ids": [1], "verbose_ids": [2]}}), list(range(12)), [])]))
proc7.update_state(BU(moved=[(2, 1, MoveDirectionality.UNIDIRECTIONAL)]))
check("move: source slot cleared", not s7.active[2] and s7.alpha[2] == 0 and s7.prompt_len_host[2] == 0 and s7.active[1] and s7.prompt_len[1] == 12)
proc7.update_state(BU(added=[(2, P(None), list(range(5)), [])]))
proc7.update_state(BU(added=[(1, P(None), list(range(5)), [])]))
check("add without contrast settings resets the slot", not s7.active[1] and s7.alpha[1] == 0 and 1 not in s7.pending)
# padding rows: feed() with a validity mask leaves them alone even when handed a real token
s5 = ThinkerStateProcessor(max_slots=3, close_ids=[9], device=torch.device("cpu")).state
s5.add(0, [1], [2], 5.0, 4); s5.add(1, [1], [2], 5.0, 4)
s5.feed(torch.tensor([0, 1, 2]), torch.tensor([9, 9, 9]), torch.tensor([10, 10, 10]), torch.tensor([True, False, False]))
check("feed: validity mask", s5.closed.tolist() == [True, False, False] and s5.ring[1].tolist() == [-1])
# padding rows (zero-length queries) are ignored by the planner
due, cross = plan_step([1, 0, 0], [47, 0, 0], [47, 0, 0], {0, 1, 2})
check("plan: zero-length rows skipped", due == [0] and cross == [])

# A preempted request replays its prompt before the instruction cache is restored.
from doubtless.vllm.state import prompt_end_clamp
check("clamp: fresh prompt unchanged", prompt_end_clamp(47, 0, 47) == 47 and prompt_end_clamp(47, 30, 17) == 17)
check("clamp: resumed chunk ends at prompt", prompt_end_clamp(47, 0, 147) == 47 and prompt_end_clamp(47, 32, 115) == 15)
check("clamp: generated chunks unchanged", prompt_end_clamp(47, 47, 60) == 60 and prompt_end_clamp(47, 64, 83) == 83)
pending, computed, chunks, plans = {0}, 0, [], []
while computed < 147:
    n = prompt_end_clamp(47, computed, min(60, 147 - computed))
    due, cross = plan_step([n], [computed], [47], pending)
    chunks.append((computed, n)); plans.append((due, cross))
    pending.difference_update(due)
    computed += n
check("resume replay: chunks", chunks == [(0, 47), (47, 60), (107, 40)])
check("resume replay: instruction restored once, never crossing", plans == [([], []), ([0], []), ([], [])])
check("resume replay: prefix cache beyond prompt", plan_step([prompt_end_clamp(47, 64, 83)], [64], [47], {0}) == ([0], []))

# Exercise the scheduler hook without requiring vLLM or a GPU in the CPU suite.
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from doubtless.vllm import sched
class Scheduler:
    def _reserve_prefill_lookahead(self, request, computed, n):
        return max(0, n - self.lookahead)
class AsyncScheduler(Scheduler):
    pass
module = ModuleType("vllm.v1.core.sched.scheduler")
module.Scheduler = Scheduler
with patch.dict(sys.modules, {module.__name__: module}):
    sched.install()
    reserved = Scheduler._reserve_prefill_lookahead
    sched.install()
    request = SimpleNamespace(num_prompt_tokens=47, sampling_params=SimpleNamespace(extra_args={"contrast": {}}))
    scheduler = Scheduler(); scheduler.lookahead = 0
    check("scheduler: installs once and async inherits", Scheduler._reserve_prefill_lookahead is reserved
          and AsyncScheduler._reserve_prefill_lookahead is reserved)
    check("scheduler: resumed contrast split", reserved(scheduler, request, 0, 147) == 47)
    check("scheduler: replay after prompt unchanged", reserved(scheduler, request, 47, 60) == 60)
    request.sampling_params.extra_args = None
    check("scheduler: plain request unchanged", reserved(scheduler, request, 0, 147) == 147)
    request.sampling_params.extra_args = {"contrast": {}}
    scheduler.lookahead = 3
    check("scheduler: original lookahead preserved", reserved(scheduler, request, 0, 30) == 27
          and reserved(scheduler, request, 0, 147) == 47)
print("\n%d failure(s)" % fails); sys.exit(1 if fails else 0)
