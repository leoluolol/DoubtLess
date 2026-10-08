"""Per-batch-slot state of the contrast model, on the GPU, maintained by a bookkeeping-only logits processor.

vLLM V1 calls a batch-level LogitsProcessor's update_state(BatchUpdate) before each step's forward with the
requests added / removed / moved in the persistent input batch (the same row order as the logits the sampler
sees).  ThinkerStateProcessor mirrors those events into GPU tensors indexed by batch slot; apply() is the identity.
The model finds the tensors through the process-global ThinkerState (one per worker process, i.e. per TP rank).

What the model needs per slot:
  instr[slot, role, :]   instruction token ids of the brief (0) / verbose (1) thinker, padded with 0
  instr_len[slot, role]  their lengths (43 / 34 for the paper's sentences)
  prompt_len[slot]     length of the request's own prompt (base prompt): tokens at positions >= prompt_len are
                       generated ones, the only ones that may close the thinking block
  kv_row[slot]         the instruction-buffer row of this slot (an indirection that follows batch reordering, so the
                       instruction buffer itself never moves)
  alpha[slot]          contrast strength of the request (0 = plain decoding in the same engine)
  ring[slot, nc], closed[slot]  the last nc generated tokens fed to the model and the 'block closed' flag,
                       updated by the model on the GPU
"""
import torch

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality
except Exception:                                  # offline tests run without vLLM
    LogitsProcessor = object

    class MoveDirectionality:
        UNIDIRECTIONAL = 1
        SWAP = 2

S_MAX = 64            # instruction tokens per thinker; the paper's sentences are 43 and 34 tokens


class ThinkerState:
    _inst = None

    def __init__(self, max_slots, close_ids, device):
        nc = max(1, len(close_ids))
        z = lambda *shape, dtype=torch.long: torch.zeros(*shape, dtype=dtype, device=device)
        self.device = device
        self.max_slots = max_slots
        self.close = torch.tensor(list(close_ids) or [-2], dtype=torch.long, device=device)
        self.close_host = list(close_ids) or [-2]
        self.instr = z(max_slots, 2, S_MAX)
        self.instr_len = z(max_slots, 2, dtype=torch.int32)
        self.prompt_len = z(max_slots, dtype=torch.int32)
        self.kv_row = torch.arange(max_slots, dtype=torch.long, device=device)
        self.alpha = z(max_slots, dtype=torch.float32)
        self.active = z(max_slots, dtype=torch.bool)
        self.ring = torch.full((max_slots, nc), -1, dtype=torch.long, device=device)
        self.closed = z(max_slots, dtype=torch.bool)

    @classmethod
    def create(cls, max_slots, close_ids, device):
        cls._inst = cls(max_slots, close_ids, device)
        return cls._inst

    @classmethod
    def get(cls):
        if cls._inst is None:
            raise RuntimeError("ThinkerState not created: is ThinkerStateProcessor in logits_processors?")
        return cls._inst

    # ---- events (host side, before the forward) -----------------------------------------------------------
    def add(self, slot, brief_ids, verbose_ids, alpha, prompt_len, out_ids=()):
        """out_ids: the request's generated tokens so far (non-empty for a request resumed after preemption; -1
        marks an async-scheduling placeholder) -- the closing ring and flag are rebuilt from them."""
        for role, ids in enumerate((brief_ids, verbose_ids)):
            ids = list(ids)[:S_MAX]
            self.instr[slot, role].zero_()
            if ids:
                self.instr[slot, role, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            self.instr_len[slot, role] = len(ids)
        self.prompt_len[slot] = int(prompt_len)
        self.alpha[slot] = float(alpha)
        self.active[slot] = True
        self.ring[slot] = -1
        self.closed[slot] = False
        # the last entry is not fed yet: it is either the async placeholder or the token the coming forward feeds
        # (and feed() then adds it), so the ring holds everything before it
        out = [int(t) for t in list(out_ids or ())[:-1] if int(t) >= 0]
        nc = self.ring.shape[1]
        if out:
            if nc > 0 and self.close_host[0] >= 0:
                cl = self.close_host
                self.closed[slot] = any(out[t - nc:t] == cl for t in range(nc, len(out) + 1))
            tail = out[-nc:]
            self.ring[slot, nc - len(tail):] = torch.tensor(tail, dtype=torch.long, device=self.device)

    def remove(self, slot):
        self.active[slot] = False
        self.alpha[slot] = 0.0
        self.closed[slot] = False
        self.ring[slot] = -1

    def swap(self, a, b):
        for t in (self.instr, self.instr_len, self.prompt_len, self.kv_row, self.alpha, self.active,
                  self.ring, self.closed):
            ta, tb = t[a].clone(), t[b].clone()
            t[a], t[b] = tb, ta

    # ---- what the model does on the GPU, as plain torch so it can be tested on the CPU --------------------
    def feed(self, rows, last_tokens, positions, valid=None):
        """rows: (n,) slot indices scheduled this step; last_tokens: (n,) the last token fed for each; positions:
        (n,) its absolute position; valid: (n,) bool, False for rows that were not fed this step (padding rows under
        FULL graphs repeat the last real token index).  Generated tokens only (position >= prompt_len) move the
        ring.  Pure tensor ops: this runs inside CUDA-graph capture."""
        gen = positions >= self.prompt_len[rows].to(positions.dtype)
        if valid is not None:
            gen = gen & valid
        tok = torch.where(gen, last_tokens.long(), torch.full_like(last_tokens, -1).long())
        ring = self.ring[rows]
        if ring.shape[1] > 1:
            ring = torch.cat([ring[:, 1:], tok.unsqueeze(1)], dim=1)
        else:
            ring = tok.unsqueeze(1)
        ring = torch.where(gen.unsqueeze(1), ring, self.ring[rows])
        self.ring[rows] = ring
        self.closed[rows] |= (ring == self.close).all(dim=1)

    def open_rows(self, rows):
        return self.active[rows] & ~self.closed[rows]

    # ---- instruction buffer: instruction K/V per batch row, (rows, 2[K,V], S_MAX, kv_heads, head_dim) per role/layer ----
    def alloc_instruction_kv(self, n_layers, kv_heads, head_dim, dtype):
        self.instr_kv = {(role, i): torch.zeros(self.max_slots, 2, S_MAX, kv_heads, head_dim, dtype=dtype, device=self.device)
                     for role in (0, 1) for i in range(n_layers)}


def prompt_end_clamp(num_prompt_tokens, num_computed, num_new):
    """Split a resumed request's recompute before generated tokens need the instruction cache."""
    if num_computed < num_prompt_tokens < num_computed + num_new:
        return num_prompt_tokens - num_computed
    return num_new


def plan_step(query_lens, num_computed, prompt_len, pending):
    """Host-side plan for one step, from the scheduled requests' query lengths and computed-token counts (lists
    over batch rows) and the per-row prompt lengths.  pending = set of rows whose instruction buffer is not filled yet.
    Returns (rows whose instruction must be prefilled BEFORE this step's forward, rows whose chunk crosses the
    prompt end).  A row is due when its prompt is fully computed; a crossing chunk (computed < prompt_len <
    computed + query_len) would feed tokens past the prompt before the instruction exists.  sched.py splits
    such chunks at the prompt end; the builder raises if one still appears."""
    due, crossing = [], []
    for r, (ql, nc) in enumerate(zip(query_lens, num_computed)):
        pl = int(prompt_len[r])
        if ql <= 0:                     # not fed this step (padding rows under FULL graphs): nothing to do yet
            continue
        if r in pending:
            if nc >= pl:
                due.append(r)
            elif nc + ql > pl:
                crossing.append(r)
    return due, crossing


class ThinkerStateProcessor(LogitsProcessor):
    """Bookkeeping only.  additional_config keys: close_ids (list[int]), alpha (float, default per request);
    per-request SamplingParams.extra_args["contrast"] = {"brief_ids": [...], "verbose_ids": [...], "alpha": float?}."""

    def __init__(self, vllm_config=None, device=None, is_pin_memory=False, max_slots=None, close_ids=None,
                 alpha=None):
        cfg = dict(getattr(vllm_config, "additional_config", None) or {}) if vllm_config is not None else {}
        self.alpha = float(alpha if alpha is not None else cfg.get("alpha", 5.0))
        close = list(close_ids if close_ids is not None else cfg.get("close_ids", []))
        if max_slots is None:
            max_slots = int(vllm_config.scheduler_config.max_num_seqs)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.state = ThinkerState.create(max_slots, close, device)
        self.slots = [None] * max_slots
        self.state.pending = set()          # rows awaiting instruction prefill
        self.state.prompt_len_host = [0] * max_slots

    def is_argmax_invariant(self):
        return False

    @classmethod
    def validate_params(cls, sampling_params):
        ex = (getattr(sampling_params, "extra_args", None) or {}).get("contrast")
        if ex is None:
            return
        if not (isinstance(ex.get("brief_ids"), (list, tuple)) and isinstance(ex.get("verbose_ids"), (list, tuple))):
            raise ValueError("contrast needs brief_ids and verbose_ids lists, got %r" % (ex,))
        if len(ex["brief_ids"]) > S_MAX or len(ex["verbose_ids"]) > S_MAX:
            raise ValueError("instructions longer than S_MAX=%d tokens: %d / %d" % (S_MAX, len(ex["brief_ids"]), len(ex["verbose_ids"])))

    def update_state(self, batch_update):
        if batch_update is None:
            return
        st = self.state
        for idx in batch_update.removed:
            self._remove(idx)
        for item in batch_update.added:
            idx, params, prompt_ids = item[0], item[1], item[2]
            if self.slots[idx] is not None:
                self._remove(idx)
            ex = (getattr(params, "extra_args", None) or {}).get("contrast")
            if ex is None:
                st.remove(idx); st.pending.discard(idx)     # a request without contrast settings must not inherit a slot's state
                continue
            plen = len(prompt_ids) if prompt_ids is not None else int(ex.get("prompt_len", 0))
            out_ids = item[3] if len(item) > 3 else ()
            st.add(idx, ex.get("brief_ids", []), ex.get("verbose_ids", []), ex.get("alpha", self.alpha), plen, out_ids)
            self.slots[idx] = True
            st.pending.add(idx); st.prompt_len_host[idx] = plen
        for mv in batch_update.moved:
            i1, i2, d = mv[0], mv[1], mv[2]
            st.swap(i1, i2)                     # a one-way move lands on an empty slot: swapping is the same
            self.slots[i1], self.slots[i2] = self.slots[i2], self.slots[i1]
            p1, p2 = i1 in st.pending, i2 in st.pending
            st.pending.discard(i1); st.pending.discard(i2)
            if p1: st.pending.add(i2)
            if p2: st.pending.add(i1)
            st.prompt_len_host[i1], st.prompt_len_host[i2] = st.prompt_len_host[i2], st.prompt_len_host[i1]
            if d != MoveDirectionality.SWAP:
                # condense() moves a request into a finished request's hole without listing that request in
                # `removed`; the source slot must not keep the finished request's state
                st.remove(i1); st.pending.discard(i1); st.prompt_len_host[i1] = 0
                self.slots[i1] = None

    def _remove(self, idx):
        if idx < len(self.slots) and self.slots[idx] is not None:
            self.state.remove(idx)
            self.slots[idx] = None
            self.state.pending.discard(idx)

    def apply(self, logits):
        return logits
