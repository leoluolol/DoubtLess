"""Apply brief-minus-verbose logits inside each request's thinking block.

Each request keeps two small-model rows. Existing rows advance before new rows
are admitted; slot moves preserve ownership and preemption rebuilds the prefix.
One-token base nuclei defer the small forward until a contrast delta is needed.
"""
import torch

from doubtless.processor.stream import ModelStream
from doubtless.contrast import topp_mask

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality
except ImportError:  # The processor's CPU tests run without vLLM.
    LogitsProcessor = object

    class MoveDirectionality:
        UNIDIRECTIONAL = 1
        SWAP = 2

NEG = -1e9


class _Req:
    """One request in the vLLM batch, with two rows on its assigned small stream."""

    def __init__(self, out, brief, verbose):
        self.out = out  # vLLM updates this list in place.
        self.brief, self.verbose = brief, verbose
        self.dev, self.rows = 0, None
        self.n_seen = 0  # Tokens inspected for the thinking-block boundary.
        self.n_fed = 0   # Tokens fed to the small model; may lag during lazy steps.
        self.in_think = self.pending = self.owned = True


class ContrastLogitsProcessor(LogitsProcessor):
    """Add alpha * (brief - verbose) to base logits during thinking."""

    def __init__(self, vllm_config=None, device=None, is_pin_memory=False,
                 small=None, tokenizer=None, alpha=None, p_keep=None, close_ids=None,
                 eos_ids=None, vocab=None, devices=None, rank=None, world=None, all_reduce=None):
        cfg = dict(getattr(vllm_config, "additional_config", None) or {})
        self.alpha = float(alpha if alpha is not None else cfg.get("alpha", 5.0))
        self.p_keep = float(p_keep if p_keep is not None else cfg.get("p_keep", 0.95))
        self.dev = torch.device(device) if device is not None else torch.device("cpu")
        devs = devices or cfg.get("small_devices")
        if devs == "local":
            devs = ["cuda:%d" % torch.cuda.current_device()] if torch.cuda.is_available() else None
        self.devs = [torch.device(d) for d in devs] if devs else [self.dev]

        if small is None:
            import copy
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from doubtless.processor.gqa_attention import register as register_gqa
            path = cfg["small"]
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation=register_gqa()).eval()
            replicas = [model.to(self.devs[0])]
            replicas.extend(copy.deepcopy(model).to(d) for d in self.devs[1:])
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(path)
        else:
            replicas = [small] * len(self.devs)
        pad = getattr(tokenizer, "pad_token_id", None) or 0
        self.close = list(close_ids if close_ids is not None else cfg.get("close_ids", []))
        self.eos = list(eos_ids if eos_ids is not None else cfg.get("eos_ids", []))
        self.vocab = int(vocab if vocab is not None else cfg.get("vocab") or replicas[0].config.vocab_size)
        self.streams = [ModelStream(m, pad, str(d)) for m, d in zip(replicas, self.devs)]
        self.nrows = [0] * len(self.streams)
        self.reqs = []

        # Each TP rank computes a disjoint share; all-reduce distributes the deltas.
        if world is None:
            rank, world = 0, 1
            if small is None and cfg.get("rank_split", True):
                from vllm.distributed import (get_tensor_model_parallel_rank,
                                              get_tensor_model_parallel_world_size)
                rank, world = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        self.rank, self.world = int(rank), int(world)
        if all_reduce is None and self.world > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce
            all_reduce = tensor_model_parallel_all_reduce
        self.all_reduce = all_reduce
        self._n_added = 0

    def is_argmax_invariant(self):
        return False                     # the contrast does change which token wins

    @classmethod
    def validate_params(cls, sampling_params):
        ex = (getattr(sampling_params, "extra_args", None) or {}).get("contrast")
        if ex is None:
            return
        if not ex.get("brief_ids") or not ex.get("verbose_ids"):
            raise ValueError("contrast needs both brief_ids and verbose_ids")
        # The base top-p mask constrains the contrast. vLLM still applies the caller's
        # temperature, top-k and top-p to the combined distribution.

    def update_state(self, batch_update):
        if batch_update is None:         # vLLM: no change to the batch this step (the common case)
            return
        # the documented order is removes, then adds, then moves
        for idx in batch_update.removed:
            self._free(idx)
        for item in batch_update.added:
            idx, params, _prompt_ids, out_ids = item[0], item[1], item[2], item[3]
            self._grow(idx)
            if self.reqs[idx] is not None:
                # A reused slot may omit the old request from `removed`.
                # Retire its small-model rows before assigning the new request.
                self._free(idx)
            ex = (getattr(params, "extra_args", None) or {}).get("contrast")
            if ex is None:               # a request that opted out: steered by nobody
                self.reqs[idx] = None
                continue
            self.reqs[idx] = _Req(out_ids, list(ex["brief_ids"]), list(ex["verbose_ids"]))
            # Restore the thinking state when a request rejoins with generated tokens.
            n = len(self.close)
            self.reqs[idx].in_think = not n or not any(
                list(out_ids[t - n:t]) == self.close for t in range(n, len(out_ids) + 1))
            self.reqs[idx].owned = (self._n_added % self.world) == self.rank
            self._n_added += 1
        for mv in batch_update.moved:
            i1, i2, direction = mv[0], mv[1], mv[2]
            self._grow(max(i1, i2))
            if direction == MoveDirectionality.SWAP:
                self.reqs[i1], self.reqs[i2] = self.reqs[i2], self.reqs[i1]
            else:
                self._free(i2)
                self.reqs[i2], self.reqs[i1] = self.reqs[i1], None
        n = batch_update.batch_size
        for idx in range(n, len(self.reqs)):
            self._free(idx)
        del self.reqs[n:]

    @torch.no_grad()
    def apply(self, logits):
        live, delta = self._local(logits)
        if self.world > 1 and live:
            delta = self.all_reduce(delta)
        return self._finish(logits, live, delta)

    def _local(self, logits):
        """Advance owned rows and return their deltas; save the base mask for _finish()."""
        live = [(i, r) for i, r in enumerate(self.reqs) if r is not None and r.in_think]
        for i, r in live:                # preemption: vLLM recomputed the sequence, KV is stale
            if len(r.out) < r.n_seen:
                self._drop_rows(r)
                r.pending, r.n_seen, r.n_fed, r.rows = True, 0, 0, None

        # 0. the top-p set of every live request, once (identical on every rank: the logits are
        #    all-gathered, so every rank makes the same skip decisions)
        keeps, need = {}, {}
        if live:
            K = topp_mask(logits[[i for i, _ in live], :self.vocab].float(), self.p_keep)
            for (i, r), row, n in zip(live, K, K.sum(1).tolist()):
                keeps[i] = row
                need[i] = n > 1

        # 1. Advance existing rows once per card when contrast or admission is
        #    needed; other cards keep accumulating deferred tokens.
        running = [(i, r) for i, r in live if not r.pending and r.owned]
        fresh = [r for _, r in live if r.pending and r.owned]
        for r in fresh:
            r.dev = min(range(len(self.streams)), key=lambda d: self.nrows[d])
            self.nrows[r.dev] += 2
        cards = {r.dev for r in fresh} | {r.dev for i, r in running if need[i] and len(r.out) > r.n_fed}
        for d in cards:
            rs = [r for _, r in running if r.dev == d]
            toks = {}
            for r in rs:
                # k >= 0 tokens per request: a request vLLM is recomputing after a preemption has
                # nothing new for a few steps and its rows stall (empty chunk, left-padded by
                # ModelStream.step) while the others on the card advance
                pend = list(r.out[r.n_fed:])
                toks[r.rows[0]] = pend if len(pend) != 1 else pend[0]
                toks[r.rows[1]] = pend if len(pend) != 1 else pend[0]
            if rs and any(len(r.out) > r.n_fed for r in rs):
                self.streams[d].step(toks)
                for r in rs:
                    r.n_fed = len(r.out)
        # the thinking state keeps pace on every rank, fed or not, one token per step
        for i, r in live:
            if not r.pending and len(r.out) > r.n_seen:
                self._advance(r)

        # 2. admit the new ones at the same position, on the card chosen above
        new_by_dev = {}
        for r in fresh:
            new_by_dev.setdefault(r.dev, []).append(r)
        for _, r in live:
            if r.pending and not r.owned:
                r.n_seen, r.n_fed, r.pending = len(r.out), len(r.out), False
        for d, rs in new_by_dev.items():
            prompts = []
            for r in rs:
                prompts.append(torch.tensor(r.brief + list(r.out), dtype=torch.long))
                prompts.append(torch.tensor(r.verbose + list(r.out), dtype=torch.long))
            ids = self.streams[d].admit(prompts)
            for k, r in enumerate(rs):
                r.rows = (ids[2 * k], ids[2 * k + 1])
                r.n_seen, r.n_fed, r.pending = len(r.out), len(r.out), False

        # 3. this rank's share of the contrast (a request that does not need it gets zeros: with a
        #    one-token top-p set the masked result is the same, and its rows may be stale)
        deltas = []
        for i, r in live:
            if not r.in_think or not r.owned or r.rows is None or not need[i]:
                continue
            st = self.streams[r.dev]
            # Move the contrast vector to the device holding the base logits.
            deltas.append((i, (st.logits_of(r.rows[0])[:self.vocab]
                               - st.logits_of(r.rows[1])[:self.vocab]).to(logits.device)))
        # All ranks contribute float32 deltas, including ranks with no owned rows.
        D = torch.zeros(logits.shape[0], self.vocab, dtype=torch.float32, device=logits.device)
        for i, d in deltas:
            D[i] = d.float()
        self._keeps = keeps
        return live, D

    def _finish(self, logits, live, D):
        """Combine all active thinking rows in one batch of tensor operations.

        A request that closed during the small-model step is excluded: these logits predict
        the next token, which is already outside the thinking block.
        """
        out = logits
        V = self.vocab
        sel = [(i, r) for i, r in live
               if r.in_think and not r.pending and not (r.owned and r.rows is None)]
        if not sel or self.alpha == 0:
            self._retire_finished()
            return out
        rows = [i for i, _ in sel]
        idx_t = torch.tensor(rows, device=logits.device, dtype=torch.long)
        ZB = logits[idx_t, :V].float()                                   # (L, V)
        K = torch.stack([self._keeps[i] for i in rows])                   # (L, V) bool
        DL = D[idx_t]                                                    # (L, V) float32, a copy
        Z = ZB + self.alpha * DL
        if self.eos:
            Z[:, self.eos] = ZB[:, self.eos]
        if self.close:
            Z[:, self.close[:1]] = ZB[:, self.close[:1]]
        # index_put_ requires matching dtypes; CPU tests also use float64 logits.
        out[idx_t, :V] = torch.where(K, Z, torch.full_like(ZB, NEG)).to(out.dtype)
        if V < logits.shape[1]:
            out[idx_t, V:] = NEG
        self._retire_finished()
        return out

    def _advance(self, r):
        """Track the thinking boundary even when the small forward is deferred."""
        r.n_seen += 1
        n = len(self.close)
        if n and r.n_seen >= n and list(r.out[r.n_seen - n:r.n_seen]) == self.close:
            r.in_think = False

    def _retire_finished(self):
        """Free the small rows of any request that has left the thinking block."""
        for r in self.reqs:
            if r is not None and not r.in_think and r.rows is not None:
                self._drop_rows(r)
                r.rows = None

    def _drop_rows(self, r):
        if r.rows is not None:
            self.streams[r.dev].evict(list(r.rows))
            self.nrows[r.dev] -= 2

    def _grow(self, idx):
        while len(self.reqs) <= idx:
            self.reqs.append(None)

    def _free(self, idx):
        if idx < len(self.reqs) and self.reqs[idx] is not None:
            self._drop_rows(self.reqs[idx])
            self.reqs[idx] = None
