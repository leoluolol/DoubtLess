"""vLLM generation API. Set small=None for ordinary decoding.

    eng = ContrastEngine("Qwen/Qwen3-32B", small="Qwen/Qwen3-0.6B", alpha=5.0, max_num_seqs=8)
    runs = eng.generate(problems, Sampling(max_tokens=32768, seed=0), reps=3)

Each prompt becomes one request; model.py handles the brief and verbose passes.
The models share a tokenizer and use a fixed assistant header after the last user
message. Contrast mode also uses a thinking-block closing tag and a separate stop
ID, since the end-of-turn token occurs in the forced header.
"""
import gc
import hashlib
import os
import time

from doubtless.prompts import BRIEF, VERBOSE
from doubtless.records import Sampling, make_run
from doubtless.tokens import chat_ids, close_ids as _close_ids, eos_ids as _eos_ids, load_tokenizer, split_instruction, template_tail
from doubtless.vllm.loop import run_requests

ARCH = "DoubtLessContrastForCausalLM"
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
DEFAULT_MAX_NUM_SEQS = 32          # contrast mode: requests in flight, and the size of the instruction buffer


def cache_salt(prompt_ids, brief_ids, verbose_ids):
    """Prefix-cache salt of a contrast request.

    A thinker's K/V for a token depend on whether the token is part of the prompt (no instruction before it) or
    comes after it (instruction visible, positions shifted).  The cache hash sees token ids only, so requests with
    different instructions must not share blocks, and neither may two turns of one conversation: the next turn's
    prompt repeats the previous turn's assistant header, whose thinker K/V were computed after the instruction.
    Keying the salt on the request's own prompt limits reuse to repeated runs of the same prompt, where every
    cached block lies inside the prompt."""
    h = hashlib.sha256()
    for part in (prompt_ids, brief_ids, verbose_ids):
        h.update(len(part).to_bytes(4, "little"))
        h.update(b"".join(int(t).to_bytes(4, "little") for t in part))
    return "contrast:" + h.hexdigest()[:32]


class ContrastEngine:
    def __init__(self, base, small=None, *, alpha=5.0, p_keep=0.95, brief=BRIEF, verbose=VERBOSE,
                 tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=34816, max_num_seqs=None,   # None: vLLM's default for plain decoding, 32 with a small model
                 enforce_eager=False, cudagraph_capture_sizes=None, seed=0, enable_prefix_caching=True,
                 close_tag="</think>", dtype="bfloat16", quiet=False, **vllm_kwargs):
        from transformers import AutoConfig
        from vllm import LLM
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # Use the torch sampler to avoid a separate JIT build.
        self.base, self.small, self.alpha, self.p_keep = base, small, float(alpha), float(p_keep)
        self.brief, self.verbose, self.close_tag = brief, verbose, close_tag
        self.contrast = small is not None
        self.tok = load_tokenizer(base, small)
        self.close_ids = _close_ids(self.tok, close_tag)
        self.eos = _eos_ids(self.tok, base if os.path.isdir(base) else None)
        self.tail = template_tail(self.tok) if self.contrast else []
        # the forced header contains the end-of-turn token: the request must ignore EOS and stop on another id, onto
        # which the model moves the end-of-turn logit once the tail is past (model.py, compute_logits)
        self.end_turn = int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else None
        self.remap = self.contrast and any(t in self.eos for t in self.tail)
        self.stop_id = None
        if self.remap:
            cands = [t for t in self.eos if t not in self.tail and t != self.end_turn]
            if not cands or self.end_turn is None:
                raise ValueError("the tokenizer needs a second end-of-sequence id besides the end-of-turn token %r "
                                 "(Qwen3 has <|endoftext|>); eos ids seen: %s" % (self.end_turn, self.eos))
            self.stop_id = cands[0]
        tp = tensor_parallel_size
        self.tp = tp
        kw = dict(model=base, dtype=dtype, skip_tokenizer_init=True, seed=seed, tensor_parallel_size=tp,
                  max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, enforce_eager=bool(enforce_eager),
                  enable_prefix_caching=bool(enable_prefix_caching))
        if self.contrast and not max_num_seqs:
            # the instruction buffer is allocated per batch slot (2 roles x layers x K,V x S_MAX x kv_heads x head_dim,
            # about 29 MB per slot for Qwen3-0.6B), so vLLM's default of 1024 slots would take 30 GB before the KV cache
            max_num_seqs = DEFAULT_MAX_NUM_SEQS
        if max_num_seqs:
            kw["max_num_seqs"] = int(max_num_seqs)
        if self.contrast:
            # Both models read the same KV blocks through FlashAttention.  vLLM selects the GPU's FA version.
            attention = dict(vllm_kwargs.pop("attention_config", None) or {})
            attention.setdefault("backend", "FLASH_ATTN")
            kw["attention_config"] = attention
            from doubtless.vllm import plugin
            plugin.register()                       # front end; the engine core and the workers load the entry point
            base_arch = (AutoConfig.from_pretrained(base).architectures or ["Qwen3ForCausalLM"])[0]
            sizes = sorted(set(cudagraph_capture_sizes or DEFAULT_CAPTURE_SIZES) | ({int(max_num_seqs)} if max_num_seqs else set()))
            kw.update(hf_overrides={"architectures": [ARCH]},
                      logits_processors=["doubtless.vllm.state:ThinkerStateProcessor"],
                      additional_config={"small": small, "alpha": self.alpha, "p_keep": self.p_keep, "base_arch": base_arch,
                                         "close_ids": self.close_ids, "eos_ids": self.eos, "tail_ids": self.tail,
                                         **({"end_turn_id": self.end_turn, "stop_id": self.stop_id} if self.remap else {})})
            if not enforce_eager:
                # Full decode graphs capture the small model's unpadded positions;
                # piecewise replay assumes a different input layout. Capture each
                # batch size and enable fused ops for the eager small decoder.
                kw["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": sizes,
                                            "custom_ops": ["+silu_and_mul", "+rotary_embedding"]}
        kw.update(vllm_kwargs)
        t0 = time.time()
        self.llm = LLM(**kw)
        self.eng = self.llm.llm_engine
        cc = self.eng.vllm_config.cache_config
        blocks = getattr(cc, "num_gpu_blocks", None) or 0
        self.kv_tokens = int(blocks) * int(cc.block_size) if blocks else None
        self.max_num_seqs = int(self.eng.vllm_config.scheduler_config.max_num_seqs)
        self.quiet = quiet
        if not quiet:
            what = ("contrast (%s + %s)" % (os.path.basename(base.rstrip("/")), os.path.basename(small.rstrip("/")))
                    if self.contrast else "plain decoding (%s)" % os.path.basename(base.rstrip("/")))
            print(">> doubtless engine up in %.0f s | %s | TP=%d | max_num_seqs %d | KV cache %s tokens"
                  % (time.time() - t0, what, tp, self.max_num_seqs, format(self.kv_tokens, ",") if self.kv_tokens else "unknown"), flush=True)

    # ---- prompts ---------------------------------------------------------------------------------------------
    def _ids(self, prompt, suffix=""):
        return chat_ids(self.tok, prompt, suffix=suffix)

    def encode(self, prompt, suffix=""):
        """Return base, brief and verbose IDs. Contrast mode removes the template
        tail from the base prompt; the model generates that tail first."""
        base = self._ids(prompt, suffix)
        if not self.contrast:
            return base, [], []
        b, _ = split_instruction(base, self._ids(prompt, suffix + self.brief))
        v, _ = split_instruction(base, self._ids(prompt, suffix + self.verbose))
        if base[-len(self.tail):] != self.tail:
            raise ValueError("the prompt does not end with the chat template's tail %s: %s" % (self.tail, base[-len(self.tail) - 2:]))
        from doubtless.vllm.state import S_MAX
        if max(len(b), len(v)) > S_MAX:
            raise ValueError("an instruction is longer than S_MAX=%d tokens (%d / %d)" % (S_MAX, len(b), len(v)))
        return base[:-len(self.tail)], b, v

    # ---- generation -----------------------------------------------------------------------------------------
    def generate(self, prompts, sampling=None, *, reps=1, seeds=None, alpha=None, suffix="", max_in_flight=None,
                 admit_per_step=8, keep_token_ids=False, on_run=None, progress_every=60):
        """Decode every prompt `reps` times.  prompts: strings (one user message each) or message lists.
        seeds: optional {(index, rep): seed}; otherwise seed = sampling.seed + 100000*rep + 10*index (or None).
        suffix: text appended to the last user message (for every model).  on_run(run) is called as each run
        finishes.  Returns the runs sorted by (rep, index)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sampling = sampling or Sampling()
        a = self.alpha if alpha is None else float(alpha)
        enc = [self.encode(p, suffix) for p in prompts]
        longest = max(len(e[0]) for e in enc) + len(self.tail)
        need = sampling.max_tokens + longest
        cap = self.max_num_seqs
        if self.kv_tokens:
            fit = self.kv_tokens // need
            if fit < 1:
                raise RuntimeError("the KV cache holds %d tokens and one run may need %d: lower max_tokens, raise "
                                   "gpu_memory_utilization or use more GPUs" % (self.kv_tokens, need))
            if self.contrast and fit < cap:
                cap = fit                    # avoid repeated preemption as long requests grow
        if max_in_flight:
            cap = min(cap, int(max_in_flight))

        def seed_of(i, k):
            if seeds and (i, k) in seeds:
                return int(seeds[(i, k)])
            return None if sampling.seed is None else int(sampling.seed) + 100000 * k + 10 * i

        def sp(i, k):
            common = dict(temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k, seed=seed_of(i, k), detokenize=False)
            if self.contrast:
                extra = {"contrast": {"brief_ids": enc[i][1], "verbose_ids": enc[i][2], "alpha": a}}
                if self.remap:
                    return SamplingParams(max_tokens=sampling.max_tokens + len(self.tail), ignore_eos=True,
                                          stop_token_ids=[int(self.stop_id)], extra_args=extra, **common)
                return SamplingParams(max_tokens=sampling.max_tokens + len(self.tail), stop_token_ids=list(self.eos), extra_args=extra, **common)
            return SamplingParams(max_tokens=sampling.max_tokens, stop_token_ids=list(self.eos), **common)

        def prompt_of(i):
            if self.contrast:
                return TokensPrompt(prompt_token_ids=enc[i][0], cache_salt=cache_salt(*enc[i]))
            return TokensPrompt(prompt_token_ids=enc[i][0])

        runs, n_tail = [], len(self.tail)

        def on_done(i, k, out, wall):
            if self.contrast:
                if out[:n_tail] != self.tail:
                    raise RuntimeError("request (%d, %d) did not start with the forced header: %s" % (i, k, out[:n_tail + 2]))
                out = out[n_tail:]
                if self.remap and out and out[-1] == self.stop_id:
                    out[-1] = self.end_turn          # the model's end of turn, moved onto the stop id
            run = make_run(self.tok, i, k, seed_of(i, k), out, len(enc[i][0]) + n_tail, self.eos, self.close_ids, wall,
                           self.close_tag, keep_token_ids)
            runs.append(run)
            if on_run:
                on_run(run)

        run_requests(self.eng, [(i, k) for k in range(reps) for i in range(len(prompts))], prompt_of, sp, on_done, cap,
                     admit_per_step, self.quiet, progress_every)
        runs.sort(key=lambda r: (r.rep, r.index))
        return runs

    # ---- teardown -------------------------------------------------------------------------------------------
    def close(self):
        """Shut the engine down (vLLM's worker processes included) so another engine can be built in this process."""
        try:
            ec = getattr(self.eng, "engine_core", None)
            if ec is not None and hasattr(ec, "shutdown"):
                ec.shutdown()
        except Exception:
            pass
        self.llm = self.eng = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
