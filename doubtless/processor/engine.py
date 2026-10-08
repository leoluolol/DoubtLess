"""Stock vLLM for the base model, with two Transformers rows per contrast request.

The small model uses memory outside vLLM's allocation, so the base defaults to 70%.
"""
import gc
import os
import time

from doubtless.prompts import BRIEF, VERBOSE
from doubtless.records import Sampling, make_run
from doubtless.tokens import load_tokenizer, chat_ids, close_ids as _close_ids, eos_ids as _eos_ids
from doubtless.vllm.loop import run_requests


def _vocab_of(path):
    from transformers import AutoConfig
    return int(AutoConfig.from_pretrained(path).vocab_size)


class ContrastEngine:
    def __init__(self, base, small=None, *, alpha=5.0, p_keep=0.95, brief=BRIEF, verbose=VERBOSE,
                 tensor_parallel_size=1, gpu_memory_utilization=0.70, max_model_len=34816, max_num_seqs=16,
                 seed=0, close_tag="</think>", dtype="bfloat16", small_devices=None,
                 rank_split=True, quiet=False, **vllm_kwargs):
        from vllm import LLM
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self.base, self.small, self.alpha, self.p_keep = base, small, float(alpha), float(p_keep)
        self.brief, self.verbose, self.close_tag = brief, verbose, close_tag
        self.contrast = small is not None
        self.tok = load_tokenizer(base, small)
        self.close_ids = _close_ids(self.tok, close_tag)
        self.eos = _eos_ids(self.tok, base if os.path.isdir(base) else None)
        tp = tensor_parallel_size
        self.tp = tp
        kw = dict(model=base, dtype=dtype, skip_tokenizer_init=True, seed=seed, tensor_parallel_size=tp,
                  max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=int(max_num_seqs),
                  disable_custom_all_reduce=True)
        if self.contrast:
            vocab = min(_vocab_of(base), _vocab_of(small))     # padded vocabularies may differ (OLMo 3 vs OLMo 2)
            devs = small_devices
            if not devs:
                devs = "local"
            kw.update(logits_processors=["doubtless.processor.processor:ContrastLogitsProcessor"],
                      additional_config={"small": small, "alpha": self.alpha, "p_keep": self.p_keep, "close_ids": self.close_ids,
                                         "eos_ids": self.eos, "vocab": vocab, "small_devices": devs, "rank_split": bool(rank_split)})
        kw.update(vllm_kwargs)
        t0 = time.time()
        self.llm = LLM(**kw)
        self.eng = self.llm.llm_engine
        self.max_num_seqs = int(self.eng.vllm_config.scheduler_config.max_num_seqs)
        self.quiet = quiet
        if not quiet:
            print(">> doubtless processor engine up in %.0f s | %s | TP=%d | max_num_seqs %d"
                  % (time.time() - t0, "contrast (%s + %s)" % (base, small) if self.contrast else "plain (%s)" % base, tp, self.max_num_seqs), flush=True)

    def encode(self, prompt, suffix=""):
        base = chat_ids(self.tok, prompt, suffix)
        if not self.contrast:
            return base, [], []
        return (base, chat_ids(self.tok, prompt, suffix + self.brief),
                chat_ids(self.tok, prompt, suffix + self.verbose))

    def generate(self, prompts, sampling=None, *, reps=1, seeds=None, alpha=None, suffix="",
                 max_in_flight=None, admit_per_step=8, keep_token_ids=False, on_run=None, progress_every=60):
        """Generate with the processor backend. A different alpha needs a new engine: the
        processor reads its strength once, at construction."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sampling = sampling or Sampling()
        if alpha is not None and abs(float(alpha) - self.alpha) > 1e-9:
            raise ValueError("the processor backend takes alpha at construction (got %s, engine has %s)" % (alpha, self.alpha))
        enc = [self.encode(p, suffix) for p in prompts]
        cap = min(self.max_num_seqs, int(max_in_flight)) if max_in_flight else self.max_num_seqs

        def seed_of(i, k):
            if seeds and (i, k) in seeds:
                return int(seeds[(i, k)])
            return None if sampling.seed is None else int(sampling.seed) + 100000 * k + 10 * i

        def sp(i, k):
            extra = None
            if self.contrast:
                extra = {"contrast": {"brief_ids": enc[i][1], "verbose_ids": enc[i][2]}}
            return SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k, max_tokens=sampling.max_tokens,
                                  seed=seed_of(i, k), detokenize=False, stop_token_ids=list(self.eos), extra_args=extra)

        runs = []

        def on_done(i, k, out, wall):
            run = make_run(self.tok, i, k, seed_of(i, k), out, len(enc[i][0]), self.eos, self.close_ids, wall, self.close_tag, keep_token_ids)
            runs.append(run)
            if on_run:
                on_run(run)

        run_requests(self.eng, [(i, k) for k in range(reps) for i in range(len(prompts))],
                     lambda i: TokensPrompt(prompt_token_ids=enc[i][0]), sp, on_done, cap, admit_per_step, self.quiet, progress_every)
        runs.sort(key=lambda r: (r.rep, r.index))
        return runs

    def close(self):
        try:
            ec = getattr(self.eng, "engine_core", None)
            if ec is not None and hasattr(ec, "shutdown"):
                ec.shutdown()
        except Exception:
            pass
        self.llm = self.eng = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
