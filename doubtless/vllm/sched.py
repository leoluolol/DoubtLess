"""Restore a preempted request's prompt before prefilling the thinkers' instructions.

vLLM 0.29 replays the prompt and generated tokens in chunks after preemption.  Splitting the chunk at the
prompt end lets the attention builder restore the instruction cache before replaying generated tokens.
The hook is shared by the running/waiting queues and AsyncScheduler; ordinary requests are unchanged.
"""
from doubtless.vllm.state import prompt_end_clamp


def install():
    """Install once in each process, from the plugin entry point and the model import."""
    from vllm.v1.core.sched.scheduler import Scheduler
    original = Scheduler._reserve_prefill_lookahead
    if getattr(original, "_doubtless", False):
        return

    def reserve(self, request, num_computed_tokens, num_new_tokens):
        n = original(self, request, num_computed_tokens, num_new_tokens)
        extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
        if n > 0 and "contrast" in extra:
            return prompt_end_clamp(request.num_prompt_tokens, num_computed_tokens, n)
        return n

    reserve._doubtless = True
    Scheduler._reserve_prefill_lookahead = reserve
