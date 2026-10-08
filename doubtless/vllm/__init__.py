"""The fast backend: a vLLM plugin that runs the base model and the two thinkers in ONE engine.

    from doubtless.vllm import ContrastEngine, Sampling
    eng = ContrastEngine("Qwen/Qwen3-32B", small="Qwen/Qwen3-0.6B", alpha=5.0)
    runs = eng.generate(["What is 17 * 23?"], Sampling(max_tokens=4096))
    print(runs[0].thinking, runs[0].answer, runs[0].think_tokens)

Requires `pip install -e .` (the vLLM plugin entry point) and vLLM 0.29.x.
"""
from doubtless.records import Run, Sampling                 # noqa: F401
from doubtless.vllm.engine import ContrastEngine             # noqa: F401
