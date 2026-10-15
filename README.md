# DoubtLess

[![arXiv](https://img.shields.io/badge/arXiv-2610.18862-b31b1b.svg)](https://arxiv.org/abs/2610.18862)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

DoubtLess is a training-free contrastive decoding method for efficient LLM reasoning. It shortens chain-of-thought (CoT) traces and reduces overthinking. This repository contains the code for the paper [DoubtLess: Training-Free Contrastive Decoding for Efficient Reasoning](https://arxiv.org/abs/2610.18862).

At every step of the large model's thinking, a small model reads the prompt and the thinking so far twice, once with a **brief** instruction and once with a **verbose** one. The difference between its two logit vectors steers the large model:

```text
logits = base_logits + alpha * (brief_logits - verbose_logits)
```

Steering applies inside `<think>...</think>`, using the base model's top-p candidates. EOS and `</think>` keep the base logits.

## Install

Requires Linux, an NVIDIA GPU and vLLM 0.29.0.

```bash
pip install -e ".[vllm]"
```

## Generate

Generate with Qwen3-32B and Qwen3-0.6B on one 80 GB GPU. Model names can also be local paths.

```python
from doubtless import ContrastEngine, Sampling

with ContrastEngine(
    "Qwen/Qwen3-32B",
    small="Qwen/Qwen3-0.6B",
    alpha=5.0,
    max_model_len=4096,
    max_num_seqs=1,
) as engine:
    results = engine.generate(
        ["What is the sum of the positive divisors of 360?"],
        Sampling(max_tokens=2048, seed=0),
    )

print(results[0].text)
print(results[0].think_tokens)
```

See [examples/generate.py](examples/generate.py). `alpha` controls steering strength; `alpha=0` keeps the base logits. Set `small=None` for ordinary generation and `tensor_parallel_size` for multiple GPUs.

`generate()` accepts a list of prompt strings or chat conversations (`role` / `content` messages). Each result contains `text`, `thinking`, `answer`, token counts and a `finish` reason (`stop` or `length`).

`Sampling` sets `temperature`, `top_p`, `top_k`, `max_tokens` and `seed`. The token budget includes both thinking and the answer.

## Backends

| Import | Implementation | Model pairs |
|---|---|---|
| `from doubtless import ContrastEngine` | Both models in vLLM | Qwen3 |
| `from doubtless.processor import ContrastEngine` | vLLM base model + Transformers small model | Qwen3, EXAONE-4.0, Granite-4.2 |

Both use the same API. The model pair must share token-to-ID mappings and use a thinking closing tag (default `</think>`). The processor backend uses a full-attention small model.

For custom decoding loops, use `doubtless.combine()`. The brief and verbose instructions are in [prompts.py](doubtless/prompts.py).

## Tests

CPU checks for the contrast formula, prompts, request state and small-model KV cache:

```bash
pip install -e .
bash tests/run_cpu_tests.sh
```

On a GPU, check the instruction-attention kernel with `python tests/gpu_test_instruction_attn.py`.

## Citation

```bibtex
@misc{luo2026doubtless,
  title         = {DoubtLess: Training-Free Contrastive Decoding for Efficient Reasoning},
  author        = {Minghao Luo and Liang Chen},
  year          = {2026},
  eprint        = {2610.18862},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.18862}
}
```

Citation metadata is also in [CITATION.cff](CITATION.cff).

## License

[Apache 2.0](LICENSE).
