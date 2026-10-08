"""General backend: vLLM for the base model, transformers for the small model.

Use it for model families that the fast backend does not cover, such as EXAONE-4.0 and
Granite-4.2. The small model must share the tokenizer and use full attention.
"""
from doubtless.processor.engine import ContrastEngine  # noqa: F401
from doubtless.records import Sampling, Run  # noqa: F401
