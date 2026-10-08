"""DoubtLess: training-free contrastive decoding for efficient reasoning."""

from doubtless.contrast import combine, topp_mask
from doubtless.prompts import BRIEF, VERBOSE
from doubtless.records import Run, Sampling
from doubtless.vllm import ContrastEngine

__version__ = "0.1.0"
__all__ = ["ContrastEngine", "Sampling", "Run", "combine", "topp_mask", "BRIEF", "VERBOSE"]
