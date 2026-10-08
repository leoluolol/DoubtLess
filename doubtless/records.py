"""Sampling settings and generation records shared by both backends."""
from dataclasses import dataclass



@dataclass
class Sampling:
    """Sampling settings; max_tokens limits the total generated tokens."""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 8192
    seed: int | None = None       # None: the engine's own randomness; else every request gets a seed derived from it


@dataclass
class Run:
    index: int                    # position of the prompt in the call
    rep: int                      # which repetition (seed index)
    seed: int | None
    text: str                     # the whole generation, decoded
    thinking: str                 # the thinking block (text before the closing tag)
    answer: str                   # the text after the closing tag ("" if the block never closed)
    think_tokens: int             # tokens up to and including the closing tag (all of them if it never closed)
    completion_tokens: int
    prompt_tokens: int
    finish: str                   # "stop" or "length"
    wall_s: float
    token_ids: list | None = None


def think_tokens_of(out, close_ids):
    """Count tokens through the closing tag using raw IDs to preserve token boundaries."""
    n = len(close_ids)
    for t in range(n, len(out) + 1):
        if list(out[t - n:t]) == list(close_ids):
            return t
    return len(out)


def split_thinking(text, close_tag="</think>"):
    if close_tag in text:
        i = text.index(close_tag)
        return text[:i], text[i + len(close_tag):]
    return text, ""


def make_run(tok, index, rep, seed, out, prompt_tokens, eos_ids, close_ids, wall_s, close_tag="</think>", keep_ids=False):
    text = tok.decode(out)
    thinking, answer = split_thinking(text, close_tag)
    return Run(index=index, rep=rep, seed=seed, text=text, thinking=thinking, answer=answer,
               think_tokens=think_tokens_of(out, close_ids), completion_tokens=len(out), prompt_tokens=prompt_tokens,
               finish="stop" if (out and out[-1] in set(eos_ids)) else "length", wall_s=round(wall_s, 2),
               token_ids=list(out) if keep_ids else None)
