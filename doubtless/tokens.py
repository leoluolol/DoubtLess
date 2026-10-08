"""Chat-template tokens, instruction spans and special IDs shared by both backends.

The small model's instruction is inserted into the last user message, before the
assistant header. split_instruction() aligns the resulting token IDs with the
base prompt to recover the inserted span.
"""
import json
import os


def chat_ids(tok, messages, suffix="", enable_thinking=True):
    """Chat-template ids for `messages` (a string is one user message); `suffix` is appended to the LAST user
    message (for the brief / verbose instruction)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    m = [dict(x) for x in messages]
    if suffix:
        user = next((x for x in reversed(m) if x.get("role") == "user"), None)
        if user is None:
            raise ValueError("The contrast instructions require a user message.")
        user["content"] += suffix
    kw = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
    x = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, **kw)
    if hasattr(x, "keys"):
        x = x["input_ids"]
    x = list(x)
    if x and hasattr(x[0], "__len__"):
        x = list(x[0])
    return [int(t) for t in x]


def split_instruction(base_ids, role_ids):
    """role prompt = base prompt with an instruction inserted before the template tail.  Returns
    (inserted ids, tail length); raises if the two do not align that way."""
    c = 0
    while c < min(len(base_ids), len(role_ids)) and base_ids[c] == role_ids[c]:
        c += 1
    t = 0
    while t < min(len(base_ids), len(role_ids)) - c and base_ids[-1 - t] == role_ids[-1 - t]:
        t += 1
    if c + t != len(base_ids):
        raise ValueError("the role prompt is not the base prompt plus an inserted instruction "
                         "(common prefix %d, common tail %d, base length %d)" % (c, t, len(base_ids)))
    return role_ids[c: len(role_ids) - t], t


def template_tail(tok):
    """The chat template's tail after the user message (Qwen3: `<|im_end|>\\n<|im_start|>assistant\\n`, 5 tokens),
    found by inserting a probe sentence into a dummy message."""
    from doubtless.prompts import BRIEF
    base = chat_ids(tok, "x")
    role = chat_ids(tok, "x", suffix=BRIEF)
    _, t = split_instruction(base, role)
    return base[len(base) - t:]


def close_ids(tok, tag="</think>"):
    ids = [int(i) for i in tok(tag, add_special_tokens=False)["input_ids"]]
    if not ids:
        raise ValueError("The thinking closing tag must contain at least one token.")
    return ids


def eos_ids(tok, model_dir=None):
    """End-of-sequence ids: the tokenizer's, the generation config's, and the common names."""
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    if model_dir and os.path.exists(os.path.join(model_dir, "generation_config.json")):
        g = json.load(open(os.path.join(model_dir, "generation_config.json"))).get("eos_token_id")
        for i in (g if isinstance(g, list) else [g]):
            if isinstance(i, int):
                ids.add(i)
    # Qwen also has an ordinary BPE token for "</s>"; only special IDs mark EOS.
    special = set(int(i) for i in (getattr(tok, "all_special_ids", None) or []))
    for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0 and i in special:
            ids.add(i)
    return sorted(ids)


def load_tokenizer(base, small=None):
    """Load the shared token-to-ID mapping used to combine the two models' logits."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if small and small != base:
        other = AutoTokenizer.from_pretrained(small)
        if tok.get_vocab() != other.get_vocab():
            raise ValueError("Base and small model must share the same token-to-ID mapping.")
    return tok
