"""Prompt placement, tokenizer mapping and generation records; CPU only."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doubtless.records import make_run
from doubtless.tokens import chat_ids, close_ids, load_tokenizer


class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"a": 0, "b": 1}

    def get_vocab(self):
        return self.vocab

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [0, 1]

    def __call__(self, text, **kwargs):
        return {"input_ids": [7, 8] if text else []}

    def decode(self, ids):
        return "reason</think>answer" if 8 in ids else "reason"


class SharedTests(unittest.TestCase):
    def test_instruction_goes_on_last_user(self):
        tok = Tokenizer()
        messages = [{"role": "user", "content": "first"},
                    {"role": "user", "content": "last"},
                    {"role": "tool", "content": "tool result"}]
        chat_ids(tok, messages, suffix=" brief")
        self.assertEqual(tok.messages[1]["content"], "last brief")
        self.assertEqual(tok.messages[2]["content"], "tool result")
        self.assertEqual(messages[1]["content"], "last")

    def test_token_id_mapping(self):
        base = Tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=[base, Tokenizer()]):
            self.assertIs(load_tokenizer("base", "small"), base)
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=[base, Tokenizer({"a": 1, "b": 0})]):
            with self.assertRaises(ValueError):
                load_tokenizer("base", "small")

    def test_closing_tag(self):
        self.assertEqual(close_ids(Tokenizer()), [7, 8])
        with self.assertRaises(ValueError):
            close_ids(Tokenizer(), "")

    def test_result_split_and_finish(self):
        run = make_run(Tokenizer(), 0, 0, 42, [3, 7, 8, 5, 9], 4, [9], [7, 8], 0.1)
        self.assertEqual((run.thinking, run.answer), ("reason", "answer"))
        self.assertEqual((run.think_tokens, run.completion_tokens, run.finish), (3, 5, "stop"))
        capped = make_run(Tokenizer(), 0, 0, 42, [3, 4], 4, [9], [7, 8], 0.1)
        self.assertEqual((capped.answer, capped.think_tokens, capped.finish), ("", 2, "length"))


if __name__ == "__main__":
    unittest.main()
