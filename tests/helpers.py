"""Shared test helpers: tiny random-init models, no downloads needed."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.tokenization_utils_base import BatchEncoding


class MockTokenizer:
    """Minimal tokenizer duck-typed to satisfy trace_flow's needs."""

    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.pad_token = "PAD"
        self.pad_token_id = 0
        self.eos_token = "EOS"
        self.eos_token_id = 1

    def __call__(self, text, return_tensors=None, padding=None,
                 truncation=None, **kwargs):
        # Handle batch input
        if isinstance(text, list):
            all_ids = []
            max_len = 0
            for t in text:
                ids = [min(ord(c) % self.vocab_size, self.vocab_size - 1)
                       for c in t[:32]]
                if not ids:
                    ids = [1]
                all_ids.append(ids)
                max_len = max(max_len, len(ids))
            # Pad to same length
            if padding:
                for ids in all_ids:
                    while len(ids) < max_len:
                        ids.append(self.pad_token_id)
            input_ids = torch.tensor(all_ids, dtype=torch.long)
        else:
            ids = [min(ord(c) % self.vocab_size, self.vocab_size - 1)
                   for c in text[:32]]
            if not ids:
                ids = [1]
            input_ids = torch.tensor([ids], dtype=torch.long)

        attention_mask = (input_ids != self.pad_token_id).long()
        if return_tensors == "pt":
            return BatchEncoding({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [f"tok_{i}" for i in ids]

    def decode(self, ids):
        if isinstance(ids, (list, tuple)) and len(ids) == 1:
            return f"tok_{int(ids[0])}"
        if isinstance(ids, torch.Tensor):
            return "tok_" + "_".join(str(int(x)) for x in ids.tolist())
        return f"tok_{int(ids)}"

    def encode(self, text, add_special_tokens=False):
        return [min(abs(hash(text)) % self.vocab_size, self.vocab_size - 1)]


def build_tiny_gpt2(seed=0):
    """3-layer, 4-head tiny GPT-2. Random weights, no download."""
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=500, n_positions=32, n_embd=48,
        n_layer=3, n_head=4, n_inner=96,
        activation_function="gelu_new",
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
        summary_first_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model, MockTokenizer()


def build_tiny_pythia(seed=0):
    """3-layer, 4-head tiny GPT-NeoX (Pythia-style). Random weights."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(
        vocab_size=500,
        hidden_size=48, num_hidden_layers=3, num_attention_heads=4,
        intermediate_size=96,
        max_position_embeddings=32,
        hidden_dropout=0.0, attention_dropout=0.0,
        use_parallel_residual=True,
    )
    cfg._attn_implementation = "eager"
    model = GPTNeoXForCausalLM(cfg)
    model.eval()
    return model, MockTokenizer()
