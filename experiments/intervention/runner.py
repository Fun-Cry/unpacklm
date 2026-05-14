"""
runner.py — CausalRunner and helpers.

CausalRunner adds run_baseline() to InterventionRunner.
resolve() converts Intervention specs to the {str: fn} dicts that run() expects.
sweep_layers() is a convenience loop.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Union, Tuple
from unpack._interventions import InterventionRunner

from .interventions import Intervention
from .analysis import Baseline


def resolve(specs: List[Intervention]) -> Dict[str, Callable]:
    """Convert Intervention specs to {hook_name: fn} dict."""
    out = {}
    for spec in specs:
        out.update(spec.resolve_hook_names())
    return out


class CausalRunner(InterventionRunner):
    """InterventionRunner + baseline caching.

    Usage:
        runner = CausalRunner(model, tokenizer, hook_manager)
        baseline = runner.run_baseline(text)

        # Accumulating
        result = runner.run(text, resolve([spec]))

        # Marginal
        result = runner.run(text, resolve([spec]),
                            baseline_hidden_states=baseline.hidden_states)
    """

    def __init__(self, model, tokenizer, hook_manager):
        super().__init__(model, tokenizer, hook_manager)
        self.baseline: Optional[Baseline] = None

    def run_baseline(self, text: str) -> Baseline:
        """Clean forward pass. Caches hidden states for marginal mode."""
        self.hook_manager.clear()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        seq_lens = np.array([m.sum().item() for m in inputs["attention_mask"]])
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            out = self.model(**inputs, output_attentions=True, output_hidden_states=True)

        self.baseline = Baseline(
            logits=out.logits.detach().cpu(),
            attentions=[a.detach().cpu() for a in out.attentions],
            hidden_states=[h.detach().cpu() for h in out.hidden_states],
            seq_lens=seq_lens,
            text=text,
        )
        return self.baseline


def sweep_layers(runner, text, intervention_fn, layers,
                 hook='layer_input', baseline_hidden_states=None,
                 per_layer=True):
    """Run intervention at each layer independently or cumulatively."""
    results = []
    layers = list(layers)
    for i, l in enumerate(layers):
        target = [l] if per_layer else layers[:i + 1]
        interventions = resolve([Intervention(intervention_fn, layers=target, hook=hook)])
        result = runner.run(text, interventions,
                            baseline_hidden_states=baseline_hidden_states)
        results.append((l, result))
    return results