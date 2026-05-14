"""
analysis.py — Measure and display what changed.

Prediction and Baseline store full probability distributions.
SweepResult tracks predictions across a layer sweep.
compare_logits, kl utilities, and printing functions operate
on results after the fact.

No model execution here — everything works on cached tensors.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Union, Tuple
from unpack._interventions import InterventionRunner


# ================================================================
#  Prediction
# ================================================================

@dataclass
class Prediction:
    """Full prediction snapshot at a single sequence position.

    Stores the complete probability distribution so you can query
    any token's probability, rank, or log-prob after the fact.
    """
    probs: torch.Tensor     # (V,) softmax distribution
    logits: torch.Tensor    # (V,) raw logits
    position: int           # which sequence position this is from

    @staticmethod
    def from_logits(logits_2d: torch.Tensor, position: int = -1) -> 'Prediction':
        """Create from (S, V) or (V,) logits tensor."""
        if logits_2d.dim() == 2:
            raw = logits_2d[position]
        else:
            raw = logits_2d
        return Prediction(
            probs=torch.softmax(raw, dim=-1),
            logits=raw,
            position=position,
        )

    def prob(self, token_id: int) -> float:
        return self.probs[token_id].item()

    def log_prob(self, token_id: int) -> float:
        return self.probs[token_id].clamp(min=1e-10).log().item()

    def rank(self, token_id: int) -> int:
        """1-indexed rank of a token (1 = top prediction)."""
        return (self.probs > self.probs[token_id]).sum().item() + 1

    def top_k(self, tokenizer, k: int = 10) -> List[Tuple[str, float]]:
        vals, idxs = torch.topk(self.probs, k)
        return [(tokenizer.decode(idx).strip(), v.item())
                for idx, v in zip(idxs, vals)]

    def top_token_id(self) -> int:
        return self.probs.argmax().item()


# ================================================================
#  Baseline
# ================================================================

@dataclass
class Baseline:
    """Cached results from a clean forward pass."""
    logits: torch.Tensor               # (B, S, V)
    attentions: List[torch.Tensor]      # list of (B, H, S, S) per layer
    hidden_states: List[torch.Tensor]   # list of (B, S, D), index L = input to layer L
    seq_lens: np.ndarray                # (B,)
    text: str = ""

    def prediction(self, position: int = -1, batch: int = 0) -> Prediction:
        """Get the full prediction at a position."""
        return Prediction.from_logits(self.logits[batch], position)

    def top_predictions(self, tokenizer, k=10, position=-1):
        """Return top-k (token, prob) pairs at given position."""
        return self.prediction(position).top_k(tokenizer, k)


# ================================================================
#  Result Helpers
# ================================================================

def result_prediction(result: InterventionRunner.Result,
                      position: int = -1, batch: int = 0) -> Prediction:
    """Extract a Prediction from an InterventionRunner.Result."""
    return Prediction.from_logits(result.logits[batch], position)


# ================================================================
#  Logit Comparison
# ================================================================

def compare_logits(baseline: 'Baseline', result: InterventionRunner.Result,
                   tokenizer, position: int = -1, top_k: int = 10,
                   track_tokens: Optional[List[int]] = None) -> dict:
    """Compare predictions between baseline and patched run.

    Args:
        track_tokens: Optional list of token IDs to track. For each,
                      returns (token_str, baseline_prob, patched_prob,
                      baseline_rank, patched_rank, prob_drop).
                      If None, automatically tracks baseline's top-1 token.

    Returns dict with keys:
        baseline_pred, patched_pred:  Prediction objects
        baseline_top, patched_top:    [(token_str, prob), ...] top-k
        kl_div:         float, KL(baseline || patched)
        tracked:        list of dicts per tracked token
        top_token:      str, baseline's argmax token
        top_token_rank_change: (1, new_rank)
    """
    b_pred = baseline.prediction(position)
    p_pred = result_prediction(result, position)

    b_top_k = b_pred.top_k(tokenizer, top_k)
    p_top_k = p_pred.top_k(tokenizer, top_k)

    kl = (b_pred.probs.clamp(min=1e-10)
          * (b_pred.probs.clamp(min=1e-10) / p_pred.probs.clamp(min=1e-10)).log()
          ).sum().item()

    top_id = b_pred.top_token_id()
    if track_tokens is None:
        track_tokens = [top_id]

    tracked = []
    for tid in track_tokens:
        tok_str = tokenizer.decode(tid).strip()
        bp = b_pred.prob(tid)
        pp = p_pred.prob(tid)
        tracked.append({
            'token_id':      tid,
            'token':         tok_str,
            'baseline_prob': bp,
            'patched_prob':  pp,
            'prob_drop':     bp - pp,
            'baseline_rank': b_pred.rank(tid),
            'patched_rank':  p_pred.rank(tid),
            'baseline_logit': b_pred.logits[tid].item(),
            'patched_logit':  p_pred.logits[tid].item(),
        })

    return {
        'baseline_pred': b_pred,
        'patched_pred':  p_pred,
        'baseline_top':  b_top_k,
        'patched_top':   p_top_k,
        'kl_div':        kl,
        'tracked':       tracked,
        'top_token':     tokenizer.decode(top_id).strip(),
        'top_token_rank_change': (1, p_pred.rank(top_id)),
    }


# ================================================================
#  Sweep Tracking
# ================================================================

@dataclass
class SweepResult:
    """Collects predictions across a sweep for easy comparison.

    Usage:
        sr = SweepResult(baseline.prediction())
        for label, result in sweep:
            sr.add(label, result_prediction(result))

        sr.print_table(tokenizer, track_tokens=[token_id])
        probs = sr.prob_series(token_id)  # list of floats
    """
    baseline: Prediction
    labels: List[str] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)

    def add(self, label: str, pred: Prediction):
        self.labels.append(label)
        self.predictions.append(pred)

    def prob_series(self, token_id: int) -> List[float]:
        """Probability of token_id across all sweep entries."""
        return [p.prob(token_id) for p in self.predictions]

    def rank_series(self, token_id: int) -> List[int]:
        """Rank of token_id across all sweep entries."""
        return [p.rank(token_id) for p in self.predictions]

    def logit_series(self, token_id: int) -> List[float]:
        return [p.logits[token_id].item() for p in self.predictions]

    def kl_series(self) -> List[float]:
        """KL(baseline || entry) for each sweep entry."""
        bp = self.baseline.probs.clamp(min=1e-10)
        return [
            (bp * (bp / p.probs.clamp(min=1e-10)).log()).sum().item()
            for p in self.predictions
        ]

    def top1_match_series(self) -> List[bool]:
        """Whether each entry's top-1 matches baseline's top-1."""
        base_top = self.baseline.top_token_id()
        return [p.top_token_id() == base_top for p in self.predictions]

    def print_table(self, tokenizer, track_tokens: Optional[List[int]] = None,
                    show_kl: bool = True):
        """Print a summary table across all sweep entries."""
        if track_tokens is None:
            track_tokens = [self.baseline.top_token_id()]

        tok_headers = [tokenizer.decode(tid).strip() for tid in track_tokens]

        header = f"{'Label':>16s}"
        for t in tok_headers:
            header += f"  {t:>8s}_p {t:>8s}_rk"
        if show_kl:
            header += f"  {'KL':>8s}"
        header += f"  {'top1':>10s}"
        print(header)

        row = f"{'baseline':>16s}"
        for tid in track_tokens:
            row += f"  {self.baseline.prob(tid):8.4f} {'1' if self.baseline.rank(tid) == 1 else str(self.baseline.rank(tid)):>8s}"
        if show_kl:
            row += f"  {'0.0000':>8s}"
        top_tok = tokenizer.decode(self.baseline.top_token_id()).strip()
        row += f"  {top_tok:>10s}"
        print(row)

        kl_list = self.kl_series() if show_kl else [None] * len(self.labels)
        for label, pred, kl_val in zip(self.labels, self.predictions, kl_list):
            row = f"{label:>16s}"
            for tid in track_tokens:
                row += f"  {pred.prob(tid):8.4f} {pred.rank(tid):>8d}"
            if show_kl:
                row += f"  {kl_val:8.4f}"
            top_tok = tokenizer.decode(pred.top_token_id()).strip()
            row += f"  {top_tok:>10s}"
            print(row)

    def to_dict(self, token_id: int) -> dict:
        """Export tracked data for a single token as a dict of lists.

        Useful for plotting:
            d = sr.to_dict(token_id)
            plt.plot(d['labels'], d['probs'])
        """
        return {
            'labels': list(self.labels),
            'probs':  self.prob_series(token_id),
            'ranks':  self.rank_series(token_id),
            'logits': self.logit_series(token_id),
            'kl':     self.kl_series(),
        }


def sweep_to_summary(baseline: 'Baseline',
                     sweep_results: List[Tuple[int, InterventionRunner.Result]],
                     position: int = -1,
                     label_fn: Optional[Callable] = None) -> SweepResult:
    """Convert sweep_layers output into a SweepResult.

    Args:
        label_fn: Maps layer_idx to a label string.
                  Default: "layer_{idx}".
    """
    if label_fn is None:
        label_fn = lambda idx: f"layer_{idx}"

    sr = SweepResult(baseline=baseline.prediction(position))
    for layer_idx, result in sweep_results:
        sr.add(label_fn(layer_idx), result_prediction(result, position))
    return sr


# ================================================================
#  Attention KL Utilities
# ================================================================

def kl_per_head(baseline: 'Baseline', result: InterventionRunner.Result,
                layer: int, position: int = -1) -> torch.Tensor:
    """KL divergence per attention head at a given layer and position.

    Returns: (H,) tensor.
    """
    p = baseline.attentions[layer][0, :, position].clamp(min=1e-10)
    q = result.attentions[layer][0, :, position].clamp(min=1e-10)
    return (p * (p / q).log()).sum(dim=-1)


def kl_table_over_layers(baseline: 'Baseline', result: InterventionRunner.Result,
                         layers: Union[List[int], range],
                         position: int = -1) -> torch.Tensor:
    """Build (num_layers, num_heads) KL table for given layers.

    Layers outside `layers` are zero-filled.
    """
    num_layers = len(baseline.attentions)
    num_heads = baseline.attentions[0].shape[1]
    table = torch.zeros(num_layers, num_heads)
    for l in layers:
        table[l] = kl_per_head(baseline, result, l, position)
    return table


# ================================================================
#  Printing
# ================================================================

def print_comparison(baseline: 'Baseline', result: InterventionRunner.Result,
                     tokenizer, position: int = -1, top_k: int = 10,
                     track_tokens: Optional[List[int]] = None):
    """Pretty-print logit comparison with tracked token drops."""
    info = compare_logits(baseline, result, tokenizer, position, top_k,
                          track_tokens=track_tokens)

    print(f"{'':>5}  {'Baseline':>20}  {'Patched':>20}")
    print(f"{'':>5}  {'Token':>12} {'Prob':>7}  {'Token':>12} {'Prob':>7}")
    for i, (b, p) in enumerate(zip(info['baseline_top'], info['patched_top'])):
        print(f"  {i+1:2d}.  {b[0]:>12s} {b[1]:.4f}  {p[0]:>12s} {p[1]:.4f}")

    print(f"\n  KL(baseline || patched) = {info['kl_div']:.4f}")

    print(f"\n  Tracked tokens:")
    for t in info['tracked']:
        print(f"    '{t['token']}': "
              f"prob {t['baseline_prob']:.4f} -> {t['patched_prob']:.4f} "
              f"(drop {t['prob_drop']:+.4f}), "
              f"rank {t['baseline_rank']} -> {t['patched_rank']}")


def print_kl_table(table: torch.Tensor, affected_start: int = 0):
    """Pretty-print per-head and per-layer KL averages."""
    num_layers, num_heads = table.shape

    print("Average KL per head (across affected layers):")
    avg_head = table[affected_start:].mean(dim=0)
    for h in range(num_heads):
        bar = "\u2588" * int(avg_head[h].item() * 100)
        print(f"  Head {h:2d}: {avg_head[h]:.4f}  {bar}")

    print("\nAverage KL per layer (across all heads):")
    avg_layer = table.mean(dim=1)
    for l in range(num_layers):
        bar = "\u2588" * int(avg_layer[l].item() * 100)
        tag = "  (unaffected)" if l < affected_start else ""
        print(f"  Layer {l:2d}: {avg_layer[l]:.4f}  {bar}{tag}")