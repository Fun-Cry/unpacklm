"""ABC mean-ablation reference preparation for IOI.

Following Wang et al., an ABC reference is a same-template prompt where
the three name slots are filled with three random distinct names not
matching the target's IO or S. The IOI duplication signal is destroyed,
so mean-ablating to ABC-mean is closer to "remove task-specific signal"
than mean-ablating to other-IOI-prompts (which still carry partial
task signal).

Public surface:

    add_abc_references(prompt_jsons, tokenizer, n_abc_refs, ...)
        For each prompt JSON, generates n_abc_refs ABC corruptions and
        returns prompts in the canonical "prompts-with-references"
        format used by both verification and ablation_tracing.runner.

The IOI-specific generation lives at utils.load_data.build_abc_for_target;
this module is the thin adapter that takes saved trace JSONs and wires
their metadata into the ABC builder.
"""

import os
import sys
from typing import List, Optional


from utils.load_data import build_abc_for_target, verify_single_token_pool


def add_abc_references(
    prompt_jsons: List[dict],
    tokenizer,
    n_abc_refs: int = 10,
    abc_seed_offset: int = 10000,
    keep_only: Optional[set] = None,
    strict_length: bool = True,
) -> List[dict]:
    """Convert per-prompt trace JSONs into the prompt-dict format used by
    the verifier and ablation_tracing.runner.

    Each output dict has:
        prompt:           str        target sentence
        target_token:     str        with leading space
        distractor_token: str        with leading space
        references:       List[str]  ABC corruptions, length-matched to target
        metadata:         dict       passthrough for downstream code

    Prompts whose metadata can't be inverted into a template (e.g.
    custom prompts not produced by the IOI loader) are silently
    skipped with a printed warning.

    Pool of names is single-token-verified per the current tokenizer;
    multi-token entries are dropped, which is logged once on first
    pool fetch.

    Length match (when strict_length=True): mean ablation requires the
    target and each ABC ref to tokenize to the same length, since the
    ablation reads activations at corresponding positions. Although
    `verify_single_token_pool` filters the ABC name pool to single-token
    entries, the *target* prompt's IO/S may have been multi-token (the
    original IOI dataset doesn't filter), or BPE may merge differently
    for different name combinations. We tokenize each generated ABC
    ref and drop any that don't match the target's token length. If
    fewer than 1 ref survives, the prompt is skipped entirely.
    """
    pool = verify_single_token_pool(tokenizer)
    out = []
    n_total_refs_built     = 0
    n_total_refs_dropped   = 0
    n_prompts_skipped      = 0
    for i, p in enumerate(prompt_jsons):
        if keep_only is not None and p["prompt"] not in keep_only:
            continue
        meta = p.get("metadata", {})
        if not all(k in meta for k in ("template_type", "IO", "S")):
            continue
        meta_for_abc = {
            "prompt":        p["prompt"],
            "IO":            meta["IO"],
            "S":             meta["S"],
            "template_type": meta["template_type"],
        }
        try:
            abc_refs = build_abc_for_target(
                meta_for_abc,
                n_refs=n_abc_refs,
                names=pool["names"],
                seed=abc_seed_offset + i,
            )
        except (ValueError, RuntimeError) as e:
            print(f"  [skip] {p['prompt'][:50]!r}: {e}")
            continue

        n_total_refs_built += len(abc_refs)

        if strict_length:
            target_len = len(tokenizer.encode(p["prompt"],
                                              add_special_tokens=False))
            kept_refs = []
            for r in abc_refs:
                r_len = len(tokenizer.encode(r, add_special_tokens=False))
                if r_len == target_len:
                    kept_refs.append(r)
            n_total_refs_dropped += (len(abc_refs) - len(kept_refs))
            abc_refs = kept_refs
            if not abc_refs:
                n_prompts_skipped += 1
                print(f"  [skip] {p['prompt'][:50]!r}: "
                      f"all {n_abc_refs} ABC refs were length-mismatched "
                      f"(target={target_len} tokens)")
                continue

        out.append({
            "prompt":           p["prompt"],
            "target_token":     p["target_token"],
            "distractor_token": p["distractor_token"],
            "references":       abc_refs,
            "metadata":         dict(meta),
        })

    if strict_length and n_total_refs_dropped > 0:
        print(f"  [length filter] dropped {n_total_refs_dropped} of "
              f"{n_total_refs_built} ABC refs as length-mismatched "
              f"(skipped {n_prompts_skipped} prompts entirely)")

    return out