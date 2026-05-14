"""Prompts for the IOI circuit-discovery experiment.

Builds N IOI sentences and resolves each one's IO / S1 / S2 / END
positions into integer token indices. The token positions are saved
under prompt['target_positions'] (for the lens) and prompt['metadata']
(for downstream task-aware display).

The discover step is task-agnostic — it doesn't know what 'IO' means.
All it knows is "the membership lens reads target_positions from the
prompt dict." The role labels live entirely in the metadata for the
ioi-specific translator (under ioi/translate.py, future work) to read
back when producing role-labeled tables.
"""

from typing import Dict, List, Optional

from utils.load_data import load_ioi_dataset


# Knobs.
N_PROMPTS = 50
SEED      = 42


def _resolve_positions(prompt: str, io_token: str, s_token: str,
                        tokenizer) -> Optional[Dict[str, int]]:
    """Resolve IO/S1/S2/END token positions for one prompt.

    Returns a dict like {"IO": 1, "S1": 2, "S2": 8, "END": 13}
    or None if the prompt has multi-token names.

    Implementation note: IOI metadata gives IO and S with a leading
    space (" Edward"), and most occurrences in the prompt also have a
    leading space — but the very first token of the prompt is the
    sentence-initial word, encoded *without* a leading space and
    therefore with a different BPE id. We accept both id variants so
    sentence-initial IO (ABBA: "Edward and Jack ...") gets resolved.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    def both_ids(s: str):
        """Token ids for the leading-space and bare versions of `s`."""
        s_stripped = s.lstrip()
        spaced = tokenizer.encode(" " + s_stripped, add_special_tokens=False)
        bare   = tokenizer.encode(s_stripped,        add_special_tokens=False)
        if len(spaced) != 1 or len(bare) != 1:
            return None
        return {spaced[0], bare[0]}

    io_set = both_ids(io_token)
    s_set  = both_ids(s_token)
    if io_set is None or s_set is None:
        return None

    out: Dict[str, int] = {}
    s_count = 0
    for i, tid in enumerate(ids):
        if tid in io_set and "IO" not in out:
            out["IO"] = i
        elif tid in s_set:
            s_count += 1
            out[f"S{s_count}"] = i
    out["END"] = len(ids) - 1
    return out


def build_prompts(tokenizer) -> List[Dict]:
    ds = load_ioi_dataset(target=N_PROMPTS, seed=SEED)
    raw = ds.metadata

    out = []
    for d in raw:
        roles = _resolve_positions(d["prompt"], d["IO"], d["S"], tokenizer)
        if roles is None or "IO" not in roles:
            # Drop the prompt; can't resolve positions reliably.
            continue

        prompt_dict = {
            # Required by the discover step.
            "prompt":           d["prompt"],
            "target_token":     d["IO"],
            "distractor_token": d["S"],

            # Used by the membership lens via 'positions_from_metadata'.
            # We default to IO and S2 — the canonical IOI positions of
            # interest. Length and diversity lenses ignore this field.
            "target_positions": [roles["IO"], roles["S1"], roles["S2"]] if "S2" in roles
                    else [roles["IO"]],

            # Saved opaquely with the per-prompt JSON for downstream
            # tools (translator, summarizer) to read.
            "metadata": {
                "io_position":     roles.get("IO"),
                "s1_position":     roles.get("S1"),
                "s2_position":     roles.get("S2"),
                "end_position":    roles.get("END"),
                "io_token":        d["IO"],
                "s_token":         d["S"],
                "template_type":   d["template_type"],
                "all_roles":       roles,
            },
        }
        out.append(prompt_dict)
    return out