"""Generate IOI/ABC prompt pairs (metadata only, no forward passes)."""

from utils.load_data import load_ioi_with_abc
from experiments.circuits.ioi_utils import resolve_positions


def generate_pairs(tokenizer, n_prompts, seed):
    """Generate IOI/ABC pairs with resolved positions and names.

    Returns list of lightweight dicts (no tensors, no preps).
    Forward passes happen one-at-a-time in run.py.
    """
    eos = tokenizer.eos_token or "<|endoftext|>"
    pairs = load_ioi_with_abc(n_prompts, n_abc_refs=1, tokenizer=tokenizer, seed=seed)
    print(f"{len(pairs)} IOI/ABC pairs generated")

    entries = []
    skipped = 0
    for i, p in enumerate(pairs):
        io_name = p["IO"]
        s_name = p["S"]

        roles = resolve_positions(p["prompt"], io_name, s_name, tokenizer)
        if roles is None or "S2" not in roles:
            skipped += 1
            continue

        io_pos = roles["IO"] + 1
        s1_pos = roles["S1"] + 1
        s2_pos = roles["S2"] + 1

        abc_ids = tokenizer.encode(p["abc_refs"][0], add_special_tokens=False)
        a_name = " " + tokenizer.decode([abc_ids[roles["IO"]]]).strip()
        b_name = " " + tokenizer.decode([abc_ids[roles["S1"]]]).strip()
        c_name = " " + tokenizer.decode([abc_ids[roles["S2"]]]).strip()

        entries.append({
            "io_pos": io_pos, "s1_pos": s1_pos, "s2_pos": s2_pos,
            "io": io_name, "s": s_name, "a": a_name, "b": b_name, "c": c_name,
            "ioi_text": eos + p["prompt"],
            "abc_text": eos + p["abc_refs"][0],
            "template_type": p.get("template_type", ""),
        })

    if skipped:
        print(f"  skipped {skipped} prompts")
    print(f"  {len(entries)} prompts ready")
    return entries
