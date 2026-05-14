"""
unpack.tasks.ioi - IOI (Indirect Object Identification) task helpers.

Provides role mappings and prompt generation for the IOI task
introduced by Wang et al. (2023).
"""

from typing import Dict, List, Optional


# ── Role mapping ──
# Maps metadata keys to role labels used in path partitioning.
# Used by tracer.discover() to group paths by structural type.

IOI_ROLE_KEYS = [
    ("io_position",  "IO"),
    ("s1_position",  "S1"),
    ("s2_position",  "S2"),
    ("end_position", "END"),
]


def ioi_roles(metadata: dict) -> Dict[int, str]:
    """Extract a {position: role_name} mapping from IOI prompt metadata.

    Args:
        metadata: the per-prompt metadata dict with keys like
            "io_position", "s1_position", etc.

    Returns:
        {int: str} mapping, e.g. {1: "IO", 5: "S1", 9: "S2", 13: "END"}.
    """
    roles = {}
    for key, label in IOI_ROLE_KEYS:
        pos = metadata.get(key)
        if pos is not None:
            roles[pos] = label
    return roles


# ── Wang et al. (2023) canonical circuit heads ──
# For reference and comparison.

WANG_CIRCUIT_HEADS = {
    # Name movers
    "attn_9_head_6":   "NM",
    "attn_9_head_9":   "NM",
    "attn_10_head_0":  "NM",
    # Backup name movers
    "attn_10_head_10": "BackupNM",
    "attn_11_head_2":  "BackupNM",
    "attn_9_head_0":   "BackupNM",
    "attn_9_head_7":   "BackupNM",
    # Negative name movers
    "attn_10_head_7":  "NegNM",
    "attn_11_head_10": "NegNM",
    # S-inhibition heads
    "attn_7_head_3":   "S-Inh",
    "attn_7_head_9":   "S-Inh",
    "attn_8_head_6":   "S-Inh",
    "attn_8_head_10":  "S-Inh",
    # Induction heads
    "attn_5_head_5":   "Induction",
    "attn_5_head_8":   "Induction",
    "attn_5_head_9":   "Induction",
    "attn_6_head_9":   "Induction",
    # Duplicate-token heads
    "attn_0_head_1":   "DupTok",
    "attn_0_head_10":  "DupTok",
    "attn_3_head_0":   "DupTok",
    # Previous-token heads
    "attn_2_head_2":   "PrevTok",
    "attn_4_head_11":  "PrevTok",
}
