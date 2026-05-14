"""Role mapping for IOI prompts.

Role labels (IO/S1/S2/END) are recovered from the metadata fields the
ioi prompt pipeline writes into each per-prompt JSON. Selection
methods consume this mapping generically; nothing in selection/ knows
about specific IOI fields.

Order matters only when two roles point at the same token position
(does not occur in IOI prompts in practice, but kept deterministic).
First match wins.
"""

ROLE_KEYS = [
    ("io_position",  "IO"),
    ("s1_position",  "S1"),
    ("s2_position",  "S2"),
    ("end_position", "END"),
]
