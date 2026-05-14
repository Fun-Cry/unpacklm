"""Selection strategies. Each module exposes a `select()` callable
with the standard signature
    select(prompts, *, terminal_role=None, role_keys=(),
           exclude=DEFAULT_EXCLUDE, **kwargs) -> SelectionResult
"""