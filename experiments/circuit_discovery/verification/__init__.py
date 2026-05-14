"""Circuit verification: faithfulness + completeness via mean ablation.

verify(prompts, circuit, *, always_on, ...) → VerifyResult
    For each prompt:
      - clean forward pass
      - faithfulness: ablate NOT(circuit), measure logit-diff
      - completeness: ablate circuit, measure logit-diff

CLI entrypoint in __main__.py reads SelectionResult JSON written by
selection.METHODS via --circuit-file.
"""