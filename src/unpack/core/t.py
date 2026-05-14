"""
unpack.tracer - Main entry point for UNPACK attribution.

Usage:
    import unpack

    tracer = unpack.Tracer("gpt2")
    result = tracer.trace(
        "Mary and John went to the store. John gave the bag to",
        target=" Mary",
        distractor=" John",
    )
    result.print()
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from unpack.config import TraceConfig, get_config
from unpack.result import TraceResult, Path
from unpack.circuit import Circuit
from unpack.models import get_adapter, load_model, ModelAdapter
from unpack.core.recursion import backward_recursive, set_beta
from unpack.core.prep import _prepare_trace_inputs
from unpack.core.flow import _run_flow_sweep


class Tracer:
    """UNPACK tracer: tokens, paths, and circuits from a single decomposition.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "gpt2",
            "EleutherAI/pythia-410m-deduped") or filesystem path.
            Ignored if ``model`` is provided directly.
        model: A pre-loaded HuggingFace model. If provided,
            ``tokenizer`` must also be given.
        tokenizer: A pre-loaded tokenizer. Required if ``model`` is given.
        adapter: A custom ModelAdapter instance. If None, auto-detected
            from the model architecture.
        device: Device string ("cpu", "cuda", "cuda:0", or "auto").
        cache_dir: HuggingFace cache directory.
        **load_kwargs: Additional kwargs for model loading (e.g. step=143000
            for Pythia checkpoints).

    Examples:
        # From HuggingFace name
        tracer = Tracer("gpt2")

        # From pre-loaded model
        tracer = Tracer(model=my_model, tokenizer=my_tokenizer)

        # With custom adapter
        tracer = Tracer(model=my_model, tokenizer=my_tok, adapter=MyAdapter())
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        *,
        model=None,
        tokenizer=None,
        adapter: Optional[ModelAdapter] = None,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **load_kwargs,
    ):
        if model is not None:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided when passing a pre-loaded model.")
            self.model = model
            self.tokenizer = tokenizer
        elif model_name_or_path is not None:
            self.model, self.tokenizer = load_model(
                model_name_or_path, device=device, cache_dir=cache_dir,
                **load_kwargs,
            )
        else:
            raise ValueError(
                "Provide either model_name_or_path or model=... + tokenizer=...")

        # Auto-detect or use provided adapter
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = get_adapter(self.model)

        # Register hooks
        self.adapter.register_hooks(self.model)

        self.device = str(self.model.device)

    # ================================================================
    #  Level 1+2: Trace a single prompt
    # ================================================================

    def prepare(
        self,
        text: str,
        *,
        target: Optional[str] = None,
        distractor: Optional[str] = None,
        config: Union[str, TraceConfig, None] = None,
        **kwargs,
    ) -> Tuple[dict, TraceConfig]:
        """Run the forward pass and precompute decomposition matrices.

        This is the expensive step. The returned prep dict can be reused
        with trace_from_prep() for multiple roots without re-running
        the forward pass.

        Returns:
            (prep, cfg) tuple. Pass both to trace_from_prep().
        """
        cfg = get_config(config)
        if kwargs:
            cfg_dict = {
                "beta": cfg.beta,
                "branches": cfg.branches,
                "branch_weights": cfg.branch_weights,
                "aligned": cfg.aligned,
                "mlp_rule": cfg.mlp_rule,
                "top_paths_k": cfg.top_paths_k,
                "path_min_frac": cfg.path_min_frac,
            }
            cfg_dict.update(kwargs)
            cfg = TraceConfig(**cfg_dict)

        set_beta(cfg.beta)

        prep = _prepare_trace_inputs(
            self.model, self.tokenizer, text,
            target_position="last",
            target_token=target,
            distractor_token=distractor,
            hook_manager=self.adapter,
            enable_q_side=cfg.enable_q_side,
            enable_v_side=cfg.enable_v_side,
            mlp_geva_enabled=(cfg.mlp_rule == "weighted"),
            mlp_outproj_enabled=cfg.aligned,
            attn_outproj_enabled=cfg.aligned,
        )

        return prep, cfg

    def trace_from_prep(
        self,
        prep: dict,
        cfg: TraceConfig,
        root: Optional[str] = None,
    ) -> TraceResult:
        """Run backward attribution from a precomputed prep dict.

        This is cheap — just numpy. Call prepare() once, then
        trace_from_prep() many times with different roots.

        Args:
            prep: from prepare().
            cfg: from prepare().
            root: "attn_L_head_H" or "attn_L_head_H@pos", or None for target.
        """
        set_beta(cfg.beta)

        if root is not None:
            comp_name, query_pos = self._parse_root(
                root, prep["t_pos"], prep["seq_len"],
                prep["component_layer"],
            )
            importance = {comp_name: 1.0}
            root_label = f"{comp_name}@{query_pos}"
        else:
            importance = prep["importance"]
            query_pos = prep["t_pos"]
            root_label = "target"

        credit_pct, suppress_ratio, component_flow = _run_flow_sweep(
            importance, prep["attn_shares"], prep["attention_weights"],
            prep["key_decomp"], prep["mlp_principled"], prep["mlp_l2"],
            prep["component_layer"], prep["component_order"],
            prep["num_layers"], prep["num_heads"], prep["seq_len"],
            query_pos,
            query_decomp=prep["query_decomp"] if cfg.enable_q_side else None,
            value_decomp=prep["value_decomp"] if cfg.enable_v_side else None,
            branch_weights=cfg._branch_weights_dict,
            attn_shares_outproj=prep.get("attn_shares_outproj"),
            mlp_geva=prep.get("mlp_geva"),
            mlp_outproj=prep.get("mlp_outproj"),
        )

        credit_raw_rec = np.zeros(prep["seq_len"])
        paths_raw = []
        backward_recursive(
            importance, query_pos,
            prep["attn_shares"], prep["attention_weights"],
            prep["key_decomp"], prep["query_decomp"],
            prep["mlp_l2"], prep["mlp_principled"],
            prep["seq_len"], credit_raw_rec, paths_raw,
            min_frac=cfg.path_min_frac,
            value_decomp=prep["value_decomp"] if cfg.enable_v_side else None,
            branch_weights=cfg._branch_weights_dict,
            mlp_geva=prep.get("mlp_geva"),
            mlp_outproj=prep.get("mlp_outproj"),
            attn_shares_outproj=prep.get("attn_shares_outproj"),
        )

        pos_total_flow = credit_pct[credit_pct > 0].sum()
        rec_pos = credit_raw_rec[credit_raw_rec > 0].sum()
        if rec_pos > 0 and pos_total_flow > 0:
            top_paths_raw = [(p, pos, val / rec_pos * 100)
                             for p, pos, val in paths_raw]
        else:
            top_paths_raw = [(p, pos, 0.0) for p, pos, val in paths_raw]
        top_paths_raw.sort(key=lambda x: abs(x[2]), reverse=True)
        top_paths_raw = top_paths_raw[:cfg.top_paths_k]

        paths_raw_sorted = sorted(paths_raw, key=lambda x: abs(x[2]),
                                  reverse=True)[:cfg.top_paths_k]

        tokens = prep["tokens"]
        path_objects = []
        for (p_str, pos, pct), (_, _, raw) in zip(top_paths_raw, paths_raw_sorted):
            path_objects.append(
                Path.from_raw(p_str, pos, pct, raw, tokens))

        comp_flow_scalar = {}
        for name, arr in component_flow.items():
            if "bias" in name:
                continue
            comp_flow_scalar[name] = float(np.asarray(arr).sum())

        return TraceResult(
            tokens=tokens,
            target_token=prep["target_token_str"],
            target_prob=prep["target_prob"],
            target_logit_centered=prep["target_logit_centered"],
            root=root_label,
            token_attribution=credit_pct,
            paths=path_objects,
            component_flow=comp_flow_scalar,
            importance=importance,
            suppress_ratio=suppress_ratio,
        )

    def trace(
        self,
        text: str,
        *,
        target: Optional[str] = None,
        distractor: Optional[str] = None,
        config: Union[str, TraceConfig, None] = None,
        root: Optional[str] = None,
        **kwargs,
    ) -> TraceResult:
        """Trace a single prompt. Convenience wrapper around prepare + trace_from_prep.

        For multiple roots on the same prompt, use prepare() + trace_from_prep()
        directly to avoid redundant forward passes.
        """
        prep, cfg = self.prepare(text, target=target, distractor=distractor,
                                 config=config, **kwargs)
        return self.trace_from_prep(prep, cfg, root=root)

    # ================================================================
    #  Helpers
    # ================================================================

    @staticmethod
    def _parse_root(root, default_pos, seq_len, component_layer):
        """Parse 'name' or 'name@pos' into (comp_name, query_pos)."""
        if "@" in root:
            comp_name, pos_str = root.rsplit("@", 1)
            query_pos = int(pos_str)
        else:
            comp_name = root
            query_pos = default_pos
        if comp_name not in component_layer:
            examples = list(component_layer.keys())[:6]
            raise ValueError(
                f"root component={comp_name!r} not found. "
                f"Examples: {examples}")
        if query_pos < 0 or query_pos >= seq_len:
            raise ValueError(
                f"root position {query_pos} out of range [0, {seq_len}).")
        return comp_name, query_pos

    def __repr__(self) -> str:
        model_type = getattr(self.model.config, "model_type", "unknown")
        return (f"Tracer(model_type={model_type!r}, "
                f"device={self.device!r}, "
                f"adapter={type(self.adapter).__name__})")