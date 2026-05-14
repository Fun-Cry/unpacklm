import torch
import torch.nn.functional as F
from unpack.models.base import ModelAdapter
from unpack.core.streamer import _marginal_normalize 
# ==========================================
#  Attention Scorer
# ==========================================

class AttentionScorer:
    """
    Compute per-component cross-attention scores from normalized components.

    Normalizes raw components with the attention-side layernorm
    (input_layernorm) before scoring.
    """

    def __init__(self, hook_manager: ModelAdapter):
        self.hook_manager = hook_manager

    def score(self, target_layer, components, component_names, hidden,
              is_last_group, *, side="key",
              comp_batch_size=1, value_states=None):
        """Unified per-component attention scoring entrance.

        Dispatches to a side-specific scorer:
          side="key"   → _score_qk(side="key")    — decompose K[s]
          side="query" → _score_qk(side="query")  — decompose Q[q]
          side="value" → _score_value()           — decompose V[s] in head_dim

        Returns shape depends on side:
          side in ("key","query"):
              (names, scores, attn_mask, value_states)
              scores: list of per-component tensors (B, H, Q, K)
          side="value":
              (names, scores)
              scores: tensor (B, C, H, S) — per-component, per-head, per-source

        Args:
            target_layer:    int — which layer to score against
            components:      (B, S, C, D) — raw residual components
            component_names: list[str] of length C
            hidden:          (B, S, D) — hidden state for variance
            is_last_group:   controls whether norm_bias / attn_bias / value-bias
                             entries are appended
            side:            "key", "query", or "value"
            comp_batch_size: only used for K/Q; ignored for V
            value_states:    only used for V; required when side="value"
        """
        if side in ("key", "query"):
            return self._score_qk(target_layer, components, component_names,
                                  hidden, is_last_group, comp_batch_size,
                                  side=side)
        elif side == "value":
            if value_states is None:
                raise ValueError("score(side='value') requires value_states")
            return self._score_value(target_layer, components, component_names,
                                     hidden, is_last_group, value_states)
        else:
            raise ValueError(
                f"side must be 'key', 'query', or 'value', got {side!r}")

    def _score_qk(self, target_layer, components, component_names, hidden,
                  is_last_group, comp_batch_size=1, side="key"):
        """
        Normalize with attention LN, then compute per-component attention scores.

        Args:
            target_layer:    int — which layer's attention to score against
            components:      (B, S, C, D) — raw residual components
            component_names: list[str] of length C
            hidden:          (B, S, D) — hidden state for variance
            is_last_group:   controls whether norm_bias / attn_bias are appended
            comp_batch_size: components per cross-attention call
            side:            "key" (default, legacy) or "query".
                             "key"   → decompose K[s], hold Q[q] fixed
                             "query" → decompose Q[q], hold K[s] fixed

        Returns:
            (names, scores, attn_mask, value_states)
            names:        list[str] — includes "norm_bias" + "attn_bias" when is_last_group
            scores:       list[Tensor] — per-component (B, H, Q, K)
            attn_mask:    Tensor or None
            value_states: Tensor (B, H, K, D)

        Each per-component score tensor has shape (B, H, Q, K). For both
        sides, the entry [b, h, q, k] is the contribution of that
        component to the (q, k) attention logit, but the component is
        indexed differently:
            side="key":    component at source position k
            side="query":  component at query position q
        """
        if side not in ("key", "query"):
            raise ValueError(f"side must be 'key' or 'query', got {side!r}")
        b, s, c, d = components.shape
        norm_w, norm_b, norm_eps = self.hook_manager.get_norm_params(target_layer)

        normalized = _marginal_normalize(components, hidden, norm_w, norm_eps)

        current_names = list(component_names)
        if is_last_group:
            bias_comp = norm_b.to(normalized.device)[None, None, None, :].expand(b, s, 1, d)
            normalized = torch.cat([normalized, bias_comp], dim=2)
            current_names.append("norm_bias")

        # Cross-attention scoring
        event_names = []
        event_scores = []
        attn_mask = None
        v_states = None

        splits = torch.split(normalized, comp_batch_size, dim=2)
        name_offset = 0

        for chunk_i, comp_batch in enumerate(splits):
            is_last_chunk = (chunk_i == len(splits) - 1) and is_last_group
            chunk_c = comp_batch.shape[2]

            kv_input = comp_batch.reshape(b, -1, d)
            chunk_scores, chunk_mask, chunk_values = (
                self.hook_manager.get_cross_attention_scores(
                    layer_idx=target_layer,
                    new_input_states=kv_input,
                    include_bias=is_last_chunk,
                    side=side,
                )
            )

            if attn_mask is None:
                attn_mask = chunk_mask
                v_states = chunk_values

            n_out = chunk_c + (1 if is_last_chunk else 0)
            # Reshape to expose the per-component axis.
            #   side="key":   chunk_scores is (B, H, Q, K*C_eff) → split last axis
            #   side="query": chunk_scores is (B, H, Q*C_eff, K) → split 2nd-last axis
            # In both cases we end with (B, H, Q, K) per component.
            if side == "key":
                # (B, H, Q, K, C)  via reshape on last axis
                chunk_scores = chunk_scores.view(
                    *chunk_scores.shape[:-1], s, n_out
                )
                # split into per-component (B, H, Q, K) tensors
                per_comp = [t.squeeze(-1)
                            for t in torch.split(chunk_scores, 1, dim=-1)]
            else:
                # (B, H, Q, C, K)  via reshape on 2nd-last axis
                chunk_scores = chunk_scores.view(
                    *chunk_scores.shape[:-2], s, n_out, chunk_scores.shape[-1]
                )
                # split into per-component (B, H, Q, K) tensors
                per_comp = [t.squeeze(-2)
                            for t in torch.split(chunk_scores, 1, dim=-2)]

            chunk_names = current_names[name_offset:name_offset + chunk_c]
            if is_last_chunk:
                chunk_names.append("attn_bias")

            event_names.extend(chunk_names)
            event_scores.extend(per_comp)
            name_offset += chunk_c

        return event_names, event_scores, attn_mask, v_states

    def project_values(self, layer_idx, value_states):
        """Project value states through W_O to residual stream space."""
        return self.hook_manager.project_values(layer_idx, value_states)

    def _score_value(self, target_layer, components, component_names, hidden,
                     is_last_group, value_states):
        """Per-component value-route scoring against the head's V[s].

        For each component c, head h, source position s:
            value_score[c, h, s] = <LN(c(s)) · W_V[h], V_actual[h, s]>

        Both vectors are in head_dim space (no W_O involved). The
        rationale: attention has already told us which positions
        matter (via α). At a position s, the head's V[s] tells us
        which direction in head_dim matters. Components contributing
        to V[s] in head_dim get credit by alignment.

        Sum over c (with biases included) ≈ <V[h,s], V[h,s]> = ||V[h,s]||²
        (modulo LN approximation).

        Args:
            target_layer:    int — which layer's W_V to use
            components:      (B, S, C, D) — raw residual components
            component_names: list[str] of length C
            hidden:          (B, S, D) — for marginal LN
            is_last_group:   include norm_bias when True
            value_states:    (B, H, S, head_dim) — head's actual V[s]
                             from forward pass (with bias)

        Returns:
            (names, scores)
            names:   list[str], length C (or C+1 with norm_bias)
            scores:  Tensor (B, C_eff, H, S) — per-component, per-head
                     alignment of c's V-contribution with V[h, s]
        """
        b, s, c, d = components.shape
        norm_w, norm_b, norm_eps = self.hook_manager.get_norm_params(target_layer)

        normalized = _marginal_normalize(components, hidden, norm_w, norm_eps)

        current_names = list(component_names)
        if is_last_group:
            bias_comp = norm_b.to(normalized.device)[None, None, None, :].expand(b, s, 1, d)
            normalized = torch.cat([normalized, bias_comp], dim=2)
            current_names.append("norm_bias")
            c = c + 1

        # W_V per head: (H, D, head_dim)
        W_V = self.hook_manager.get_value_weight(target_layer).to(normalized.device)
        H, _, head_dim = W_V.shape

        # Project each component through W_V[h]: (B, S, C, D) × (H, D, head_dim)
        # → (B, H, S, C, head_dim)
        # einsum: bscd,hdk -> bhsck
        comp_v = torch.einsum("bscd,hdk->bhsck", normalized, W_V)

        # Dot with the head's actual V[h, s, :] (head_dim) → (B, H, S, C)
        # value_states: (B, H, S, head_dim)
        score = torch.einsum("bhsck,bhsk->bhsc", comp_v,
                             value_states.to(comp_v.device))
        # Reshape to (B, C, H, S)
        score = score.permute(0, 3, 1, 2).contiguous()

        # ── value-bias synthetic component (only on last group) ──
        # V_actual[h,s] = LN(residual(s)) · W_V[h] + b_V[h]
        # The b_V[h] term is not produced by any residual-stream
        # component; treat it as its own "value_bias" entry so the
        # closure Σ_c value_decomp == ||V[h,s]||² holds exactly.
        # Per-(h,s) contribution: <b_V[h], V_actual[h,s]>
        if is_last_group:
            b_V = self.hook_manager.get_value_bias(target_layer)  # (H, head_dim)
            if b_V is not None:
                b_V = b_V.to(score.device)
                v_for_bias = value_states.to(score.device)
                # einsum: hk,bhsk -> bhs
                vb_score = torch.einsum("hk,bhsk->bhs", b_V, v_for_bias)
                # → (B, 1, H, S) and concat onto C axis
                vb_score = vb_score.unsqueeze(1)
                score = torch.cat([score, vb_score], dim=1)
                current_names.append("value_bias")

        return current_names, score

    # ==========================================
    #  end of AttentionScorer
    # ==========================================

class MLPScorer:
    """
    Compute per-component element-wise L2 norms through combined LN + W_up.

    Normalizes raw components with the MLP-side layernorm
    (post_attention_layernorm for parallel-residual models) then projects
    through dense_h_to_4h using a Gram matrix for efficiency.

    Only one Gram matrix is held at a time (lazy, single-layer cache).
    """

    def __init__(self, hook_manager: ModelAdapter):
        self.hook_manager = hook_manager
        self.device = hook_manager.device

        # Lazy single-layer cache
        self._cached_layer = -1
        self._cached_gram = None       # (D, D) on device
        self._cached_bias_norm = 0.0

    def _get_gram(self, layer_idx):
        """Return cached Gram matrix, recomputing if target layer changed."""
        if layer_idx != self._cached_layer:
            W_up, b_up = self.hook_manager.get_mlp_up_params(layer_idx)
            self._cached_gram = W_up.T @ W_up
            self._cached_bias_norm = torch.norm(b_up).item()
            self._cached_layer = layer_idx
        return self._cached_gram, self._cached_bias_norm

    def score(self, target_layer, components, component_names, hidden,
              is_last_group):
        """
        Normalize with MLP LN, then compute per-component L2 norms via Gram matrix.

        Args:
            target_layer:    int — which layer's MLP to project through
            components:      (B, S, C, D) — raw residual components
            component_names: list[str] of length C
            hidden:          (B, S, D) — hidden state for variance
            is_last_group:   controls whether mlp_norm_bias / mlp_up_bias are appended

        Returns:
            (names, norms_np)
            names:    list[str]
            norms_np: (B, S, C') numpy array of element-wise L2 norms
        """
        b, s, c, d = components.shape
        norm_w, norm_b, norm_eps = self.hook_manager.get_mlp_norm_params(target_layer)

        normalized = _marginal_normalize(components, hidden, norm_w, norm_eps)

        mlp_names = list(component_names)
        if is_last_group:
            bias_comp = norm_b.to(normalized.device)[None, None, None, :].expand(b, s, 1, d)
            normalized = torch.cat([normalized, bias_comp], dim=2)
            mlp_names.append("mlp_norm_bias")

        # Gram-matrix L2 norms
        G, up_bias_norm = self._get_gram(target_layer)
        B, S, C_total, D = normalized.shape

        flat = normalized.reshape(-1, D).to(self.device)
        sq_norms = (flat @ G * flat).sum(dim=-1)
        norms = torch.sqrt(sq_norms.clamp(min=0)).reshape(B, S, C_total)

        if is_last_group:
            bias_col = torch.full((B, S, 1), up_bias_norm, device=norms.device)
            norms = torch.cat([norms, bias_col], dim=-1)
            mlp_names.append("mlp_up_bias")

        return mlp_names, norms.cpu().numpy()