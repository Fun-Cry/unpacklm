import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from transformers.models.gpt_neox.modeling_gpt_neox import rotate_half, apply_rotary_pos_emb
import torch.nn.functional as F

from unpack.models.base import ModelAdapter


def _apply_periodic_rope(q, k, position_embeddings):
    """Apply rotary position embeddings to (q, k).

    Symmetric in q vs k: whichever side has been expanded along the
    sequence axis (because per-component decomposition flattens
    (S, C) → S*C) gets cos/sin repeat-interleaved to match. This
    happens for the K side under K-side decomposition (q at original
    length, k at S*C) and for the Q side under Q-side decomposition
    (q at S*C, k at original length).
    """
    cos, sin = position_embeddings
    q_len = q.shape[2]
    k_len = k.shape[2]
    unsqueeze_dim = 1
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    # Match cos/sin to the actual q-side and k-side sequence lengths.
    # The "original" side has length cos.shape[-2]; the "decomposed"
    # side has length original_S * num_components.
    orig_S = cos.shape[-2]
    if q_len == orig_S:
        q_cos, q_sin = cos, sin
    else:
        # q is decomposed: expand cos/sin S→S*C.
        n_comp_q = q_len // orig_S
        q_cos = torch.repeat_interleave(cos, repeats=n_comp_q, dim=-2)
        q_sin = torch.repeat_interleave(sin, repeats=n_comp_q, dim=-2)
    if k_len == orig_S:
        k_cos, k_sin = cos, sin
    else:
        n_comp_k = k_len // orig_S
        k_cos = torch.repeat_interleave(cos, repeats=n_comp_k, dim=-2)
        k_sin = torch.repeat_interleave(sin, repeats=n_comp_k, dim=-2)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rot, q_pass = q_rot.to(q_cos.device), q_pass.to(q_cos.device)
    q_embed = (q_rot * q_cos) + (rotate_half(q_rot) * q_sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)

    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    k_rot, k_pass = k_rot.to(k_cos.device), k_pass.to(k_cos.device)
    k_embed = (k_rot * k_cos) + (rotate_half(k_rot) * k_sin)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class GPTNeoXAdapter(ModelAdapter):
    """
    Hook manager for GPT-NeoX / Pythia models.
    
    Hook points (in execution order through each layer):
    
        embedding                - token embedding output
        attn_{L}_input           - hidden states entering attention (post-layernorm)
        attn_{L}_pre_dense       - concatenated heads before W_O projection
        attn_{L}                 - full attention output after dense projection
        mlp_{L}                  - MLP sublayer output
        resid_{L}                - residual stream after layer L (attn + mlp + skip)
    
    All hook points support intervention via register_intervention().
    Residual stream capture (for iter_source_groups) uses:
    embedding, attn_{L}_pre_dense, mlp_{L}.

    Pythia uses parallel residual: input_layernorm feeds attention,
    post_attention_layernorm feeds MLP.  get_norm_params returns the
    attention LN, get_mlp_norm_params returns the MLP LN.
    """
    def __init__(self, capture_mlp_intermediates=False):
        super().__init__()
        self.model: nn.Module = None
        self.mlp_outputs = []
        self.embedding_outputs = []
        self.pre_dense_inputs = []
        self.attention_input_cache: Dict[int, Dict[str, Any]] = {}
        self.pre_final_norm = None
        self.device: torch.device = None

        # Optional: capture actual dense_h_to_4h outputs for sanity check.
        self.capture_mlp_intermediates = capture_mlp_intermediates
        self.mlp_intermediate_outputs = []

    def register_hooks(self, model):
        self.model = model
        self.device = model.device

        self._wire_hook(model.gpt_neox.embed_in, "embedding", "output", "embedding_outputs")

        for i, layer in enumerate(model.gpt_neox.layers):
            self._wire_attn_input_hook(layer.attention, i)
            self._wire_hook(layer.attention.dense, f"attn_{i}_pre_dense", "input", "pre_dense_inputs")
            self._wire_hook(layer.attention,       f"attn_{i}",           "tuple_output")
            self._wire_hook(layer,                 f"layer_{i}_input",    "input")
            self._wire_hook(layer.mlp,             f"mlp_{i}",            "output", "mlp_outputs")
            self._wire_hook(layer,                 f"resid_{i}",          "tuple_output")

            # Layernorm inputs (for selective branch ablation)
            self._wire_hook(layer.input_layernorm,          f"attn_ln_{i}_input", "input")
            self._wire_hook(layer.post_attention_layernorm, f"mlp_ln_{i}_input",  "input")

            if self.capture_mlp_intermediates:
                def _mlp_intermediate_hook(mod, inp, out, _i=i):
                    self.mlp_intermediate_outputs.append(out.detach().clone().cpu())
                self.handles.append(
                    layer.mlp.dense_h_to_4h.register_forward_hook(_mlp_intermediate_hook)
                )

        def pre_final_norm_hook(mod, inp):
            self.pre_final_norm = inp[0].detach().clone()
        self.handles.append(
            model.gpt_neox.final_layer_norm.register_forward_pre_hook(pre_final_norm_hook)
        )

    def _wire_attn_input_hook(self, attention_module, layer_idx):
        name = f"attn_{layer_idx}_input"
        def hook(mod, args, kwargs, _n=name, _i=layer_idx):
            if not hasattr(mod, 'layer_idx'):
                return
            self.attention_input_cache[_i] = {
                "hidden_states": args[0],
                "attention_mask": kwargs.get("attention_mask"),
                "head_mask": kwargs.get("head_mask"),
                "layer_past": kwargs.get("layer_past"),
                "cache_position": kwargs.get("cache_position"),
                "position_embeddings": kwargs.get("position_embeddings"),
                "kwargs": {k: v for k, v in kwargs.items() if k not in 
                           ["attention_mask", "head_mask", "layer_past", 
                            "cache_position", "position_embeddings"]}
            }
            result, changed = self._run_hook(_n, args[0])
            if changed:
                self.attention_input_cache[_i]["hidden_states"] = result
                return (result,) + args[1:], kwargs
        self.handles.append(
            attention_module.register_forward_pre_hook(hook, with_kwargs=True)
        )
        
    def clear(self):
        super().clear()
        self.mlp_outputs = []
        self.embedding_outputs = []
        self.pre_dense_inputs = []
        self.attention_input_cache = {}
        self.pre_final_norm = None
        self.mlp_intermediate_outputs = []

    # ==========================================
    #  Streaming Source Groups
    # ==========================================

    def iter_source_groups(self):
        if self.embedding_outputs:
            emb = self.embedding_outputs[0].cpu()
            self.embedding_outputs = []
            yield emb.unsqueeze(2), ['embedding'], -1

        num_layers = len(self.mlp_outputs)
        config = self.model.gpt_neox.config
        num_heads = config.num_attention_heads
        head_size = config.hidden_size // num_heads

        for i in range(num_layers):
            layer = self.model.gpt_neox.layers[i]
            merged = self.pre_dense_inputs[i]
            W_O = layer.attention.dense.weight
            b_O = layer.attention.dense.bias

            unmerged = merged.view(*merged.shape[:-1], num_heads, head_size)
            W_O_slices = W_O.split(head_size, dim=1)

            components = []
            names = []
            for j in range(num_heads):
                h_j = unmerged[..., j, :]
                contrib = (h_j @ W_O_slices[j].T).cpu()
                components.append(contrib)
                names.append(f'attn_{i}_head_{j}')

            components.append(b_O.expand_as(self.mlp_outputs[i]).cpu())
            names.append(f'attn_{i}_bias')

            components.append(self.mlp_outputs[i].cpu())
            names.append(f'mlp_{i}')

            group_tensor = torch.stack(components, dim=2)
            del components

            self.pre_dense_inputs[i] = None
            self.mlp_outputs[i] = None

            yield group_tensor, names, i

    def free_attention_cache(self, layer_idx):
        if layer_idx in self.attention_input_cache:
            del self.attention_input_cache[layer_idx]

    # ==========================================
    #  Cross-Attention & Projection
    # ==========================================

    def get_cross_attention_scores(self, layer_idx: int,
                                   new_input_states: torch.Tensor,
                                   include_bias: bool = False,
                                   side: str = "key") -> torch.Tensor:
        """Cross-attention scoring on either the key side or the query side.

        See gpt2/manager.py for the full docstring.

        Pythia uses RoPE: both query and key receive rotary position
        embeddings before the dot product. We apply rope to both sides
        regardless of which is decomposed; only the projection slice
        (Q vs K column-block of query_key_value) and which side is held
        fixed differ between modes.
        """
        if side not in ("key", "query"):
            raise ValueError(f"side must be 'key' or 'query', got {side!r}")
        new_input_states = new_input_states.to(self.device)
        if layer_idx not in self.attention_input_cache:
            raise KeyError(f"No cached inputs for layer {layer_idx}.")
        inputs = self.attention_input_cache[layer_idx]
        attention_module = self.model.gpt_neox.layers[layer_idx].attention
        original_hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        position_embeddings = inputs["position_embeddings"]

        if original_hidden_states.shape[-1] != new_input_states.shape[-1]:
            raise ValueError("Hidden dimension mismatch")

        # Original Q/K/V (with bias from query_key_value), needed to hold
        # the non-decomposed side fixed and to return value_states.
        input_shape_q = original_hidden_states.shape[:-1]
        hidden_shape_q = (*input_shape_q, -1, 3 * attention_module.head_size)
        qkv_q = attention_module.query_key_value(original_hidden_states)
        qkv_q = qkv_q.view(hidden_shape_q).transpose(1, 2)
        original_query, original_key, original_value = qkv_q.chunk(3, dim=-1)

        # Decomposed-side projection of new_input_states (no bias by default;
        # bias is appended as a synthetic component when include_bias=True).
        qkv_kv = F.linear(new_input_states,
                          attention_module.query_key_value.weight.to(self.device),
                          None)
        b, t1_len, d = original_hidden_states.shape
        chunk_size = new_input_states.shape[1] // t1_len
        hidden_dim_qkv = qkv_kv.shape[-1]
        qkv_kv = qkv_kv.view(b, t1_len, chunk_size, hidden_dim_qkv)

        if include_bias:
            expanded_bias = attention_module.query_key_value.bias[None, None, None, :].expand(
                qkv_kv.shape[0], qkv_kv.shape[1], 1, -1).to(self.device)
            qkv_kv = torch.concat([qkv_kv, expanded_bias], dim=2)
        qkv_kv = qkv_kv.reshape(b, -1, hidden_dim_qkv)

        input_shape_kv = qkv_kv.shape[:-1]
        hidden_shape_kv = (*input_shape_kv, -1, 3 * attention_module.head_size)
        qkv_kv = qkv_kv.view(hidden_shape_kv).transpose(1, 2)
        decomposed_q, decomposed_k, _ = qkv_kv.chunk(3, dim=-1)

        if side == "key":
            # Q held fixed at original; K decomposed. RoPE both — original
            # query at original positions, decomposed key at expanded positions.
            # _apply_periodic_rope handles different sequence lengths via
            # broadcasting position_embeddings.
            query_states, key_states = _apply_periodic_rope(
                original_query, decomposed_k, position_embeddings)
        else:
            # K held fixed at original; Q decomposed.
            query_states, key_states = _apply_periodic_rope(
                decomposed_q, original_key, position_embeddings)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) * attention_module.scaling

        return attn_weights, attention_mask, original_value

    def project_values(self, layer_idx, values_states):
        batch_size, num_head, seq_len, dim = values_states.shape
        dense_layer = self.model.gpt_neox.layers[layer_idx].attention.dense
        weight_per_head = dense_layer.weight.view(dense_layer.weight.shape[0], num_head, dim)
        per_head_transformed = torch.einsum("bshd,ohd->bsho", values_states.transpose(1, 2), weight_per_head)
        return per_head_transformed.transpose(1, 2)

    def get_value_weight(self, layer_idx):
        """Return W_V for this layer, shaped (num_heads, d_model, head_dim).

        Pythia's query_key_value is a Linear layer with weight (3*d_model, d_model).
        Per-head layout interleaves Q/K/V: rows [h*3*hs : (h+1)*3*hs] produce
        head h's QKV, with [0:hs] = Q, [hs:2*hs] = K, [2*hs:3*hs] = V.

        For the W_V slice in Linear convention (output = x @ W.T):
            W_V[h] = W[h*3*hs + 2*hs : (h+1)*3*hs, :].T  → (d_model, head_dim)

        Returns shape (H, d_model, head_dim) matching the GPT-2 convention
        used by AttentionScorer._score_value (`bscd,hdk->bhsck`).
        """
        attn = self.model.gpt_neox.layers[layer_idx].attention
        num_heads = self.model.config.num_attention_heads
        head_dim  = attn.head_size
        d_model   = num_heads * head_dim

        W_qkv = attn.query_key_value.weight  # (3*d_model, d_model)

        # Build per-head W_V by slicing out the V rows for each head.
        per_head = []
        for h in range(num_heads):
            base = h * 3 * head_dim
            W_V_h = W_qkv[base + 2 * head_dim : base + 3 * head_dim, :]  # (hs, D)
            per_head.append(W_V_h.T)  # (D, hs)
        return torch.stack(per_head, dim=0).contiguous()  # (H, D, hs)

    def get_value_bias(self, layer_idx):
        """Return b_V for this layer, shaped (num_heads, head_dim).

        Pythia's query_key_value bias has the same per-head interleaved
        QKV layout as the weight: bias[h*3*hs : h*3*hs + 3*hs] is head h's
        full QKV bias, with [2*hs : 3*hs] being the V part.
        """
        attn = self.model.gpt_neox.layers[layer_idx].attention
        num_heads = self.model.config.num_attention_heads
        head_dim  = attn.head_size

        b_qkv = attn.query_key_value.bias  # (3*d_model,)

        per_head = []
        for h in range(num_heads):
            base = h * 3 * head_dim
            b_V_h = b_qkv[base + 2 * head_dim : base + 3 * head_dim]  # (hs,)
            per_head.append(b_V_h)
        return torch.stack(per_head, dim=0).contiguous()  # (H, hs)

    def get_value_at_position(self, layer_idx, hidden_states):
        """Compute the actual V[b, h, s, :] vectors (head_dim) from a
        hidden state. Mirrors what the forward pass computes internally
        before splitting into heads.

        Pythia applies input_layernorm before query_key_value.
        Note: Pythia uses RoPE on Q/K but NOT V, so V is the raw
        post-LN-projected vector.

        Args:
            hidden_states: (B, S, D)

        Returns:
            V: (B, num_heads, S, head_dim)
        """
        layer = self.model.gpt_neox.layers[layer_idx]
        attn = layer.attention
        num_heads = self.model.config.num_attention_heads
        head_dim  = attn.head_size

        # Apply attention LN first (matches forward pass).
        hs = layer.input_layernorm(hidden_states.to(layer.input_layernorm.weight.device))
        # query_key_value projects QKV; output shape (B, S, 3*d_model)
        qkv = attn.query_key_value(hs)
        b, s, _ = qkv.shape
        # Reshape to (B, S, num_heads, 3*head_dim), then take V slice.
        qkv = qkv.view(b, s, num_heads, 3 * head_dim)
        V = qkv[..., 2 * head_dim:3 * head_dim]  # (B, S, H, hs)
        # → (B, H, S, hs)
        V = V.transpose(1, 2).contiguous()
        return V

    # ==========================================
    #  Layer Parameters
    # ==========================================

    def get_norm_params(self, layer_idx):
        """Attention-side layernorm (input_layernorm)."""
        norm_layer = self.model.gpt_neox.layers[layer_idx].input_layernorm
        return norm_layer.weight, norm_layer.bias, norm_layer.eps

    def get_mlp_norm_params(self, layer_idx):
        """MLP-side layernorm (post_attention_layernorm).
        
        Pythia uses parallel residual: attention and MLP each have their
        own layernorm on the same input.
        """
        norm_layer = self.model.gpt_neox.layers[layer_idx].post_attention_layernorm
        return norm_layer.weight, norm_layer.bias, norm_layer.eps
    
    def get_num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def get_num_heads(self) -> int:
        return self.model.config.num_attention_heads

    def get_d_model(self) -> int:
        return self.model.config.hidden_size

    def get_mlp_up_params(self, layer_idx):
        """Return (weight, bias) of dense_h_to_4h for the given layer."""
        mlp = self.model.gpt_neox.layers[layer_idx].mlp
        return mlp.dense_h_to_4h.weight, mlp.dense_h_to_4h.bias

    def get_mlp_down_params(self, layer_idx):
        """Return (weight, bias) of dense_4h_to_h for the given layer."""
        mlp = self.model.gpt_neox.layers[layer_idx].mlp
        return mlp.dense_4h_to_h.weight, mlp.dense_4h_to_h.bias

    def get_final_norm_params(self):
        """Return (weight, bias, eps) for gpt_neox.final_layer_norm."""
        ln = self.model.gpt_neox.final_layer_norm
        return ln.weight, ln.bias, ln.eps

    def get_unembed_weight(self):
        """Return embed_out.weight (vocab_size, d_model)."""
        return self.model.embed_out.weight

    def mlp_up_forward(self, layer_idx, normed_input):
        """GPT-NeoX uses GELU activation after dense_h_to_4h."""
        mlp = self.model.gpt_neox.layers[layer_idx].mlp
        pre_act = mlp.dense_h_to_4h(normed_input)
        activated = torch.nn.functional.gelu(pre_act)
        return pre_act, activated

    # ==========================================
    #  Logit Lens
    # ==========================================
    
    def apply_logit_lens(self, components, marginal=False, final_logits=None):
        layer_norm = self.model.gpt_neox.final_layer_norm
        embed_out = self.model.embed_out
        with torch.no_grad():
            if marginal:
                assert final_logits is not None
                mean = components.mean(dim=-1, keepdim=True)
                components_zero_mean = components - mean
                layer_var = final_logits.var(dim=-1, unbiased=False, keepdim=True)
                layer_std = torch.sqrt(layer_var + layer_norm.eps)
                normalized_components = components_zero_mean / layer_std * layer_norm.weight
            else:
                normalized_components = layer_norm(components)
            outputs = embed_out(normalized_components)
        return outputs
    
    def apply_attn_norm(self, layer_idx, hidden):
        """Apply this layer's pre-attention layer norm (input_layernorm)
        to a tensor. Delegates to the model's actual LN module.
        """
        return self.model.gpt_neox.layers[layer_idx].input_layernorm(hidden)