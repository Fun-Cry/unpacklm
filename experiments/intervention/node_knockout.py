"""
node_knockout.py — Communication-only ablation for sender components.

Per source layer, selects N components spread across strength range.
Two analyses:
  - Within-layer: r(strength, Δppl) per layer, separately for attn/mlp
  - Across-layer: mean Δppl per layer (if not decreasing → strength > depth)

Usage:
    python -m experiments.intervention.node_knockout \
        --cache-dir /data/s4283341 --name 160m_143k \
        --model-size 160m --step 143000 \
        --per-layer 5 --max-sentences 200
"""

import argparse
import json
import os
import re
import sqlite3
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from unpack.core import ComponentStreamer


# ==================================================================
#  Parsing
# ==================================================================

def _parse_names(raw):
    if isinstance(raw, str):
        try:
            return eval(raw)
        except Exception:
            return []
    return list(raw)


def parse_sender_label(label):
    if label == "embedding":
        return ("embedding", -1, None)
    m = re.match(r"mlp_(\d+)", label)
    if m:
        return ("mlp", int(m.group(1)), None)
    m = re.match(r"attn_(\d+)_head_(\d+)", label)
    if m:
        return ("attn_head", int(m.group(1)), int(m.group(2)))
    m = re.match(r"attn_(\d+)_bias", label)
    if m:
        return ("attn_bias", int(m.group(1)), None)
    return ("unknown", None, None)


def get_source_hook(label):
    ht, layer, _ = parse_sender_label(label)
    if ht == "mlp":
        return f"mlp_{layer}"
    elif ht == "attn_head":
        return f"attn_{layer}"
    elif ht == "embedding":
        return "embedding"
    return None


# ==================================================================
#  Load & select
# ==================================================================

def load_senders(summary_path, model_id=1):
    prefix = f"model_{model_id}"
    strengths = {}

    with h5py.File(summary_path, "r") as f:
        meta = f.get(f"{prefix}/meta")
        attn_names = _parse_names(meta.attrs.get("attn_component_names", "[]")) if meta else []
        mlp_names = _parse_names(meta.attrs.get("mlp_component_names", "[]")) if meta else []
        gbl = f.get(f"{prefix}/global")
        if not gbl:
            return []

        if "attn_edge_mean" in gbl:
            totals = np.abs(np.array(gbl["attn_edge_mean"])).sum(axis=(0, 1))
            if not attn_names:
                attn_names = [f"comp_{i}" for i in range(len(totals))]
            for c, name in enumerate(attn_names):
                strengths.setdefault(name, {"attn": 0.0, "mlp": 0.0})
                strengths[name]["attn"] = float(totals[c])

        if "mlp_edge_mean" in gbl:
            totals = np.abs(np.array(gbl["mlp_edge_mean"])).sum(axis=0)
            if not mlp_names:
                mlp_names = [f"comp_{i}" for i in range(len(totals))]
            for c, name in enumerate(mlp_names):
                strengths.setdefault(name, {"attn": 0.0, "mlp": 0.0})
                strengths[name]["mlp"] = float(totals[c])

    senders = []
    for label, s in strengths.items():
        ht, src, _ = parse_sender_label(label)
        if ht in ("unknown", "attn_bias"):
            continue
        senders.append({"label": label, "source_layer": src,
                        "attn_strength": s["attn"], "mlp_strength": s["mlp"]})
    return senders


def select_per_layer(senders, per_layer):
    by_layer = defaultdict(list)
    for s in senders:
        if s["attn_strength"] > 1e-6 or s["mlp_strength"] > 1e-6:
            by_layer[s["source_layer"]].append(s)

    selected = []
    for layer in sorted(by_layer.keys()):
        nodes = by_layer[layer]
        if len(nodes) <= per_layer:
            selected.extend(nodes)
            continue

        # Spread by attn_strength
        vals = np.array([n["attn_strength"] for n in nodes])
        targets = np.linspace(vals.min(), vals.max(), per_layer)
        used = set()
        for t in targets:
            for idx in np.argsort(np.abs(vals - t)):
                if idx not in used:
                    selected.append(nodes[idx])
                    used.add(idx)
                    break

    return sorted(selected, key=lambda n: (n["source_layer"], n["attn_strength"]))


def build_plan(summary_path, per_layer, model_id=1):
    senders = load_senders(summary_path, model_id)
    plan = select_per_layer(senders, per_layer)

    by_layer = defaultdict(list)
    for s in plan:
        by_layer[s["source_layer"]].append(s)
    for l in sorted(by_layer.keys()):
        nodes = by_layer[l]
        print(f"  layer {l:>3}: {len(nodes)} nodes")
    print(f"  Total: {len(plan)}")
    return plan


# ==================================================================
#  Ablation
# ==================================================================

def _batched_loss(model, input_ids, attention_mask):
    """Compute total cross-entropy loss over non-padding tokens."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    per_token = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1), reduction="none")
    per_token = per_token.view(shift_labels.shape)
    total_loss = (per_token * shift_mask).sum().item()
    total_tokens = shift_mask.sum().item()
    return total_loss, int(total_tokens)


def compute_perplexity_baseline(model, tokenizer, hook_manager, sentences,
                                batch_size=16):
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True).to(model.device)
        hook_manager.clear()
        hook_manager.clear_interventions()
        loss, tokens = _batched_loss(model, inputs["input_ids"],
                                     inputs["attention_mask"])
        total_loss += loss
        total_tokens += tokens
    if total_tokens == 0:
        return float("inf"), float("inf"), 0
    mean_loss = total_loss / total_tokens
    return np.exp(mean_loss), mean_loss, total_tokens


def extract_batch_components(model, tokenizer, hook_manager, batch, comp_names):
    """Extract multiple component vectors in one forward pass + decomposition."""
    hook_manager.clear()
    hook_manager.clear_interventions()
    streamer = ComponentStreamer(model, tokenizer, hook_manager)
    streamer.set_context(batch)
    name_set = set(comp_names)
    found = {}
    for group_tensor, group_names, src_layer in hook_manager.iter_source_groups():
        for name in group_names:
            if name in name_set:
                idx = group_names.index(name)
                found[name] = group_tensor[:, :, idx, :].clone()
        if len(found) == len(name_set):
            break
    return found


def compute_perplexity_with_vec(model, hook_manager, input_ids, attention_mask,
                                comp_vec, comp_name, source_layer, num_layers,
                                channel):
    """Pass 2 only: intervene with a pre-extracted component vector."""
    from .interventions import mute, add

    hook_manager.clear()
    hook_manager.clear_interventions()

    downstream = list(range(source_layer + 1, num_layers))
    if channel == "both":
        source_hook = get_source_hook(comp_name)
        last_layer_input = f"layer_{num_layers - 1}_input"
        hook_manager.register_interventions({
            source_hook: mute(comp_vec),
            last_layer_input: add(comp_vec),
        })
    else:
        interventions = {}
        for L in downstream:
            if channel == "attn":
                interventions[f"attn_ln_{L}_input"] = mute(comp_vec)
            else:
                interventions[f"mlp_ln_{L}_input"] = mute(comp_vec)
        hook_manager.register_interventions(interventions)

    loss, tokens = _batched_loss(model, input_ids, attention_mask)
    hook_manager.clear()
    hook_manager.clear_interventions()
    return loss, tokens


# ==================================================================
#  Main
# ==================================================================

def load_sentences_from_db(db_path, max_sentences=None):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, content FROM Sentences ORDER BY id").fetchall()
    conn.close()
    sentences = [row[1] for row in rows]
    if max_sentences and len(sentences) > max_sentences:
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(sentences), max_sentences, replace=False)
        sentences = [sentences[i] for i in sorted(idxs)]
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Per-layer sender knockout")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--deduped", action="store_true")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--model-id", type=int, default=1)
    parser.add_argument("--per-layer", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=200)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available)")
    args = parser.parse_args()

    summary_path = os.path.join(args.cache_dir, f"{args.name}_summary.h5")
    db_path = os.path.join(args.cache_dir, f"{args.name}.db")
    out_path = args.out or os.path.join(args.cache_dir, f"{args.name}_knockout.json")

    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
    else:
        print(f"Building plan from {summary_path}")
        plan = build_plan(summary_path, args.per_layer, args.model_id)
        plan_path = out_path.replace(".json", "_plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"Plan saved to {plan_path}")

    total = len(plan)
    print(f"\nTotal: {total} nodes")

    sentences = load_sentences_from_db(db_path, args.max_sentences)
    print(f"Using {len(sentences)} sentences")

    print(f"Loading pythia-{args.model_size}...")
    from unpack.models import load_model, get_adapter
    deduped_suffix = "-deduped" if args.deduped else ""
    model_name = f"EleutherAI/pythia-{args.model_size}{deduped_suffix}"
    model, tokenizer = load_model(model_name, device=args.device,
                                  cache_dir=args.model_cache_dir,
                                  step=args.step)
    print(f"Model on {model.device}")
    hook_manager = get_adapter(model)
    hook_manager.register_hooks(model)
    num_layers = hook_manager.get_num_layers()

    print("Baseline...")
    base_ppl, base_loss, base_tokens = compute_perplexity_baseline(
        model, tokenizer, hook_manager, sentences, batch_size=args.batch_size)
    print(f"  ppl={base_ppl:.2f}  loss={base_loss:.4f}  tokens={base_tokens}")

    results = {
        "baseline": {"ppl": base_ppl, "loss": base_loss, "tokens": base_tokens},
        "config": {"model_size": args.model_size, "step": args.step,
                   "per_layer": args.per_layer, "num_sentences": len(sentences),
                   "num_layers": num_layers},
        "knockouts": [],
    }

    # Group plan by source layer
    plan_by_layer = defaultdict(list)
    for node in plan:
        if get_source_hook(node["label"]) is not None:
            plan_by_layer[node["source_layer"]].append(node)

    done = 0
    for src in sorted(plan_by_layer.keys()):
        nodes = plan_by_layer[src]
        comp_names = [n["label"] for n in nodes]
        print(f"\n  ── Layer {src} ──")

        # Accumulate loss/tokens per component, per channel
        accum = {label: {"attn": [0.0, 0], "mlp": [0.0, 0]} for label in comp_names}

        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i:i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True).to(model.device)

            # One extract pass for all components in this layer
            comp_vecs = extract_batch_components(
                model, tokenizer, hook_manager, batch, comp_names)

            # Intervention passes
            for label in comp_names:
                for channel in ("attn", "mlp"):
                    loss, tokens = compute_perplexity_with_vec(
                        model, hook_manager,
                        inputs["input_ids"], inputs["attention_mask"],
                        comp_vecs[label], label, src, num_layers, channel)
                    accum[label][channel][0] += loss
                    accum[label][channel][1] += tokens

        for node in nodes:
            label = node["label"]
            loss_a, tok_a = accum[label]["attn"]
            loss_m, tok_m = accum[label]["mlp"]
            ppl_attn = np.exp(loss_a / tok_a) if tok_a else float("inf")
            ppl_mlp = np.exp(loss_m / tok_m) if tok_m else float("inf")
            d_attn = ppl_attn - base_ppl
            d_mlp = ppl_mlp - base_ppl

            results["knockouts"].append({
                "label": label, "source_layer": src,
                "attn_strength": node["attn_strength"],
                "mlp_strength": node["mlp_strength"],
                "delta_ppl_cut_attn": d_attn,
                "delta_ppl_cut_mlp": d_mlp,
                "ppl_cut_attn": ppl_attn,
                "ppl_cut_mlp": ppl_mlp,
            })
            done += 1
            print(f"  [{done:>3}/{total}] {label:>24s}  "
                  f"attn_str={node['attn_strength']:>7.1f}  "
                  f"mlp_str={node['mlp_strength']:>7.1f}  "
                  f"Δppl(cut_attn)={d_attn:>+8.2f}  "
                  f"Δppl(cut_mlp)={d_mlp:>+8.2f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Analysis ──
    knockouts = results["knockouts"]
    by_layer = defaultdict(list)
    for k in knockouts:
        by_layer[k["source_layer"]].append(k)

    for cut, strength_key, name in [
        ("delta_ppl_cut_attn", "attn_strength", "Cut ATTN path, predict by attn_strength"),
        ("delta_ppl_cut_attn", "mlp_strength",  "Cut ATTN path, predict by mlp_strength"),
        ("delta_ppl_cut_mlp",  "attn_strength",  "Cut MLP path, predict by attn_strength"),
        ("delta_ppl_cut_mlp",  "mlp_strength",   "Cut MLP path, predict by mlp_strength"),
    ]:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  {'layer':>5}  {'n':>3}  {'r':>7}  {'mean Δppl':>10}")
        for layer in sorted(by_layer.keys()):
            items = by_layer[layer]
            strs = [k[strength_key] for k in items]
            deltas = [k[cut] for k in items]
            r = np.corrcoef(strs, deltas)[0, 1] if len(items) > 2 else float("nan")
            print(f"  {layer:>5}  {len(items):>3}  {r:>+.4f}  {np.mean(deltas):>+10.2f}")

    print(f"\n{'='*60}")
    print(f"  Mean Δppl by layer")
    print(f"{'='*60}")
    print(f"  {'layer':>5}  {'cut_attn':>10}  {'cut_mlp':>10}")
    for layer in sorted(by_layer.keys()):
        items = by_layer[layer]
        mean_a = np.mean([k["delta_ppl_cut_attn"] for k in items])
        mean_m = np.mean([k["delta_ppl_cut_mlp"] for k in items])
        print(f"  {layer:>5}  {mean_a:>+10.2f}  {mean_m:>+10.2f}")


if __name__ == "__main__":
    main()