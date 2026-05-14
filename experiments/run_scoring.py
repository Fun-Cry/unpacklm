"""
Scoring pipeline: compute per-component importance scores on a corpus.

Processes sentences through the model, computes attention-side and
MLP-side importance scores per component, and saves results to
HDF5 + SQLite for downstream knockout experiments.

Usage:
    python -m experiments.run_scoring \
        --model-family pythia --model-size 160m --step 143000 --deduped \
        --cache-dir /data/models --name pile_160m \
        --dataset pile --limit 1000 --device cuda:0

    python -m experiments.run_scoring \
        --model-family gpt2 --model-size small \
        --cache-dir /data/models --name pile_gpt2 \
        --dataset pile --limit 1000 --device cuda:0
"""

import argparse
import os

from unpack.models import load_model as _unpack_load_model, get_adapter


def load_dataset(args):
    """Load dataset based on --dataset flag."""
    from utils.load_data import ListAdapter

    if args.dataset == "pile":
        from utils.load_data import load_pile_sentences
        ds = load_pile_sentences(
            target=args.limit or 10000, seed=args.seed,
            cache_dir=args.cache_dir)
    elif args.dataset == "ioi":
        from utils.load_data import load_ioi_dataset
        ds = load_ioi_dataset(target=args.limit or 100, seed=args.seed)
    elif args.dataset == "file":
        if not args.sentences:
            raise ValueError("--sentences required when --dataset file")
        with open(args.sentences) as f:
            sentences = [line.strip() for line in f if line.strip()]
        ds = ListAdapter(sentences)
        print(f"Loaded {len(ds)} sentences from {args.sentences}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return ds


def load_model(args):
    """Load model + tokenizer + adapter using unpack."""
    family = args.model_family

    if family == "pythia":
        deduped_suffix = "-deduped" if args.deduped else ""
        model_name = f"EleutherAI/pythia-{args.model_size}{deduped_suffix}"
        load_kwargs = {"step": args.step}
    elif family == "gpt2":
        name_map = {"small": "gpt2", "medium": "gpt2-medium",
                    "large": "gpt2-large", "xl": "gpt2-xl"}
        model_name = name_map.get(args.model_size, args.model_size)
        load_kwargs = {}
    else:
        raise ValueError(f"Unknown model family: {family}")

    print(f"Loading {model_name}...")
    model, tokenizer = _unpack_load_model(
        model_name, device=args.device, cache_dir=args.cache_dir,
        **load_kwargs)

    adapter = get_adapter(model)
    return model, tokenizer, adapter


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-component importance scores on a corpus")
    parser.add_argument("--model-family", default="pythia",
                        choices=["pythia", "gpt2"])
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--deduped", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=".")
    parser.add_argument("--name", required=True,
                        help="Run name; determines output file paths "
                             "({cache-dir}/{name}.db, {name}.h5, etc.)")
    parser.add_argument("--dataset", default="pile",
                        choices=["pile", "ioi", "file"])
    parser.add_argument("--sentences", default=None,
                        help="Path to .txt file (used with --dataset file)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--comp-batch-size", type=int, default=16)
    args = parser.parse_args()

    # Set derived paths expected by ExperimentRunner
    args.db_path = os.path.join(args.cache_dir, f"{args.name}.db")
    args.h5_path = os.path.join(args.cache_dir, f"{args.name}.h5")
    args.state_file = f"{args.name}.txt"

    print(f"Loading dataset: {args.dataset} (limit: {args.limit})...")
    dataset = load_dataset(args)

    model, tokenizer, adapter = load_model(args)

    from experiments.scoring.runner import ExperimentRunner
    from experiments.scoring.processors import (
        HeadPreferenceProcessor, MLPIntermediateProcessor,
    )

    runner = ExperimentRunner(args, dataset, model, tokenizer, adapter)

    runner.add_processor(HeadPreferenceProcessor(
        storage=runner.storage,
        model_id=runner.model_id,
    ))
    runner.add_processor(MLPIntermediateProcessor(
        storage=runner.storage,
        model_id=runner.model_id,
    ))

    runner.run()

    if os.path.exists(args.state_file):
        with open(args.state_file) as f:
            last_idx = int(f.read().strip())
        if last_idx >= len(dataset) - 1:
            os.remove(args.state_file)
            print("Scoring complete. State file removed.")
        else:
            print(f"Scoring incomplete ({last_idx + 1}/{len(dataset)}). "
                  f"State file kept for resume.")

    print("Done.")


if __name__ == "__main__":
    main()
