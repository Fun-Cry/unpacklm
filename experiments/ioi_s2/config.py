"""Experiment configuration for IOI S2 three-way comparison."""

ALL_CONFIGS = ["k_only_weighted", "k_only_l2", "k_only_aligned", "kqv_weighted", "kqv_l2", "kqv_aligned"]
BEST_CONFIG = ["kqv_aligned"]

MODELS = {
    "gpt2": {
        "name": "gpt2",
        "configs": ALL_CONFIGS,
        "device": "cuda:0",
        "cache_dir": "/local/s4283341",
    },
    "pythia-160m": {
        "name": "EleutherAI/pythia-160m-deduped",
        "configs": BEST_CONFIG,
        "device": "cuda:0",
        "cache_dir": "/local/s4283341",
    },
    "pythia-410m": {
        "name": "EleutherAI/pythia-410m-deduped",
        "configs": BEST_CONFIG,
        "device": "cuda:0",
        "cache_dir": "/local/s4283341",
    },
    "pythia-1.4b": {
        "name": "EleutherAI/pythia-1.4b-deduped",
        "configs": BEST_CONFIG,
        "device": "cuda:0",
        "cache_dir": "/local/s4283341",
    },
    "pythia-2.8b": {
        "name": "EleutherAI/pythia-2.8b-deduped",
        "configs": BEST_CONFIG,
        "device": "cuda:1",
        "cache_dir": ".",
    },
    "pythia-6.9b": {
        "name": "EleutherAI/pythia-6.9b-deduped",
        "configs": BEST_CONFIG,
        "device": "cuda:1",
        "cache_dir": ".",
    },
}

DEFAULT_N_PROMPTS = 100
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = "results/ioi_s2"