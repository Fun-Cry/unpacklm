"""Entry point: `python -m experiments.circuit_discovery <folder>`.

Currently routes to discover. Once validate/summarize are added, this
will dispatch to subcommands.
"""

from .discover import main

if __name__ == "__main__":
    main()
