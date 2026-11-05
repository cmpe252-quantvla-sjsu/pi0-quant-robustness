"""``python -m quantvla.cli``: VLAQuantBench's command line with our ``libero_plus`` benchmark added.

Same commands and options as ``vqb`` (run, inspect, tasks, summarize, ...). Run it from the
repo root inside the conda environment, e.g.

    python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 --benchmark libero_plus --suite libero_spatial \\
        --preset BASELINE --episodes 20 --bench-kwargs condition=clean \\
        --out results/E1/clean/E1a-BF16-spatial.jsonl
"""
from __future__ import annotations

from vlaquantbench import registry

registry.BENCHMARKS.setdefault("libero_plus", "quantvla.libero_plus:LiberoPlusRunner")


def main(argv: list[str] | None = None) -> None:
    from vlaquantbench.cli import main as vqb_main

    vqb_main(argv)


if __name__ == "__main__":
    main()
