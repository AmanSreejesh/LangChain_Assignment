# patentsearch_pipeline.py

"""Thin entrypoint wrapper kept for compatibility with prior usage.

This file delegates implementation to the `patentsearch` package so the
codebase has a clearer structure (CLI, pipeline, API, prompts, LLM).

Running `python patentsearch_pipeline.py` preserves the original CLI
behaviour.
"""

from patentsearch.cli import main


if __name__ == "__main__":
    main()
