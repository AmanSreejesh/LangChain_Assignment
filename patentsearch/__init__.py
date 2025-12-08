"""Package entry for patentsearch pipeline.

Exports a simple `run` entry so `patentsearch_pipeline.py` can remain a thin wrapper.
"""
from .pipeline import run_patentsearch_pipeline, pretty_print_result

__all__ = ["run_patentsearch_pipeline", "pretty_print_result"]
