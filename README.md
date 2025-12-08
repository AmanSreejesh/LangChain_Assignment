# LangChain_Assignment

**Project:** Patent-analysis pipeline using an Ollama model and PatentSearch

**Status:** Refactored — single-file pipeline split into a `patentsearch/` package.

**What this repository contains**
- Top-level entrypoint: `patentsearch_pipeline.py` — thin wrapper that runs the CLI.
- Package: `patentsearch/` — modules for config, API calls, prompts, LLM factory, pipeline, and CLI.
- `Modelfile` — your Ollama model definition used to create the `patent-analyst` model.

**Quick start**
1. Install dependencies (example):

```bash
python -m pip install -r requirements.txt
```

2. Provide your PatentSearch API key via environment variable or `.env` file at the repo root:

```bash
export PATENTSEARCH_API_KEY="your_api_key_here"
# or add a file named .env containing:
# PATENTSEARCH_API_KEY=your_api_key_here
```

3. Ensure Ollama is running and the `patent-analyst` model exists (if you intend to run the LLM parts):

```bash
# create the model locally (if using Modelfile):
ollama create patent-analyst -f Modelfile
# start ollama server if not running (follow your Ollama setup)
```

4. Run the CLI (reads idea text from stdin):

```bash
python patentsearch_pipeline.py
# Then paste your idea and end with Ctrl+D (Linux/Mac) or Ctrl+Z (Windows)
```

Notes
- The project currently raises a `RuntimeError` at startup if `PATENTSEARCH_API_KEY` is not set — this preserves the original behavior. If you want a mock/fallback mode so the pipeline can run without a key (useful while waiting for the API key), I can add `PATENTSEARCH_USE_MOCK=1` and a small fixture.
- The code expects an Ollama server and your `patent-analyst` model for the LLM steps. You can still test parts of the repository by mocking the LLM or using a local test model.

Project layout

- `patentsearch/`
	- `config.py` — env loading and constants (validates `PATENTSEARCH_API_KEY`)
	- `api.py` — PatentSearch query and HTTP helpers
	- `prompts.py` — system prompt and templates
	- `llm.py` — `get_llm()` factory (uses `ChatOllama`)
	- `pipeline.py` — main pipeline logic and pretty-printer
	- `cli.py` — command-line interface wrapper

If you'd like, I can:
- Add a `requirements.txt` based on imports.
- Add a mock mode so the repo runs end-to-end without the real API key.
- Add a simple test harness or example input file for quick experimentation.

---
_If you want me to change the README wording or add extra sections (tests, CI, examples), tell me what to include and I'll update it._
