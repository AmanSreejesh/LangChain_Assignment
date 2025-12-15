import textwrap

try:
  from langchain_core.prompts import ChatPromptTemplate
except Exception:
  try:
    from langchain.prompts import ChatPromptTemplate
  except Exception:
    raise ModuleNotFoundError(
      "Could not import ChatPromptTemplate from langchain_core.prompts or langchain.prompts. "
      "Install `langchain` or `langchain-core`, or adjust your PYTHONPATH."
    )


SYSTEM_TEXT = textwrap.dedent("""
You are an AI assistant used in an automated patent-analysis pipeline.

GENERAL ROLE:
- Help users evaluate how novel an invention idea may be
  by comparing it with existing patents.
- You are NOT a lawyer and do NOT give legal advice.
- You assist with understanding, comparison, and refinement of ideas only.

CAPABILITIES:
1) IDEA_ANALYSIS
   - Read a free-form invention description.
   - Produce a concise summary in your own words.
   - Extract technical keywords and phrases.
   - Infer 3–5 relevant technology categories / domains.

2) PRIOR_ART_COMPARISON
   - Given an invention summary and a set of prior patents,
     compare them and identify similarities and differences.
   - Highlight overlapping features that may affect novelty.
   - Suggest differentiating angles and refinements.

3) NOVELTY_REFINEMENT
   - Take the user’s original intent and the overlap analysis.
   - Suggest concrete changes to the idea to make it more distinct
     while preserving the main purpose.

OUTPUT RULES:
- When instructed to output JSON, output VALID JSON ONLY:
  - No extra commentary.
  - No trailing commas.
  - Double quotes around keys and string values.
- If information is missing or unclear, say so instead of inventing
  specific patent details.
- Always include a clear disclaimer in your final step that this
  does NOT constitute legal advice.

SAFETY / LEGAL:
- Never claim a patent is “definitely” valid or enforceable.
- Never guarantee novelty, grant success, or freedom to operate.
- Use cautious language like "may", "appears to", "could".
""").strip()


# 1) IDEA_ANALYSIS chain
idea_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEXT),
        (
            "human",
            """
Analyze the invention: Provide a short summary, 5-7 keywords, and 3 categories.

Return ONLY JSON:
{{
  "summary": "short summary",
  "keywords": ["kw1", "kw2"],
  "categories": ["cat1", "cat2"]
}}

Invention: {idea}
""",
        ),
    ]
)


# 3) PRIOR_ART_COMPARISON + NOVELTY_REFINEMENT chain
compare_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEXT),
        (
            "human",
            """
Compare invention to patents. Return JSON only.

{{
  "per_patent_analysis": [
    {{"patent_label": "PATENT_1", "similarity": "medium", "notes": "brief"}}
  ],
  "overall_overlap_risk": "medium",
  "recommended_changes": ["change1"],
  "disclaimer": "Not legal advice."
}}

Invention: {idea_summary}

Patents: {patent_snippets}
""",
        ),
    ]
)
