import textwrap
from langchain_core.prompts import ChatPromptTemplate


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
You are in mode: IDEA_ANALYSIS.

Read the user invention description below and perform:
1) A concise neutral summary (max 200 words).
2) A list of 5–15 technical keywords/phrases.
3) 3–5 high-level technology categories/domains.

Return ONLY valid JSON with this exact schema:
{{
  "summary": "<string>",
  "keywords": ["<string>", ...],
  "categories": ["<string>", ...]
}}

USER_IDEA:
{idea}
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
You are in mode: PRIOR_ART_COMPARISON.

You will:
- Compare the user's invention with provided prior patents.
- Highlight overlapping and differentiating features.
- Suggest changes to improve distinctiveness.
- This is NOT legal advice and you must say so in the disclaimer.

USER_INVENTION_SUMMARY:
{idea_summary}

PRIOR_PATENTS_TEXT:
{patent_snippets}

Return ONLY valid JSON with this schema:
{{
  "per_patent_analysis": [
    {{
      "patent_label": "<e.g. PATENT_1>",
      "patent_id": "<string>",
      "similarity": "<low|medium|high>",
      "overlapping_features": ["<string>", ...],
      "differentiating_features": ["<string>", ...],
      "notes": "<string>"
    }}
  ],
  "overall_overlap_risk": "<low|medium|high>",
  "recommended_changes": ["<string>", ...],
  "disclaimer": "<string>"
}}
""",
        ),
    ]
)
