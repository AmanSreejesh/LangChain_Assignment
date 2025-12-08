# patentsearch_pipeline.py

import json
import os
import textwrap
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------
# ENV + CONSTANTS
# -------------------------------------------------------

load_dotenv()

PATENTSEARCH_API_KEY = os.getenv("PATENTSEARCH_API_KEY")
if not PATENTSEARCH_API_KEY:
    raise RuntimeError(
        "PATENTSEARCH_API_KEY not set. Put it in a .env file or environment variable."
    )

# PatentSearch patents endpoint
PATENTSEARCH_PATENT_ENDPOINT = "https://search.patentsview.org/api/v1/patent/"

# -------------------------------------------------------
# LLM SETUP (YOUR MODEFILE MODEL)
# -------------------------------------------------------

def get_llm():
    """
    Uses your custom Modelfile model called 'patent-analyst'.

    Make sure you have run on a machine that *does* have the Ollama CLI:
        ollama create patent-analyst -f Modelfile

    And that an Ollama server is running and reachable (OLLAMA_HOST env if remote).
    """
    return ChatOllama(
        model="patent-analyst",   # <- THIS USES YOUR MODEFILE MODEL
        temperature=0.4,
    )

# -------------------------------------------------------
# PROMPTS / CHAINS
# -------------------------------------------------------

SYSTEM_TEXT = """
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
""".strip()

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
{
  "summary": "<string>",
  "keywords": ["<string>", ...],
  "categories": ["<string>", ...]
}

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
{
  "per_patent_analysis": [
    {
      "patent_label": "<e.g. PATENT_1>",
      "patent_id": "<string>",
      "similarity": "<low|medium|high>",
      "overlapping_features": ["<string>", ...],
      "differentiating_features": ["<string>", ...],
      "notes": "<string>"
    }
  ],
  "overall_overlap_risk": "<low|medium|high>",
  "recommended_changes": ["<string>", ...],
  "disclaimer": "<string>"
}
""",
        ),
    ]
)

# -------------------------------------------------------
# PATENTSEARCH ONLINE CALLS
# -------------------------------------------------------

def build_patentsearch_query(summary: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Build a PatentSearch 'q' JSON object.
    Restrict to patents from 2010 onward and search title/abstract.
    """
    keyword_phrase = " ".join(keywords) if keywords else summary

    q = {
        "_and": [
            {"_gte": {"patent_date": "2010-01-01"}},
            {
                "_or": [
                    {"_text_any": {"patent_title": keyword_phrase}},
                    {"_text_any": {"patent_abstract": keyword_phrase}},
                ]
            },
        ]
    }
    return q


def search_patentsearch(summary: str, keywords: List[str], size: int = 5) -> List[Dict[str, Any]]:
    """
    Call the PatentSearch /api/v1/patent/ endpoint.
    Returns a list of { patent_id, title, abstract, date } dicts.
    """
    q_obj = build_patentsearch_query(summary, keywords)

    f_obj = [
        "patent_id",
        "patent_title",
        "patent_abstract",
        "patent_date",
    ]

    o_obj = {
        "size": size
    }

    params = {
        "q": json.dumps(q_obj),
        "f": json.dumps(f_obj),
        "o": json.dumps(o_obj),
    }

    headers = {
        "X-Api-Key": PATENTSEARCH_API_KEY,
    }

    resp = requests.get(PATENTSEARCH_PATENT_ENDPOINT, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"PatentSearch API error: {data}")

    patents = data.get("patents", [])
    results = []
    for p in patents:
        results.append(
            {
                "patent_id": p.get("patent_id", ""),
                "title": p.get("patent_title", ""),
                "abstract": p.get("patent_abstract", ""),
                "date": p.get("patent_date", ""),
            }
        )
    return results


def format_patents_for_llm(patents: List[Dict[str, Any]]) -> str:
    """
    Turn patents into a text block for the LLM.
    Labels them PATENT_1, PATENT_2, ...
    """
    chunks = []
    for idx, p in enumerate(patents, start=1):
        label = f"PATENT_{idx}"
        pid = p.get("patent_id", "unknown")
        title = p.get("title", "").strip()
        abstract = p.get("abstract", "").strip()
        date = p.get("date", "")

        chunk = textwrap.dedent(
            f"""
            {label} (patent_id={pid}, date={date})
            TITLE: {title}
            ABSTRACT: {abstract}
            """
        ).strip()
        chunks.append(chunk)

    return "\n\n".join(chunks)

# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def run_patentsearch_pipeline(idea_text: str) -> Dict[str, Any]:
    llm = get_llm()

    # 1) IDEA ANALYSIS
    idea_chain = idea_prompt | llm
    idea_res = idea_chain.invoke({"idea": idea_text})

    try:
        idea_info = json.loads(idea_res.content)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse idea analysis JSON: {idea_res.content}")

    # 2) ONLINE PATENT SEARCH
    patents = search_patentsearch(
        summary=idea_info["summary"],
        keywords=idea_info["keywords"],
        size=5,
    )

    if not patents:
        return {
            "idea_analysis": idea_info,
            "comparison": {
                "per_patent_analysis": [],
                "overall_overlap_risk": "low",
                "recommended_changes": [],
                "disclaimer": (
                    "No relevant prior patents were retrieved with PatentSearch for this query. "
                    "This does NOT guarantee novelty or patentability. Consult a qualified patent attorney."
                ),
            },
        }

    patent_snippets = format_patents_for_llm(patents)

    # 3) PRIOR ART COMPARISON + NOVELTY SUGGESTIONS
    comparison_chain = compare_prompt | llm
    comp_res = comparison_chain.invoke({
        "idea_summary": idea_info["summary"],
        "patent_snippets": patent_snippets,
    })

    try:
        comp_info = json.loads(comp_res.content)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse comparison JSON: {comp_res.content}")

    # Map PATENT_n labels to patent_id if missing
    label_to_id = {f"PATENT_{i+1}": p["patent_id"] for i, p in enumerate(patents)}
    for entry in comp_info.get("per_patent_analysis", []):
        label = entry.get("patent_label")
        if label and not entry.get("patent_id"):
            entry["patent_id"] = label_to_id.get(label, "")

    return {
        "idea_analysis": idea_info,
        "comparison": comp_info,
    }

# -------------------------------------------------------
# SIMPLE CLI
# -------------------------------------------------------

def pretty_print_result(result: Dict[str, Any]):
    print("\n=== IDEA ANALYSIS ===")
    print("Summary:\n", result["idea_analysis"]["summary"])
    print("\nKeywords:", ", ".join(result["idea_analysis"]["keywords"]))
    print("Categories:", ", ".join(result["idea_analysis"]["categories"]))

    print("\n=== PRIOR ART COMPARISON ===")
    comp = result["comparison"]
    print("Overall overlap risk:", comp.get("overall_overlap_risk"))

    print("\nRecommended changes:")
    for ch in comp.get("recommended_changes", []):
        print(" -", ch)

    print("\nDisclaimer:")
    print(comp.get("disclaimer", ""))


if __name__ == "__main__":
    print("Enter invention description (end with Ctrl+D / Ctrl+Z):")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    idea = "\n".join(lines).strip()
    if not idea:
        print("No idea text provided.")
    else:
        result = run_patentsearch_pipeline(idea)
        pretty_print_result(result)
        # If you need the raw JSON:
        # print(json.dumps(result, indent=2))
