import json
from typing import Dict, Any, List

from .llm import get_llm
from .prompts import idea_prompt, compare_prompt
from .api import search_patentsearch, format_patents_for_llm


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
