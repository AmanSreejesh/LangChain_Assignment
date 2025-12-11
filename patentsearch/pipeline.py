import json
import os
from typing import Dict, Any, List

from .llm import get_llm
from .prompts import idea_prompt, compare_prompt
from .api import search_patentsearch, format_patents_for_llm


def run_patentsearch_pipeline(idea_text: str) -> Dict[str, Any]:
    # If mock mode enabled, return deterministic fixtures so the pipeline
    # can be exercised without Ollama or network access.
    use_mock = os.getenv("PATENTSEARCH_USE_MOCK", "0").lower() in ("1", "true", "yes")
    if use_mock:
        # Simple mock idea analysis
        idea_info = {
            "summary": (idea_text.strip()[:400] or "(no idea provided)"),
            "keywords": [w.strip(".,") for w in idea_text.split()[:8]],
            "categories": ["IoT", "Environmental Monitoring", "Wearables"],
        }

        # Mock patents returned by PatentSearch
        patents = [
            {
                "patent_id": "US-MOCK-0001",
                "title": "Wearable air-quality monitor with feedback",
                "abstract": "A wearable device that measures air quality and provides feedback to the wearer.",
                "date": "2022-05-01",
            },
            {
                "patent_id": "US-MOCK-0002",
                "title": "Adaptive breathing guidance system",
                "abstract": "A system that suggests breathing exercises based on sensor input.",
                "date": "2020-10-20",
            },
        ]

        patent_snippets = format_patents_for_llm(patents)

        # Mock comparison info
        comp_info = {
            "per_patent_analysis": [
                {
                    "patent_label": "PATENT_1",
                    "patent_id": "US-MOCK-0001",
                    "similarity": "medium",
                    "overlapping_features": ["wearable sensor", "air quality measurement"],
                    "differentiating_features": ["robust casing for harsh conditions"],
                    "notes": "PATENT_1 shares sensing features but differs in durability and deployment context.",
                },
                {
                    "patent_label": "PATENT_2",
                    "patent_id": "US-MOCK-0002",
                    "similarity": "low",
                    "overlapping_features": ["breathing guidance"],
                    "differentiating_features": ["hotspot connectivity features"],
                },
            ],
            "overall_overlap_risk": "medium",
            "recommended_changes": [
                "Emphasize ruggedized power and sealed enclosure",
                "Add low-bandwidth mesh connectivity for remote areas",
            ],
            "disclaimer": "This mock comparison is for demonstration only and is NOT legal advice.",
        }

        return {"idea_analysis": idea_info, "comparison": comp_info, "similar_patents": patents}

    # Real mode: call the LLM and online PatentSearch
    llm = get_llm()

    def _extract_json(text: str) -> dict:
        """Try to extract JSON from model text reliably.

        The model sometimes returns fenced code blocks or extra commentary.
        This helper attempts JSON loads directly, then strips Markdown
        fences (```), then falls back to extracting the first {...} pair.
        """
        s = text.strip()
        # direct parse
        try:
            return json.loads(s)
        except Exception:
            pass

        # remove fenced code block markers
        if s.startswith("```") and "```" in s[3:]:
            # remove first and last fence
            try:
                inner = s.split("```", 2)[2]
                inner = inner.rsplit("```", 1)[0]
                return json.loads(inner.strip())
            except Exception:
                pass

        # find first { and last }
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = s[first : last + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # give up with original content in the error
        raise ValueError(f"Failed to parse JSON from model response: {text}")

    # 1) IDEA ANALYSIS
    print("Running LLM: idea analysis...")
    idea_chain = idea_prompt | llm
    idea_res = idea_chain.invoke({"idea": idea_text})

    idea_info = _extract_json(idea_res.content)

    # 2) ONLINE PATENT SEARCH
    # Get more patents than needed, then filter for actual similarity
    raw_patents = search_patentsearch(
        summary=idea_info["summary"],
        keywords=idea_info["keywords"],
        size=15,
    )

    # Compute keyword overlap for each patent (case-insensitive, in title or abstract)
    def keyword_overlap(patent, keywords):
        title = (patent.get("title") or "").lower()
        abstract = (patent.get("abstract") or "").lower()
        overlap = 0
        for kw in keywords:
            kw = kw.lower()
            if kw and (kw in title or kw in abstract):
                overlap += 1
        return overlap

    # Only include patents with at least one keyword overlap
    filtered_patents = [
        p for p in raw_patents if keyword_overlap(p, idea_info["keywords"]) > 0
    ]

    # If none, fallback to the top 1-2 by default (so user sees something)
    if not filtered_patents and raw_patents:
        filtered_patents = raw_patents[:2]

    if not filtered_patents:
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
            "similar_patents": [],
        }

    patent_snippets = format_patents_for_llm(filtered_patents)

    # 3) PRIOR ART COMPARISON + NOVELTY SUGGESTIONS
    print("Running LLM: prior-art comparison + novelty suggestions...")
    comparison_chain = compare_prompt | llm
    comp_res = comparison_chain.invoke({
        "idea_summary": idea_info["summary"],
        "patent_snippets": patent_snippets,
    })

    comp_info = _extract_json(comp_res.content)

    # Map PATENT_n labels to patent_id if missing
    label_to_id = {f"PATENT_{i+1}": p["patent_id"] for i, p in enumerate(filtered_patents)}
    for entry in comp_info.get("per_patent_analysis", []):
        label = entry.get("patent_label")
        if label and not entry.get("patent_id"):
            entry["patent_id"] = label_to_id.get(label, "")

    return {
        "idea_analysis": idea_info,
        "comparison": comp_info,
        "similar_patents": filtered_patents,
    }


def pretty_print_result(result: Dict[str, Any]):
    print("\n=== IDEA ANALYSIS ===")
    print("Summary:\n", result["idea_analysis"]["summary"])
    print("\nKeywords:", ", ".join(result["idea_analysis"]["keywords"]))
    print("Categories:", ", ".join(result["idea_analysis"]["categories"]))

    print("\n=== SIMILAR PATENTS (FOR REFERENCE) ===")
    patents = result.get("similar_patents", [])
    if patents:
        for idx, p in enumerate(patents, start=1):
            print(f"\nPATENT_{idx}:")
            print(f"  ID: {p.get('patent_id', 'N/A')}")
            print(f"  Title: {p.get('title', 'N/A')}")
            print(f"  Date: {p.get('date', 'N/A')}")
            abstract = p.get('abstract', '')
            if abstract:
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                print(f"  Abstract: {abstract}")
    else:
        print("No similar patents found.")

    print("\n=== PRIOR ART COMPARISON ===")
    comp = result["comparison"]
    print("Overall overlap risk:", comp.get("overall_overlap_risk"))

    print("\nRecommended changes:")
    for ch in comp.get("recommended_changes", []):
        print(" -", ch)

    print("\nDisclaimer:")
    print(comp.get("disclaimer", ""))
