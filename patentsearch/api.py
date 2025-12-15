import json
import textwrap
from typing import Dict, Any, List

import requests

from .config import PATENTSEARCH_API_KEY, PATENTSEARCH_PATENT_ENDPOINT


def build_patentsearch_query(summary: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Build a PatentSearch 'q' JSON object.
    Restrict to patents from 2020 onward and search title/abstract.
    """
    # Build OR conditions for each keyword in title and abstract
    title_conditions = [{"_text_phrase": {"patent_title": kw.lower()}} for kw in keywords if kw]
    abstract_conditions = [{"_text_phrase": {"patent_abstract": kw.lower()}} for kw in keywords if kw]
    
    # If no keywords, fall back to summary
    if not title_conditions:
        title_conditions = [{"_text_any": {"patent_title": summary}}]
        abstract_conditions = [{"_text_any": {"patent_abstract": summary}}]

    q_obj = {
        "_and": [
            {"_gte": {"patent_date": "2020-01-01"}},
            {
                "_or": title_conditions + abstract_conditions
            },
        ]
    }
    return q_obj


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

    headers = {}
    if PATENTSEARCH_API_KEY:
        headers["X-Api-Key"] = PATENTSEARCH_API_KEY

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
