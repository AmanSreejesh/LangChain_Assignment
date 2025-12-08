import json
import textwrap
from typing import Dict, Any, List

import requests

from .config import PATENTSEARCH_API_KEY, PATENTSEARCH_PATENT_ENDPOINT


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
