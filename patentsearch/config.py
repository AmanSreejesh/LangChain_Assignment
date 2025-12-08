import os
from dotenv import load_dotenv

load_dotenv()

PATENTSEARCH_API_KEY = os.getenv("PATENTSEARCH_API_KEY")
if not PATENTSEARCH_API_KEY:
    raise RuntimeError(
        "PATENTSEARCH_API_KEY not set. Put it in a .env file or environment variable."
    )

# PatentSearch patents endpoint
PATENTSEARCH_PATENT_ENDPOINT = "https://search.patentsview.org/api/v1/patent/"
