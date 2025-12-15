import os
from dotenv import load_dotenv

load_dotenv()

PATENTSEARCH_API_KEY = os.getenv("PATENTSEARCH_API_KEY")
# API key is required for Lens.org API (get from https://www.lens.org/lens/user/subscriptions#scholar)

# PatentSearch patents endpoint
PATENTSEARCH_PATENT_ENDPOINT = "https://search.patentsview.org/api/v1/patent/"
