"""
Diagnostic: read-only check of what's actually in MongoDB Atlas for PET_ID.

Does not call OpenRouter, LlamaParse, or Playwright -- just queries Mongo
directly to get ground truth on document count, since the Atlas UI's
"Documents (Estimated)" figure on the Search Index page can lag or round.

Run from inside pethealthsystem/ with your venv active:
    python check_mongo_state.py
"""

import os
from collections import Counter

from dotenv import load_dotenv

from health_agent import get_mongo_collection

load_dotenv()


def main():
    pet_id = os.getenv("PET_ID")
    if not pet_id:
        raise RuntimeError("PET_ID is not set in .env")

    collection = get_mongo_collection()
    total = collection.count_documents({"pet_id": pet_id})
    print(f"📊 Ground-truth document count for pet_id={pet_id!r}: {total}")

    docs = list(
        collection.find({"pet_id": pet_id}, {"chunk_index": 1, "extracted_at": 1, "text_content": 1})
    )

    # Group by extracted_at to see how many distinct sync runs are represented.
    by_sync = Counter(doc["extracted_at"] for doc in docs)
    print(f"\n🕒 Documents grouped by extracted_at (one group per sync run that "
          f"actually wrote new/changed content):")
    for extracted_at, count in sorted(by_sync.items()):
        print(f"   {extracted_at} -> {count} documents")

    # Group by chunk_index to see if the "same" chunk position has multiple
    # distinct text_content values across sync runs (would confirm the
    # underlying parsed markdown differed between runs, not just metadata).
    by_index = {}
    for doc in docs:
        by_index.setdefault(doc["chunk_index"], set()).add(doc["text_content"])

    diverged = {idx: texts for idx, texts in by_index.items() if len(texts) > 1}
    print(f"\n🔎 chunk_index values with more than one distinct text_content "
          f"across stored documents: {len(diverged)} out of {len(by_index)} indices")
    if diverged:
        sample_idx = sorted(diverged)[0]
        sample_texts = list(diverged[sample_idx])
        print(f"\n   Example: chunk_index={sample_idx} has {len(sample_texts)} distinct versions.")
        print(f"   Version A preview: {sample_texts[0][:150]!r}")
        print(f"   Version B preview: {sample_texts[1][:150]!r}")


if __name__ == "__main__":
    main()
