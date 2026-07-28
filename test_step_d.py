"""
Standalone diagnostic for Step D only: embeddings + Mongo upsert.

Does NOT touch Steps A or B. Reads the existing ./parsed_output.md, re-runs
the real chunk_markdown() from Step C on it, then calls the real
embed_and_upsert_chunks() -- this makes LIVE calls to OpenRouter
(nvidia/nemotron-3-embed-1b:free) and LIVE writes to your MongoDB Atlas
cluster (pethealthsystem.medical_record_chunks).

NOTE: this is a real write, not a dry run. Since dedup hasn't been added
yet, running this multiple times will insert duplicate documents for
unchanged chunks -- that's expected for this first test and will be
addressed by the planned hash-based upsert follow-up.

Run from inside pethealthsystem/ with your venv active:
    python test_step_d.py
"""

import asyncio

from dotenv import load_dotenv

from health_agent import chunk_markdown, embed_and_upsert_chunks, get_mongo_collection

load_dotenv()

import os

INPUT_PATH = "./parsed_output.md"


async def main():
    pet_id = os.getenv("PET_ID")
    if not pet_id:
        raise RuntimeError("PET_ID is not set in .env")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    chunks = chunk_markdown(markdown_text)
    print(f"🤖 Chunked into {len(chunks)} pieces. Embedding + upserting for pet_id={pet_id!r}...")
    print("   (this makes real calls to OpenRouter and writes to MongoDB Atlas)")

    upserted_count = await embed_and_upsert_chunks(pet_id, chunks)
    print(f"✅ embed_and_upsert_chunks() reported {upserted_count} documents upserted")

    # Sanity-check by reading back from Mongo directly.
    collection = get_mongo_collection()
    total_for_pet = collection.count_documents({"pet_id": pet_id})
    print(f"📊 Total documents now in Mongo for pet_id={pet_id!r}: {total_for_pet}")

    sample = collection.find_one({"pet_id": pet_id}, sort=[("chunk_index", 1)])
    if sample:
        vec = sample.get("vector_embedding", [])
        print(f"🔎 Sample document -> chunk_index={sample.get('chunk_index')}, "
              f"vector_embedding length={len(vec)}, "
              f"text_content preview={sample.get('text_content', '')[:100]!r}")
    else:
        print("⚠️ No document found on read-back -- something is off.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        print(f"❌ Step D failed: {error}")
        raise
