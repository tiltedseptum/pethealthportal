"""
One-off cleanup: deletes the stale test documents inserted before the
hash-based dedup upsert was added to embed_and_upsert_chunks().

Those 29 documents used MongoDB's auto-generated ObjectIds as _id, so the
new content-hash _id scheme won't recognize or overwrite them -- they'd
just sit alongside the new hash-keyed documents as orphaned duplicates.
This script removes all documents for PET_ID (from .env) so you can re-run
test_step_d.py once and get a clean, fully hash-keyed collection.

This is a real delete against your live MongoDB Atlas cluster. Only run
this if you're sure -- it removes ALL documents currently stored for this
pet_id, not just the specific 29 from the earlier test.

Run from inside pethealthsystem/ with your venv active:
    python cleanup_test_data.py
"""

import os

from dotenv import load_dotenv

from health_agent import get_mongo_collection

load_dotenv()


def main():
    pet_id = os.getenv("PET_ID")
    if not pet_id:
        raise RuntimeError("PET_ID is not set in .env")

    collection = get_mongo_collection()
    existing_count = collection.count_documents({"pet_id": pet_id})
    print(f"⚠️  About to delete {existing_count} document(s) for pet_id={pet_id!r} "
          f"from {collection.full_name}")

    confirm = input("Type 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
        print("Aborted -- no documents deleted.")
        return

    result = collection.delete_many({"pet_id": pet_id})
    print(f"✅ Deleted {result.deleted_count} document(s). Collection is now clean "
          f"for pet_id={pet_id!r}.")
    print("You can now re-run test_step_d.py to repopulate with hash-keyed documents.")


if __name__ == "__main__":
    main()
