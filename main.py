"""
FastAPI server application and user-facing endpoints.

Exposes exactly two CORS-safe endpoints for the Emergent UI layout:
  - POST /api/sync  -> triggers health_agent.py's ingestion pipeline in the background
  - POST /api/chat  -> grounded chat over vectors stored in MongoDB Atlas

Requires the Atlas Search vector index (see AGENTS.md's "Database Security &
Manual Index Alert") to already exist under the name VECTOR_INDEX_NAME below
before /api/chat will return results.
"""

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from health_agent import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    get_mongo_collection,
    get_openrouter_client,
    run_health_agent_pipeline,
    truncate_embedding,
)

load_dotenv()

app = FastAPI(title="pethealthsystem")

# TODO: restrict allow_origins to the actual Emergent deployment URL once known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

CHAT_MODEL = "meta-llama/llama-3-8b-instruct:free"
# Must match the name given to the Atlas Search index you create manually,
# per AGENTS.md's "Database Security & Manual Index Alert" section.
VECTOR_INDEX_NAME = "vector_index"
TOP_K_CHUNKS = 3


class ChatRequest(BaseModel):
    user_question: str


@app.post("/api/sync")
async def sync(background_tasks: BackgroundTasks):
    """
    Trigger the health_agent.py ingestion sequence (Playwright -> LlamaParse ->
    LangChain chunking -> OpenRouter embeddings -> MongoDB upsert) as a
    background task and return immediately so the UI stays responsive.
    """
    background_tasks.add_task(run_health_agent_pipeline)
    return {"status": "sync_started"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Embed the raw user_question via OpenRouter's nvidia/nemotron-3-embed-1b:free,
    run MongoDB Atlas $vectorSearch for the top 3 matching chunks, and feed
    them into meta-llama/llama-3-8b-instruct:free for a grounded, hallucination-
    resistant answer.
    """
    client = get_openrouter_client()

    # Sent verbatim -- no stop-word stripping or cleaning of the raw question.
    # encoding_format="float" avoids OpenRouter's base64 proxy issue (see
    # health_agent.embed_and_upsert_chunks for details).
    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=request.user_question, encoding_format="float"
    )
    raw_query_vector = embedding_response.data[0].embedding
    # Model natively returns 2048 dims; must apply the identical truncation
    # used at ingestion time or the query vector won't be comparable to the
    # stored vectors in $vectorSearch.
    query_vector = truncate_embedding(raw_query_vector, EMBEDDING_DIMENSIONS)

    collection = get_mongo_collection()
    try:
        results = list(
            collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_INDEX_NAME,
                            "path": "vector_embedding",
                            "queryVector": query_vector,
                            "numCandidates": 100,
                            "limit": TOP_K_CHUNKS,
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "text_content": 1,
                            "pet_id": 1,
                            "extracted_at": 1,
                            "score": {"$meta": "vectorSearchScore"},
                        }
                    },
                ]
            )
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "MongoDB $vectorSearch failed. Confirm the Atlas Search index "
                f"'{VECTOR_INDEX_NAME}' has been created per AGENTS.md before "
                f"calling this endpoint. Original error: {error}"
            ),
        ) from error

    context_blocks = "\n\n---\n\n".join(result["text_content"] for result in results)

    system_prompt = (
        "You are a grounded veterinary record assistant. Answer the user's "
        "question using ONLY the medical record context blocks below. If the "
        "context does not contain the answer, say so explicitly instead of "
        "guessing or relying on outside knowledge. Do not hallucinate facts, "
        "doses, dates, or diagnoses that are not present in the context.\n\n"
        f"CONTEXT:\n{context_blocks}"
    )

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_question},
        ],
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "pet_id": result.get("pet_id"),
                "extracted_at": result.get("extracted_at"),
                "score": result.get("score"),
            }
            for result in results
        ],
    }
