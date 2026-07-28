"""
Asynchronous ingestion pipeline: Playwright -> LlamaParse (llama-cloud SDK) -> LangChain -> MongoDB.

Sequence (per AGENTS.md):
  A. Automated download   - headless Playwright login + export flow on VetBuddy,
                             saved locally as ./latest_visit.pdf
  B. Parsing              - llama-cloud's Parse product converts the PDF to
                             structured markdown (migrated off the deprecated
                             llama-parse SDK)
  C. Chunking             - LangChain MarkdownTextSplitter (512 tokens, 50 overlap),
                             split along markdown header hierarchy
  D. Embedding + upsert   - OpenRouter nvidia/nemotron-3-embed-1b:free (1024-dim),
                             bulk upsert into MongoDB Atlas

Credentials are read exclusively from the local .env via python-dotenv. This
file never hardcodes secrets and Claude never types/handles the VetBuddy
login manually -- health_agent.py itself supplies the credentials to the
browser context.

NOTE ON STEP A: the selectors below are written against the documented
VetBuddy workflow (login -> Common Patient Records -> "Show EMR in"
Descending -> check "Common Patient Medical Record" -> ensure "1 Visit(s)"
is unchecked -> "Generate EMR Group by Visit" -> click "PDF"). They have not
been verified against the live site's actual DOM, since that would require
logging in interactively, which Claude does not do. Treat these as a strong
first draft -- expect to adjust label/role names once run against the real
pages, ideally with Playwright's codegen/inspector open alongside a manual
login.

NOTE ON STEP B: uses `llama_cloud.AsyncLlamaCloud` (llama-cloud>=2.8), the
current SDK -- the old `llama_parse.LlamaParse` class is deprecated. Parse
tier is "agentic" for the highest-fidelity output on tabular/grid medical
data, matching the original high-fidelity requirement.
"""

import hashlib
import math
import os
from datetime import datetime, timezone

import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownTextSplitter
from llama_cloud import AsyncLlamaCloud
from openai import OpenAI
from playwright.async_api import async_playwright
from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection

load_dotenv()

VETBUDDY_URL = "https://nextdoorvets.thevetbuddy.com/"
DOWNLOAD_PATH = "./latest_visit.pdf"
# Portal-side display name of the pet, used to click into their record. This
# is a single named constant rather than a literal buried in the function --
# move it to .env if you add a second pet later.
VETBUDDY_PET_NAME = "Theo"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBEDDING_DIMENSIONS = 1024

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

MONGO_DB_NAME = "pethealthsystem"
MONGO_COLLECTION_NAME = "medical_record_chunks"

_mongo_client: MongoClient | None = None
_token_encoding = None


def _token_length(text: str) -> int:
    """
    Real token count (not characters), used so chunk_size/overlap are
    token-accurate. Lazily loaded on first use (not at import time) so a slow
    or unavailable tiktoken download can't block the whole app from starting.
    """
    global _token_encoding
    if _token_encoding is None:
        _token_encoding = tiktoken.get_encoding("cl100k_base")
    return len(_token_encoding.encode(text))


def truncate_embedding(vector: list[float], dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """
    nvidia/nemotron-3-embed-1b:free natively returns 2048-dim vectors -- the
    API has no way to request a smaller native output. NVIDIA trains this
    model family with Matryoshka representation learning, so slicing to the
    first `dimensions` values and L2-renormalizing yields a valid, still
    highly-functional embedding at that width (documented, supported
    technique -- not an approximation hack). Used to match EMBEDDING_DIMENSIONS
    (1024) and the Atlas Search index configured for that width. Must be
    applied identically to both stored (ingestion) and query (chat) vectors
    or cosine similarity comparisons become meaningless.
    """
    sliced = vector[:dimensions]
    norm = math.sqrt(sum(component * component for component in sliced))
    if norm == 0:
        return sliced
    return [component / norm for component in sliced]


def _chunk_document_id(pet_id: str, text_content: str) -> str:
    """
    Stable id for a chunk so re-syncs are idempotent. VetBuddy's export is
    always the FULL visit history since registration, not an incremental
    delta -- so every sync re-parses and re-chunks visits already ingested
    in a prior sync. Unchanged visits produce byte-identical chunk text, so
    hashing (pet_id, text_content) gives the same id on re-sync: re-upserting
    an unchanged chunk overwrites itself with identical data (a no-op in
    practice), and only genuinely new visit content gets a new id and
    inserts as a new document. Including pet_id keeps this collision-safe
    across pets when multi-pet support is added later.
    """
    digest_input = f"{pet_id}:{text_content}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def get_openrouter_client() -> OpenAI:
    """OpenRouter exposes an OpenAI-compatible API, so the openai SDK works pointed at its base_url."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def get_llamacloud_client() -> AsyncLlamaCloud:
    """Current llama-cloud SDK client (replaces the deprecated llama_parse.LlamaParse)."""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set in .env")
    return AsyncLlamaCloud(api_key=api_key)


def get_mongo_collection() -> Collection:
    """Lazily create the Mongo client from MONGO_URI and return the chunks collection."""
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("MONGO_URI is not set in .env")
        _mongo_client = MongoClient(mongo_uri)
    return _mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


async def download_latest_visit_pdf(headless: bool = True) -> str:
    """
    Step A: Launch a Chromium context via Playwright, log in to VetBuddy,
    navigate to Common Patient Records, set "Show EMR in" to Descending,
    check "Common Patient Medical Record" (ensuring "1 Visit(s)" stays
    unchecked), trigger "Generate EMR Group by Visit", and intercept the PDF
    download.

    headless defaults to True for production (server background worker).
    Pass headless=False for local diagnostic verification, so you can
    visually watch the automation and debug selectors -- see the
    `if __name__ == "__main__"` block at the bottom of this file.

    Returns the local path to the downloaded PDF (./latest_visit.pdf).
    """
    email = os.getenv("VETBUDDY_EMAIL")
    password = os.getenv("VETBUDDY_PASSWORD")
    if not email or not password:
        raise RuntimeError("VETBUDDY_EMAIL / VETBUDDY_PASSWORD are not set in .env")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            print(f"🤖 Navigating to initial portal hook: {VETBUDDY_URL}")
            await page.goto(VETBUDDY_URL)
            await page.wait_for_load_state("load")

            # Capture whatever URL the portal immediately redirects to
            current_url = page.url.lower()
            print(f"📊 Current landing viewport URL: {page.url}")

            # Check for any variation of the authenticated portal dashboard pages
            if "whiteboard" in current_url or "client_patient" in current_url:
                print("💡 Session recognized: Bypassed login form automatically.")
            else:
                print("🔒 Session missing. Programmatically executing login form injection...")
                try:
                    # Verified via playwright codegen against the live site: these
                    # fields have no associated <label>, so get_by_label() never
                    # matched them. They're plain inputs keyed by name attribute.
                    username_field = page.locator('input[name="userid"]')
                    await username_field.wait_for(state="visible", timeout=5000)

                    await username_field.fill(email)
                    await page.locator('input[name="passwd"]').fill(password)
                    await page.get_by_role("button", name="Sign in").click()
                    await page.wait_for_load_state("networkidle")
                    print("✅ Sign-in action successfully posted.")
                except Exception as e:
                    print(f"⚠️ Interactive login failed or element not present: {str(e)}")
                    # If we happen to be past the gate anyway, let it keep moving
                    if "whiteboard" not in page.url.lower() and "client_patient" not in page.url.lower():
                        raise e

            # --- Proceed straight to your specific nav parameters from here ---
            print("🚀 Safely inside session boundaries. Proceeding to navigation blocks...")

            # --- Select the pet ---
            # Verified via codegen: "Theo" and "Common Patient Medical Record"
            # are both link roles, not plain text nodes.
            await page.get_by_role("link", name=VETBUDDY_PET_NAME).click()

            # --- Open "Common Patient Medical Record" (appears below the pet name once selected) ---
            await page.get_by_role("link", name="Common Patient Medical Record").click()

            # --- Click into the left-nav "Info for <pet>" section ---
            # Verified via codegen: this is a direct click, not a hover-to-reveal
            # submenu as originally assumed. The live accessible name has extra
            # padding whitespace ("  Info for Theo  "), which Playwright's
            # role-name matching normalizes automatically.
            await page.get_by_role("link", name=f"Info for {VETBUDDY_PET_NAME}").click()

            # "Print, Fax or Email Medical" opens a popup window -- everything from
            # here on happens on that popup, not the original page. Verified exact
            # link text via codegen (shorter/differently-cased than first guessed).
            async with page.expect_popup() as popup_info:
                await page.get_by_role("link", name="Print, Fax or Email Medical").click()
            export_page = await popup_info.value
            await export_page.wait_for_load_state("load")
            print(f"📄 Export window opened: {export_page.url}")

            # --- Generate the EMR export grouped by visit ---
            await export_page.get_by_role("button", name="Generate EMR Group By Visit").click()
            await export_page.wait_for_load_state("networkidle")

            # No date-selection or intermediate click needed here -- confirmed
            # unnecessary; the PDF button is reachable directly after Generate.

            # --- Download the PDF ---
            # Confirmed via real runs: clicking "PDF" navigates to Chrome's
            # built-in inline PDF viewer, which streams the file via HTTP Range
            # requests rather than one full GET (it often fetches the *end* of
            # the file first, to read the PDF cross-reference table) -- so
            # whatever single response we intercept off that navigation can be a
            # partial chunk, not the complete document, even with the right
            # Content-Type and a 200/206 status. Capturing the click just to
            # learn the real URL, then making our own separate, explicit,
            # non-range GET request to that URL -- reusing the already
            # authenticated session's cookies via context.request -- to get the
            # full file in one clean response instead of trusting the viewer's
            # own fetch.
            async with export_page.expect_response(
                lambda response: "loadpdf.pdf" in response.url
            ) as response_info:
                await export_page.get_by_role("button", name="PDF").click()
            pdf_response = await response_info.value
            pdf_url = pdf_response.url

            direct_response = await export_page.context.request.get(pdf_url)

            if not direct_response.ok:
                raise RuntimeError(
                    f"Direct PDF fetch failed with status {direct_response.status}: {pdf_url}"
                )

            content_type = direct_response.headers.get("content-type", "")
            pdf_bytes = await direct_response.body()

            print(f"📦 Direct-fetched: {pdf_url}")
            print(f"📦 Content-Type: {content_type!r}, size: {len(pdf_bytes)} bytes")
            print(f"📦 First 16 bytes: {pdf_bytes[:16]!r}")

            if not pdf_bytes.startswith(b"%PDF-"):
                raise RuntimeError(
                    "Direct fetch still did not return a real PDF (missing the "
                    f"%PDF- magic bytes). Content-Type was {content_type!r}. The "
                    "URL may need additional headers/cookies beyond what the "
                    "browser context attaches automatically, or may not be "
                    "fetchable outside the page context at all."
                )

            with open(DOWNLOAD_PATH, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)
            print(f"✅ PDF downloaded to {DOWNLOAD_PATH} ({len(pdf_bytes)} bytes)")
        finally:
            await browser.close()

    return DOWNLOAD_PATH


async def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Step B: Convert the PDF to structured markdown using the current
    llama-cloud SDK's Parse product (tier="agentic" for high-fidelity output
    on grids/tables), preserving structural formatting and medical tabular
    context.
    """
    client = get_llamacloud_client()

    uploaded_file = await client.files.create(file=pdf_path, purpose="parse")
    result = await client.parsing.parse(
        file_id=uploaded_file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    if result.markdown is None:
        raise RuntimeError("LlamaParse response did not include markdown output.")

    page_texts = []
    for page in result.markdown.pages:
        if not page.success:
            raise RuntimeError(f"LlamaParse failed on page {page.page_number}: {page.error}")
        page_texts.append(page.markdown)

    return "\n\n".join(page_texts)


def chunk_markdown(markdown_text: str) -> list[str]:
    """
    Step C: Split markdown into 512-token chunks with 50-token overlap using
    LangChain's MarkdownTextSplitter, segmented along header hierarchy so
    clinical events and lab modules aren't cut mid-block. Chunk size/overlap
    are measured in real tokens (tiktoken), not characters.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
    )
    return splitter.split_text(markdown_text)


async def embed_and_upsert_chunks(pet_id: str, chunks: list[str]) -> int:
    """
    Step D: Embed each chunk via OpenRouter's nvidia/nemotron-3-embed-1b:free
    (1024 dimensions after truncation) and upsert into the MongoDB Atlas
    collection using the documented schema (pet_id, extracted_at,
    chunk_index, text_content, vector_embedding).

    Each document's _id is a stable hash of (pet_id, text_content) -- see
    _chunk_document_id() -- so re-running this against an unchanged
    VetBuddy export is idempotent instead of duplicating every chunk on
    every sync.

    Returns the number of chunks upserted (inserted or overwritten).
    """
    client = get_openrouter_client()
    collection = get_mongo_collection()
    extracted_at = datetime.now(timezone.utc).isoformat()

    operations: list[ReplaceOne] = []
    for chunk_index, text_content in enumerate(chunks):
        # Sent verbatim, no stop-word stripping, so connecting context tokens survive.
        # encoding_format="float" is required: the SDK defaults to "base64",
        # which OpenRouter's proxy doesn't reliably support for this
        # endpoint -- it silently returns an empty data[] array instead of
        # erroring, which the SDK then reports as "No embedding data received".
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=text_content, encoding_format="float"
        )
        raw_vector = response.data[0].embedding

        if len(raw_vector) < EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Raw embedding from {EMBEDDING_MODEL} has {len(raw_vector)} "
                f"dimensions, which is smaller than the {EMBEDDING_DIMENSIONS} "
                "we truncate down to -- cannot proceed."
            )

        # Model natively returns 2048 dims; truncate + L2-renormalize to
        # EMBEDDING_DIMENSIONS (1024) to match the Atlas Search index. See
        # truncate_embedding() docstring for why this is valid for this model.
        vector_embedding = truncate_embedding(raw_vector, EMBEDDING_DIMENSIONS)

        document = {
            "_id": _chunk_document_id(pet_id, text_content),
            "pet_id": pet_id,
            "extracted_at": extracted_at,
            "chunk_index": chunk_index,
            "text_content": text_content,
            "vector_embedding": vector_embedding,
        }
        operations.append(
            ReplaceOne({"_id": document["_id"]}, document, upsert=True)
        )

    if operations:
        collection.bulk_write(operations)

    return len(operations)


async def run_health_agent_pipeline(pet_id: str | None = None, headless: bool = True) -> dict:
    """
    Runs steps A through D in sequence. Called by main.py's /api/sync route
    with the default headless=True. Pass headless=False for a local visible
    diagnostic run (see the `if __name__ == "__main__"` block below).
    """
    resolved_pet_id = pet_id or os.getenv("PET_ID")
    if not resolved_pet_id:
        raise RuntimeError("No pet_id provided and PET_ID is not set in .env")

    pdf_path = await download_latest_visit_pdf(headless=headless)
    markdown_text = await parse_pdf_to_markdown(pdf_path)
    chunks = chunk_markdown(markdown_text)
    upserted_count = await embed_and_upsert_chunks(resolved_pet_id, chunks)

    return {"pet_id": resolved_pet_id, "chunks_upserted": upserted_count}


if __name__ == "__main__":
    # Manual local diagnostic run: `python health_agent.py [pet_id]`
    #
    # Runs the full pipeline with headless=False so the Chromium window is
    # visible -- use this to watch the VetBuddy form-filling step live and
    # fix any selectors in download_latest_visit_pdf() that don't match the
    # real site (see the NOTE ON STEP A docstring above).
    import asyncio
    import sys

    cli_pet_id = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        result = asyncio.run(run_health_agent_pipeline(pet_id=cli_pet_id, headless=False))
        print(f"Sync complete. pet_id={result['pet_id']} chunks_upserted={result['chunks_upserted']}")
    except Exception as error:
        print(f"Sync failed: {error}")
        raise
