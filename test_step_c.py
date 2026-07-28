"""
Standalone diagnostic for Step C only: markdown -> token-based chunks.

Does NOT touch Steps A, B, or D. Reads the existing ./parsed_output.md
(the real output from your successful Step B run) and calls
chunk_markdown() directly.

Run from inside pethealthsystem/ with your venv active:
    python test_step_c.py
"""

from health_agent import chunk_markdown, _token_length, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

INPUT_PATH = "./parsed_output.md"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    print(f"🤖 Chunking {INPUT_PATH} ({len(markdown_text)} chars, "
          f"{_token_length(markdown_text)} tokens)...")
    print(f"   target chunk_size={CHUNK_SIZE_TOKENS} tokens, "
          f"chunk_overlap={CHUNK_OVERLAP_TOKENS} tokens")

    chunks = chunk_markdown(markdown_text)

    print(f"✅ Produced {len(chunks)} chunks\n")

    oversized = []
    for i, chunk in enumerate(chunks):
        tok_count = _token_length(chunk)
        flag = ""
        if tok_count > CHUNK_SIZE_TOKENS:
            flag = "  ⚠️ OVER TARGET"
            oversized.append(i)
        print(f"--- Chunk {i} | {tok_count} tokens | {len(chunk)} chars{flag} ---")
        preview = chunk.strip().replace("\n", " ")[:120]
        print(f"    {preview}...")

    print(f"\n📊 Summary: {len(chunks)} chunks, "
          f"{len(oversized)} over the {CHUNK_SIZE_TOKENS}-token target "
          f"({oversized if oversized else 'none'})")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"❌ Step C failed: {error}")
        raise
