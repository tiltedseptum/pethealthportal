"""
Standalone diagnostic for Step B only: PDF -> markdown via llama-cloud.

Does NOT touch Step A (no Playwright run) or Steps C/D. Reads the existing
./latest_visit.pdf and calls parse_pdf_to_markdown() directly.

Run from inside pethealthsystem/ with your venv active:
    python test_step_b.py
"""

import asyncio

from dotenv import load_dotenv

from health_agent import parse_pdf_to_markdown

load_dotenv()

PDF_PATH = "./latest_visit.pdf"
OUTPUT_PATH = "./parsed_output.md"


async def main():
    print(f"🤖 Parsing {PDF_PATH} via llama-cloud (tier=agentic)...")
    markdown = await parse_pdf_to_markdown(PDF_PATH)

    print(f"✅ Parsed markdown length: {len(markdown)} chars")
    print("----- First 1000 chars -----")
    print(markdown[:1000])
    print("----- End preview -----")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"📄 Full output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        print(f"❌ Step B failed: {error}")
        raise
