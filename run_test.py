from src.infrastructure.document_splitter import DocumentSplitter  # Make sure file name matches
from src.infrastructure.pdf_loader import PdfLoader

# 1. Initialize Adapters
loader = PdfLoader()
splitter = DocumentSplitter()

# 2. Load
pages = loader.load("2025-pwc-network-sustainability-report.pdf")
print(f"Total pages loaded: {len(pages)}")

# 3. Split
all_splits = []
for page in pages:
    splits = splitter.split(page)
    all_splits.extend(splits)

# 4. Verify
print(f"Total chunks AFTER split: {len(all_splits)}")

if all_splits:
    print(f"\n--- First Split Chunk Preview ---")
    print(f"Content: {all_splits[0].content[:200]}...")  # Print first 200 chars
    print(f"Source Page: {all_splits[0].page_number}")

    # Check the overlap: Let's look at the second chunk
    print(f"\n--- Second Split Chunk Preview ---")
    print(f"Content: {all_splits[1].content[:200]}...")
    print(f"Source Page: {all_splits[1].page_number}")
else:
    print("No chunks found.")