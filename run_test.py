from src.infrastructure.pdf_loader import PdfLoader

# 1. Initialize
loader = PdfLoader()

# 2. Load
chunks = loader.load("test.pdf")

# 3. Verify
print(f"Total pages loaded: {len(chunks)}")

if chunks:
    print(f"\n--- Page {chunks[0].page_number} Preview ---")
    print(chunks[0].content[:200]) # Print first 200 characters
    print(f"Metadata: {chunks[0].model_dump()}") # Pydantic's method to see all fields
else:
    print("No chunks found.")