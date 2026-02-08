from pydantic import BaseModel

class DocumentChunk(BaseModel):
    content: str
    page_number: int