from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ProcessedDocument(BaseModel):
    id: str
    filename: str
    extracted_text: str
    metadata: Dict[str, Any] 

class QueryRequest(BaseModel):
    question: str
    filename: Optional[str] = None