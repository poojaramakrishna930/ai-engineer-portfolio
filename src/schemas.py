from pydantic import BaseModel, Field
from typing import List

# Define what you want the LLM to return
class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="List of source documents used")
    confidence: float = Field(description="Confidence score between 0 and 1")
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

# In interviews, explain: Pydantic forces LLMs to return structured JSON
# instead of freeform text — critical for building reliable pipelines
sample = RAGResponse(
    answer="RAG combines retrieval with generation.",
    sources=["doc1.pdf", "doc2.txt"],
    confidence=0.92,
    follow_up_questions=["How does chunking affect RAG quality?"]
)
print(sample.model_dump_json(indent=2))