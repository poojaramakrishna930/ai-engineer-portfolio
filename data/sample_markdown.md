# Introduction to RAG

Retrieval-Augmented Generation is a technique that combines retrieval systems with language models.
It was introduced to reduce hallucinations in LLM outputs by grounding answers in real documents.

## How RAG Works

RAG works in two stages. First, relevant documents are retrieved from a knowledge base.
Second, those documents are passed as context to the LLM along with the user question.

### Retrieval Stage

The retrieval stage uses a vector database to find semantically similar chunks.
Each chunk is scored using cosine similarity against the query embedding.
The top k chunks are selected and passed downstream.

### Generation Stage

The generation stage takes the retrieved chunks and the original question.
The LLM uses the chunks as grounding context to produce a factual answer.
Without this context, LLMs often hallucinate or give outdated information.

## Chunking Strategies

Chunking is the process of splitting documents into smaller pieces before embedding.
The quality of chunking directly affects retrieval quality and final answer accuracy.

### Fixed Size Chunking

Fixed size chunking splits text every N characters regardless of sentence boundaries.
It is simple and fast but often produces meaningless chunks that cut mid-sentence.

### Recursive Chunking

Recursive chunking respects natural text boundaries like paragraphs and sentences.
It is the recommended default for most RAG pipelines due to its balance of speed and quality.