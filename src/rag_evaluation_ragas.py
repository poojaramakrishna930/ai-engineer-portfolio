from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from datasets import Dataset
import json
import os

load_dotenv()

# ── NOTE on RAGAS + LLM ───────────────────────────────────────────────────────
# RAGAS faithfulness and answer relevancy need an LLM to evaluate.
# We will use HuggingFace inference API (free) as the evaluator LLM.
# If you have an OpenAI key, swap in ChatOpenAI for better results.

# ── Load data ──────────────────────────────────────────────────────────────────
eval_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
with open(eval_path, "r") as f:
    eval_data = json.load(f)

docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(docs_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

# ── Build retriever ────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
vectorstore = Chroma.from_documents(
    docs, embeddings,
    persist_directory=persist_dir,
    collection_name="eval_ragas"
)
bm25_retriever  = BM25Retriever.from_documents(docs, k=5)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

# ── Simulate RAG answers (without LLM — for structure demo) ───────────────────
# In Phase 2 projects you will replace this with actual LLM generation
def simulate_rag_answer(question: str, context_docs: list[Document]) -> str:
    """
    Simulates what an LLM would return given context.
    In real RAG this is replaced by:
    chain.invoke({"question": question, "context": context_docs})
    """
    # For evaluation demo: concatenate top chunk as "answer"
    # Replace with real LLM call in your project
    if context_docs:
        return context_docs[0].page_content
    return "I don't know."

# ── Build RAGAS evaluation dataset ────────────────────────────────────────────
# RAGAS expects a HuggingFace Dataset with these exact column names:
# - question      : the user query
# - answer        : the generated answer from your RAG pipeline
# - contexts      : list of retrieved chunk texts
# - ground_truth  : the correct answer for reference

print("Building RAGAS evaluation dataset...")
ragas_data = {
    "question":    [],
    "answer":      [],
    "contexts":    [],
    "ground_truth": []
}

for item in eval_data:
    question     = item["question"]
    ground_truth = item["ground_truth"]

    # Retrieve context
    retrieved_docs = hybrid_retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs[:3]]

    # Generate answer (replace with real LLM in projects)
    answer = simulate_rag_answer(question, retrieved_docs)

    ragas_data["question"].append(question)
    ragas_data["answer"].append(answer)
    ragas_data["contexts"].append(contexts)
    ragas_data["ground_truth"].append(ground_truth)

    print(f"  Processed: {question[:60]}...")

# Convert to HuggingFace Dataset — RAGAS requires this format
ragas_dataset = Dataset.from_dict(ragas_data)

print(f"\nDataset built: {len(ragas_dataset)} samples")
print(f"Columns: {ragas_dataset.column_names}")
print(f"\nSample entry:")
print(f"  Question : {ragas_dataset[0]['question']}")
print(f"  Answer   : {ragas_dataset[0]['answer'][:100]}...")
print(f"  Contexts : {len(ragas_dataset[0]['contexts'])} chunks retrieved")
print(f"  Ground   : {ragas_dataset[0]['ground_truth'][:100]}...")

# ── Run RAGAS evaluation ───────────────────────────────────────────────────────
# NOTE: Full RAGAS evaluation requires an LLM API key (OpenAI or HuggingFace)
# The code below shows the correct pattern — swap in your LLM when ready

print("\n" + "=" * 60)
print("RAGAS Evaluation Setup")
print("=" * 60)

try:
    from ragas import evaluate
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # ── Option A: With OpenAI (uncomment when you have a key) ─────────────────
    # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    # result = evaluate(
    #     ragas_dataset,
    #     metrics=[
    #         faithfulness,
    #         answer_relevancy,
    #         context_precision,
    #         context_recall,
    #         context_relevancy,
    #     ],
    #     llm=ChatOpenAI(model="gpt-3.5-turbo"),
    #     embeddings=OpenAIEmbeddings(),
    # )
    # print(result)

    # ── Option B: With HuggingFace inference API (free) ───────────────────────
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in .env")

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
        huggingfacehub_api_token=hf_token
    )
    hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Running RAGAS evaluation with HuggingFace...")
    result = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=hf_embeddings,
    )

    print("\nRAGAS Results:")
    print(f"  Faithfulness      : {result['faithfulness']:.4f}")
    print(f"  Answer Relevancy  : {result['answer_relevancy']:.4f}")
    print(f"  Context Precision : {result['context_precision']:.4f}")
    print(f"  Context Recall    : {result['context_recall']:.4f}")

    # Convert to pandas for easy inspection
    df = result.to_pandas()
    print(f"\nPer-question breakdown:")
    print(df[["question", "faithfulness", "answer_relevancy",
              "context_precision", "context_recall"]].to_string())

except Exception as e:
    print(f"\nRAGAS LLM eval skipped: {e}")
    print("This is expected if no API key is set.")
    print("The dataset structure above is correct and ready for evaluation.")
    print("Add your HuggingFace token to .env to run full evaluation.")