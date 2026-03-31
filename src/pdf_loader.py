from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pdfplumber
import os

load_dotenv()

# ── Create a sample PDF for testing ───────────────────────────────────────────
def create_sample_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "sample.pdf")

    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("RAG Pipeline Architecture", styles["Heading1"]))
    story.append(Paragraph(
        "RAG systems combine retrieval with language model generation. "
        "They reduce hallucinations by grounding responses in real documents.",
        styles["Normal"]
    ))

    story.append(Paragraph("Document Processing", styles["Heading2"]))
    story.append(Paragraph(
        "Documents must be loaded, chunked, and embedded before storage. "
        "Chunking strategy directly affects retrieval quality.",
        styles["Normal"]
    ))

    story.append(Paragraph("Vector Storage Comparison", styles["Heading2"]))
    table_data = [
        ["Database", "Type",   "Best For"],
        ["Chroma",   "Local",  "Development"],
        ["FAISS",    "Local",  "Speed"],
        ["Pinecone", "Cloud",  "Production"],
        ["Weaviate", "Hybrid", "Scale"],
    ]
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID",       (0, 0), (-1, -1), 1, colors.black),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
    ]))
    story.append(table)

    doc.build(story)
    print(f"Sample PDF created: {path}")
    return path


# ── Load PDF with pdfplumber (text + tables separately) ───────────────────────
def load_pdf_with_plumber(path: str) -> list[Document]:
    """
    Load PDF using pdfplumber preserving:
    - Page text with page number metadata
    - Tables extracted as structured text separately
    - Page dimensions for layout-aware processing
    """
    documents = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # ── Text extraction ────────────────────────────────────────────────
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source":      os.path.basename(path),
                        "page":        page_num + 1,
                        "total_pages": len(pdf.pages),
                        "type":        "text",
                        "width":       page.width,
                        "height":      page.height,
                    }
                ))

            # ── Table extraction ───────────────────────────────────────────────
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table:
                    continue

                rows = []
                for row in table:
                    clean_row = [
                        cell.strip() if cell else ""
                        for cell in row
                    ]
                    rows.append(" | ".join(clean_row))

                table_text = "\n".join(rows)
                documents.append(Document(
                    page_content=table_text,
                    metadata={
                        "source":      os.path.basename(path),
                        "page":        page_num + 1,
                        "type":        "table",
                        "table_index": table_idx,
                    }
                ))

    return documents


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = create_sample_pdf()

    # Method 1: PyPDFLoader (simple)
    print("\n" + "=" * 60)
    print("Method 1 — PyPDFLoader (simple, page by page)")
    print("=" * 60)
    pypdf_loader = PyPDFLoader(pdf_path)
    pypdf_docs = pypdf_loader.load()
    print(f"Pages loaded: {len(pypdf_docs)}")
    for i, page in enumerate(pypdf_docs):
        print(f"\nPage {i + 1}:")
        print(f"  Metadata : {page.metadata}")
        print(f"  Content  : {page.page_content[:150]}...")

    # Method 2: PDFPlumber (richer extraction)
    print("\n" + "=" * 60)
    print("Method 2 — PDFPlumber (tables + text separately)")
    print("=" * 60)
    plumber_docs = load_pdf_with_plumber(pdf_path)
    print(f"Total chunks: {len(plumber_docs)}")
    for doc in plumber_docs:
        print(f"\n  Type : {doc.metadata['type']} | Page: {doc.metadata['page']}")
        print(f"  Text : {doc.page_content[:150]}...")

    # Chunk and store in Chroma
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    text_docs   = [d for d in plumber_docs if d.metadata["type"] == "text"]
    table_docs  = [d for d in plumber_docs if d.metadata["type"] == "table"]
    split_texts = splitter.split_documents(text_docs)
    all_chunks  = split_texts + table_docs

    print(f"\nText chunks : {len(split_texts)}")
    print(f"Table chunks: {len(table_docs)}")
    print(f"Total chunks: {len(all_chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    store = Chroma.from_documents(
        all_chunks,
        embeddings,
        persist_directory=persist_dir,
        collection_name="pdf_docs"
    )

    query = "Which database is best for production?"
    print(f"\nQuery: '{query}'")

    print("\nFilter type='table':")
    table_results = store.similarity_search(query, k=2, filter={"type": "table"})
    for doc in table_results:
        print(f"  Page {doc.metadata['page']} → {doc.page_content}")

    print("\nAll results with page citations:")
    all_results = store.similarity_search_with_score(query, k=3)
    for doc, score in all_results:
        print(f"  Score={score:.4f} | "
              f"Page {doc.metadata.get('page', '?')} | "
              f"Type={doc.metadata['type']} | "
              f"{doc.page_content[:60]}...")