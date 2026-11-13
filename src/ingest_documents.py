
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_PATHS = [
    BASE_DIR / "reports" / "business_insights.md",
    BASE_DIR / "reports" / "ai_business_insights.md",
    BASE_DIR / "documents" / "customer_metrics.md"
]

CHROMA_PATH = BASE_DIR / "database" / "chroma_db"

# -----------------------------------------
# 2. Load embedding model
# -----------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

# -----------------------------------------
# 3. Initialize ChromaDB
# -----------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="telecom_customer_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# -----------------------------------------
# 4. Read and split documents
# -----------------------------------------

documents = []
metadatas = []
ids = []

chunk_size = 500
overlap = 100

for document_path in DOCUMENT_PATHS:

    if not document_path.exists():
        print(f"Skipping missing file: {document_path}")
        continue

    text = document_path.read_text(encoding="utf-8")

    words = text.split()

    start = 0
    chunk_number = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            documents.append(chunk)

            metadatas.append({
                "source": document_path.name,
                "chunk": chunk_number
            })

            ids.append(
                f"{document_path.stem}_{chunk_number}"
            )

            chunk_number += 1

        start += chunk_size - overlap

print(f"\nTotal document chunks: {len(documents)}")

if not documents:
    raise RuntimeError("No documents found to ingest.")

# -----------------------------------------
# 5. Generate embeddings and store
# -----------------------------------------

print("\nGenerating embeddings and storing in ChromaDB...")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

# -----------------------------------------
# 6. Verify database
# -----------------------------------------

print("\n========== INGESTION COMPLETE ==========")

print("Documents processed:", len(DOCUMENT_PATHS))
print("Total chunks stored:", collection.count())
print("Database location:", CHROMA_PATH)
