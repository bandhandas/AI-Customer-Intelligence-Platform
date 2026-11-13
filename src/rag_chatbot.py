
from pathlib import Path
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# 1. Project configuration
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "database" / "chroma_db"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# -----------------------------------------
# 2. Load embedding model
# -----------------------------------------

print("Loading RAG components...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------------
# 3. Connect ChromaDB
# -----------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="telecom_customer_knowledge"
)

if collection.count() == 0:
    raise RuntimeError(
        "Knowledge base is empty. Run ingest_documents.py first."
    )

print("RAG chatbot ready!")
print("Knowledge chunks:", collection.count())

# -----------------------------------------
# 4. Retrieve relevant documents
# -----------------------------------------

def retrieve_documents(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved_context = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown")
        retrieved_context.append(
            f"Source: {source}\n{document}"
        )

    return retrieved_context

# -----------------------------------------
# 5. Generate grounded AI answer
# -----------------------------------------

def generate_answer(question):

    context = retrieve_documents(question)

    if not context:
        return "No relevant information found in the knowledge base."

    context_text = "\n\n".join(context)

    prompt = f"""
You are a telecom customer intelligence assistant.

Answer the user's question using ONLY the retrieved context below.

RETRIEVED KNOWLEDGE:
{context_text}

USER QUESTION:
{question}

RULES:
- Give clear and simple answers.
- Do not invent statistics or business facts.
- If the answer is not available in the retrieved context,
  clearly say that the knowledge base does not contain enough information.
- Distinguish facts from possible explanations.
- Mention the source document names when relevant.

ANSWER:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    return answer, context

# -----------------------------------------
# 6. Interactive chatbot
# -----------------------------------------

print("\n========== TELECOM RAG CHATBOT ==========")
print("Ask questions about customer churn.")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Chatbot closed.")
        break

    if not question:
        print("Please enter a question.")
        continue

    try:
        answer, sources = generate_answer(question)

        print("\nAI:", answer)

        print("\nSources:")
        for source in sources:
            print("-", source.split("\n")[0])

        print("\n" + "=" * 50 + "\n")

    except requests.exceptions.RequestException as error:
        print("Ollama connection error:", error)

    except Exception as error:
        print("Error:", error)
