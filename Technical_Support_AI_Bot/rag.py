import os
import time
import requests
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from ingest import main

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

COLLECTION_NAME = "upwork_api_docs"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

TOP_K = 3

MAX_DISTANCE = 0.60

CANDIDATE_MULTIPLIER = 8

MIN_CANDIDATES = 30

MIN_CONTENT_LENGTH = 200

NEIGHBORS_BEFORE = 1

NEIGHBORS_AFTER = 3

SYSTEM_PROMPT = """
You are a Senior Technical Support Consultant specializing in the Upwork API.

Your task is to answer user questions using ONLY the information provided in the current conversation. Treat that information as your complete and only knowledge source.

Follow these instructions strictly:

1. Answer every question exclusively using the information provided in the current conversation.

2. Never use external knowledge, prior knowledge, assumptions, inference, or logical extrapolation beyond the provided information.

3. Present answers as direct technical support responses.

   Correct:
   "Use the POST /offers endpoint to create an offer."

   Correct:
   "The request must include the Authorization header."

   Correct:
   "The response contains the offer_id field."

   Incorrect:
   "According to the documentation..."
   "Based on the documentation..."
   "The documentation states..."
   "The provided information says..."
   "The retrieved context mentions..."
   "From the API documentation..."
   "The source indicates..."

4. Never mention or imply where the information came from. Respond as though the information is already part of your knowledge.

5. When available, include all relevant technical details exactly as provided, including:
   - API endpoints
   - HTTP methods
   - Headers
   - Request parameters
   - Request bodies
   - Authentication requirements
   - Response fields
   - Error codes
   - Example requests
   - Example responses

6. If multiple pieces of information are relevant, combine them into one complete answer without mentioning their source.

7. Do not invent, infer, or modify any information that is not explicitly provided.

8. If the provided information does not explicitly answer the user's question, respond with exactly:

I'm sorry, but the provided documentation does not contain that information.

Do not add any explanation, apology, suggestion, or additional text.

9. Organize answers using bullet points or numbered lists whenever they improve readability.

10. Maintain a concise, accurate, and professional technical support tone at all times.
"""

def get_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR
    )

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )

    except Exception:
        print("Collection not found. Building vector database...")

        main()

        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )

def fetch_chunk(collection, chunk_index):

    if chunk_index < 0:
        return None

    try:

        result = collection.get(
            ids=[f"chunk_{chunk_index}"]
        )

        if result["documents"]:
            return result["documents"][0]

    except Exception:
        return None

    return None

def remove_overlap(previous, current):

    maximum = min(
        len(previous),
        len(current),
        120,
    )

    for size in range(maximum, 0, -1):

        if previous[-size:] == current[:size]:
            return current[size:]

    return current

def expand_chunk(collection, candidate):

    metadata = candidate["metadata"]

    chunk_index = metadata.get("chunk_index")

    if chunk_index is None:
        return candidate

    pieces = []

    for offset in range(
        NEIGHBORS_BEFORE,
        0,
        -1,
    ):

        text = fetch_chunk(
            collection,
            chunk_index - offset,
        )

        if text:
            pieces.append(text)

    pieces.append(candidate["text"])

    for offset in range(
        1,
        NEIGHBORS_AFTER + 1,
    ):

        text = fetch_chunk(
            collection,
            chunk_index + offset,
        )

        if text:
            pieces.append(text)

    merged = pieces[0]

    for piece in pieces[1:]:

        merged += "\n"

        merged += remove_overlap(
            merged,
            piece,
        )

    updated = dict(candidate)

    updated["text"] = merged

    return updated

def get_relevant_chunks(query, top_k=TOP_K):

    collection = get_collection()

    collection_size = collection.count()

    if collection_size == 0:
        return []

    n_candidates = max(
        top_k * CANDIDATE_MULTIPLIER,
        MIN_CANDIDATES,
    )

    n_candidates = min(
        n_candidates,
        collection_size,
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_candidates,
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return []

    candidates = []

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas,
    ):

        candidates.append(
            {
                "text": document,
                "distance": distance,
                "metadata": metadata,
            }
        )

    best_distance = min(
        candidate["distance"]
        for candidate in candidates
    )

    if best_distance > MAX_DISTANCE:
        return []

    candidates.sort(
        key=lambda candidate: (
            len(candidate["text"]) < MIN_CONTENT_LENGTH,
            candidate["distance"],
        )
    )

    selected = candidates[:top_k]

    expanded = []

    for candidate in selected:

        expanded.append(
            expand_chunk(
                collection,
                candidate,
            )
        )

    return expanded

def build_context(retrieved_chunks):

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"""[Source {index}]
Page: {page}

{chunk["text"]}
"""
        )

    return "\n\n-----------------------------\n\n".join(
        context_parts
    )

def call_llm(messages):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 700,
    }

    response = requests.post(
        LLM_API_BASE_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"].strip()

def ask(query, top_k=TOP_K):

    start_time = time.time()

    retrieved_chunks = get_relevant_chunks(
        query=query,
        top_k=top_k,
    )

    if not retrieved_chunks:

        return {
            "answer": (
                "I'm sorry, but the provided documentation "
                "does not contain that information."
            ),
            "sources": [],
            "latency_seconds": time.time() - start_time,
        }

    context = build_context(retrieved_chunks)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content":
                f"Documentation:\n\n"
                f"{context}\n\n"
                f"Question: {query}",
        },
    ]

    try:

        answer = call_llm(messages)

    except requests.exceptions.RequestException as exc:

        answer = (
            "Error communicating with the language model:\n"
            f"{exc}"
        )

    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "latency_seconds": time.time() - start_time,
    }

def print_sources(sources):

    if not sources:
        return

    print("\n----------------------------")
    print("Retrieved Sources")
    print("----------------------------")

    for index, source in enumerate(sources, start=1):

        metadata = source.get("metadata", {})

        print(
            f"{index}. "
            f"Page {metadata.get('page', 'Unknown')} | "
            f"Chunk {metadata.get('chunk_index', 'Unknown')} | "
            f"Distance {source['distance']:.4f}"
        )

def main():

    print("=" * 60)
    print("Upwork API Documentation Assistant")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:

        try:
            query = input("\nQuestion: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):

            print("Goodbye!")
            break

        result = ask(query)

        print("\nAnswer")
        print("-" * 60)
        print(result["answer"])

        print(
            f"\nLatency: "
            f"{result['latency_seconds']:.2f} seconds"
        )

        print_sources(result["sources"])

if __name__ == "__main__":
    main()
