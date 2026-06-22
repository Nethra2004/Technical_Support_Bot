import os
import re
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

SOURCE_PDF_PATH = os.getenv("SOURCE_PDF_PATH")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

COLLECTION_NAME = "upwork_api_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 250

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf_pages(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text() or ""
        text = clean_text(text)

        if len(text) < 20:
            continue

        pages.append(
            {
                "page": page_num + 1,
                "text": text
            }
        )

    return pages


def chunk_pages(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = []

    global_chunk = 0

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for local_chunk, chunk in enumerate(page_chunks):

            chunk = chunk.strip()

            if len(chunk) < 80:
                continue

            chunks.append(
                {
                    "id": f"chunk_{global_chunk}",
                    "text": chunk,
                    "metadata": {
                        "page": page["page"],
                        "chunk_index": global_chunk,
                        "page_chunk": local_chunk,
                        "source": SOURCE_PDF_PATH
                    }
                }
            )

            global_chunk += 1

    return chunks

def build_vector_store(chunks):

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print("\n======================================")
    print("Vector Store Created Successfully")
    print("======================================")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Chunks     : {len(chunks)}")
    print(f"Directory  : {CHROMA_PERSIST_DIR}")
    print("======================================\n")

def print_statistics(chunks):

    print("\n============= INGEST SUMMARY =============")

    print(f"Total chunks : {len(chunks)}")

    total_chars = sum(len(c["text"]) for c in chunks)

    avg = total_chars / len(chunks)

    print(f"Average chunk size : {avg:.0f}")

    print("\nSample metadata:")

    print(chunks[0]["metadata"])

    print("\nSample chunk:\n")

    print(chunks[0]["text"][:700])

    print("\n==========================================\n")

def main():

    if not os.path.exists(SOURCE_PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found:\n{SOURCE_PDF_PATH}"
        )

    print("Loading PDF...")

    pages = load_pdf_pages(SOURCE_PDF_PATH)

    print(f"Pages extracted : {len(pages)}")

    print("Chunking pages...")

    chunks = chunk_pages(pages)

    print_statistics(chunks)

    print("Building Chroma vector database...")

    build_vector_store(chunks)

    print("Done.")

if __name__ == "__main__":
    main()
