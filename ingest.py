import os
import re
import lancedb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configuration
PDF_DIR = "./pdfs"
DB_DIR = "./vector_db"
TABLE_NAME = "pdf_chunks"

# Setup Local Embedding & LLM (Ollama)
# Using nomic-embed-text for high performance local embeddings
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)

def extract_date_from_filename(filename):
    """
    Extracts date from filename expecting YYYY-MM-DD - Title.pdf
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    return "1970-01-01"

def filename_metadata_extractor(file_path):
    """
    Custom metadata extractor to add date from filename
    """
    filename = os.path.basename(file_path)
    date_str = extract_date_from_filename(filename)
    return {
        "file_name": filename,
        "version_date": date_str,
    }

def run_ingestion():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"Created {PDF_DIR}. Please add PDFs there.")
        return

    # Connect to LanceDB
    vector_store = LanceDBVectorStore(uri=DB_DIR, table_name=TABLE_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load documents with custom metadata
    # LlamaIndex keeps track of which files have been loaded via Docstore
    reader = SimpleDirectoryReader(
        input_dir=PDF_DIR,
        file_metadata=filename_metadata_extractor,
        recursive=True
    )

    documents = reader.load_data()

    if not documents:
        print("No documents found in pdfs directory.")
        return

    # To achieve true incremental indexing and avoid duplicates,
    # we first check if the index exists.
    try:
        # Load existing index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
        # refresh_ref_docs will only insert nodes for files that are new or changed
        refreshed_docs = index.refresh_ref_docs(documents)
        print(f"Incremental update: {sum(refreshed_docs)} new/updated documents processed.")
    except Exception:
        # Create fresh index if it doesn't exist
        print("Creating fresh index...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print(f"Successfully indexed {len(documents)} document pages into {TABLE_NAME}.")

if __name__ == "__main__":
    run_ingestion()
