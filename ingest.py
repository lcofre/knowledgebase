import os
import re
import lancedb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from models import NomicOllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configuration
PDF_DIR = "./pdfs"
DB_DIR = "./vector_db"
TABLE_NAME = "pdf_chunks"

# Setup Local Embedding & LLM (Ollama)
# Using nomic-embed-text for high performance local embeddings
Settings.embed_model = NomicOllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

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

    # Try to load existing storage context
    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=DB_DIR
        )
        index_exists = True
    except Exception:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index_exists = False

    # Load documents with custom metadata
    # LlamaIndex keeps track of which files have been loaded via Docstore
    # Use PyMuPDFReader for better parsing
    reader = SimpleDirectoryReader(
        input_dir=PDF_DIR,
        file_metadata=filename_metadata_extractor,
        recursive=True,
        file_extractor={".pdf": PyMuPDFReader()}
    )

    documents = reader.load_data()

    # Metadata exclusion from embedding
    for doc in documents:
        doc.excluded_embed_metadata_keys = ["file_name", "version_date"]

    if not documents:
        print("No documents found in pdfs directory.")
        return

    # To achieve true incremental indexing and avoid duplicates,
    # we first check if the index exists.
    if index_exists:
        try:
            # Load existing index
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            # refresh_ref_docs will only insert nodes for files that are new or changed
            refreshed_docs = index.refresh_ref_docs(documents)
            print(f"Incremental update: {sum(refreshed_docs)} new/updated documents processed.")
            # Persist the updated docstore
            storage_context.persist(persist_dir=DB_DIR)
        except Exception as e:
            print(f"Error refreshing index: {e}. Creating fresh index...")
            index_exists = False

    if not index_exists:
        # Create fresh index if it doesn't exist
        print("Creating fresh index...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        # Persist the storage context (including docstore)
        storage_context.persist(persist_dir=DB_DIR)
        print(f"Successfully indexed {len(documents)} document pages into {TABLE_NAME}.")

if __name__ == "__main__":
    run_ingestion()
