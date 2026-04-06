import os
import lancedb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer,
    PromptTemplate
)
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from models import NomicOllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# Configuration
DB_DIR = "./vector_db"
TABLE_NAME = "pdf_chunks"

# Setup Local Models (Ollama)
Settings.embed_model = NomicOllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)

# Custom Prompt for Date Priority & Citations
QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are an assistant analyzing a library of documents.\n"
    "Each snippet includes 'version_date' and 'file_name' in its metadata.\n"
    "CRITICAL: If you find conflicting information between documents, "
    "always prioritize the information from the document with the most LATEST date.\n"
    "If the query asks for a list or table, please provide the answer in a clear Markdown format.\n"
    "Always cite the source 'file_name' and 'version_date' for each part of your answer.\n"
    "Query: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

def get_query_engine():
    # Connect to LanceDB
    vector_store = LanceDBVectorStore(uri=DB_DIR, table_name=TABLE_NAME)

    # Load storage context for docstore
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=DB_DIR
    )

    # Load index from stored data
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    # Configure Retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # Configure Response Synthesizer with custom prompt
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=QA_PROMPT
    )

    # Create Query Engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine

def ask_question(question):
    engine = get_query_engine()
    response = engine.query(question)
    return response

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        resp = ask_question(query)
        print("\n--- Answer ---")
        print(resp)
        print("\n--- Sources ---")
        for node in resp.source_nodes:
            metadata = node.node.metadata
            print(f"- File: {metadata.get('file_name')} (Date: {metadata.get('version_date')})")
            print(f"  Snippet: {node.node.get_content()[:200]}...")
    else:
        print("Please provide a question as a command line argument.")
