# 📄 Local PDF Semantic Search (Offline & Confidential)

A minimalist, developer-friendly Local RAG system to perform semantic search across private PDFs. This setup ensures your data remains completely offline and confidential.

## ✨ Key Features
- **Offline First:** Uses local LLMs (via Ollama) and local embeddings. No data ever leaves your machine.
- **Incremental Indexing:** Ingests only new PDFs added to your collection.
- **Temporal Priority:** Handles document versioning by prioritizing newer files when conflicts arise, based on the filename date.
- **Traceability:** Provides full citations, including the source PDF and specific sections used for answers.
- **Easily Shared:** Since it uses LanceDB (a serverless vector database), you can simply zip and share the `vector_db/` folder with colleagues.

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Python 3.10+**
- **Ollama:** [Install Ollama](https://ollama.com/) on your machine.
- **LLM & Embedding Models:** Once Ollama is installed, run:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```
  *(Note: You can change the models in `ingest.py` and `query.py` if needed.)*

### 2. Installation
Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Usage

#### Option A: Using the Streamlit UI (Recommended)
Launch the interactive web interface:
```bash
streamlit run app.py
```
- **Upload:** Drag and drop your PDFs into the sidebar.
- **Naming:** Files should follow the `YYYY-MM-DD - Title.pdf` format for proper date-based prioritization.
- **Process:** Click "Process & Index Documents" to embed them.
- **Search:** Ask questions in the main chat area.

#### Option B: Using the CLI
You can also run ingestion and queries from the terminal:

**To index documents:**
1. Place your PDFs in the `pdfs/` folder.
2. Run:
   ```bash
   python ingest.py
   ```

**To ask a question:**
```bash
python query.py "What is the policy on [topic]?"
```

## 📂 Project Structure
- `app.py`: Streamlit-based user interface.
- `ingest.py`: Core logic for reading PDFs and updating the LanceDB vector store.
- `query.py`: Retrieval and generation logic using LlamaIndex and Ollama.
- `pdfs/`: Folder containing your original document files.
- `vector_db/`: Local database storage (safe to share with other developers).

## 🤝 Sharing the Index
To share the search capabilities with colleagues without sharing the original PDFs:
1. Zip the `vector_db/` folder.
2. Share the zip file via a secure channel.
3. Your colleague should extract the `vector_db/` folder into the same directory as the project scripts.
4. They can then run `streamlit run app.py` and immediately search your pre-indexed documents.

*Note: The raw text chunks remain stored in the database for search purposes, so only share with trusted parties.*
