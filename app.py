import streamlit as st
import os
import shutil
from ingest import run_ingestion, PDF_DIR
from query import ask_question

st.set_page_config(page_title="Local PDF Semantic Search", layout="wide")

st.title("📄 Local PDF Semantic Search (Offline & Confidential)")

# Sidebar for file upload
st.sidebar.header("Manage Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload new PDFs",
    type="pdf",
    accept_multiple_files=True,
    help="Naming convention: YYYY-MM-DD - Title.pdf"
)

if st.sidebar.button("Process & Index Documents"):
    if uploaded_files:
        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)
            st.sidebar.success(f"Saved: {uploaded_file.name}")

        with st.spinner("Indexing documents..."):
            run_ingestion()
        st.sidebar.success("Indexing complete!")
    else:
        st.sidebar.warning("No files uploaded.")

# Show currently indexed files
if os.path.exists(PDF_DIR):
    files = os.listdir(PDF_DIR)
    if files:
        st.sidebar.write("### Currently Indexed:")
        for f in sorted(files):
            st.sidebar.text(f"• {f}")

# Main Chat Interface
st.header("Search & Chat")
question = st.text_input("Ask a question about your documents:", placeholder="e.g., What is the current policy on remote work?")

if st.button("Search") or (question and st.session_state.get('last_q') != question):
    if question:
        st.session_state['last_q'] = question
        with st.spinner("Thinking..."):
            try:
                response = ask_question(question)

                st.markdown("### Answer")
                st.write(response.response)

                st.markdown("### Sources")
                for i, node in enumerate(response.source_nodes):
                    metadata = node.node.metadata
                    with st.expander(f"Source {i+1}: {metadata.get('file_name')} (Date: {metadata.get('version_date')})"):
                        st.write(node.node.get_content())
                        st.write(f"**Relevance Score:** {node.score:.4f}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running and the model (e.g. llama3.2) is pulled.")
    else:
        st.warning("Please enter a question.")

# Instructions for sharing
with st.expander("How to share this with colleagues"):
    st.markdown("""
    1. **Install requirements:** `pip install -r requirements.txt`
    2. **Setup Ollama:** Ensure they have Ollama installed and have run:
       ```bash
       ollama pull llama3.2
       ollama pull nomic-embed-text
       ```
    3. **Copy the index:** Share your `vector_db/` folder with them. They should place it in the same directory as these scripts.
    4. **Run the app:** They can run `streamlit run app.py` and immediately start searching the shared database.
    """)
