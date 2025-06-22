import streamlit as st
import os
from pipelines.query_pdf import ask_pdf
from ingest.pdf_embedder import embed_pdf_to_qdrant
from qdrant_client import QdrantClient

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="📚 PDF Q&A Chatbot", layout="centered")

# 🌟 Custom Style for Light & Dark Mode
st.markdown("""
<style>
    body, .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .user-msg {
        background-color: var(--secondary-background-color);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .bot-msg {
        background-color: var(--primary-background-color);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .pdf-pill {
        display: inline-block;
        background: #f2e8ff;
        color: #512da8;
        padding: 4px 10px;
        margin-right: 4px;
        border-radius: 12px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# 🔧 Utilities
def clean_collection_name(filename: str) -> str:
    return filename.lower().replace(" ", "_").replace(".pdf", "")

def existing_qdrant_collections():
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    return [col.name for col in client.get_collections().collections]

def embed_all_pdfs_in_folder(folder_path=UPLOAD_DIR):
    collections = existing_qdrant_collections()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            full_path = os.path.join(folder_path, file_name)
            collection_name = clean_collection_name(file_name)
            if collection_name not in collections:
                st.toast(f"📥 Embedding {file_name}...")
                embed_pdf_to_qdrant(full_path, collection_name=collection_name)
                st.toast(f"✅ Embedded `{file_name}`")

# 📂 File Uploader
st.title("📚 Chat with Your PDFs")
st.caption("Upload PDFs, embed them, and ask questions using a powerful RAG + Groq LLM backend.")

with st.expander("📤 Upload new PDF"):
    uploaded_file = st.file_uploader("Upload a new PDF to embed", type=["pdf"])
    if uploaded_file:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ Uploaded `{uploaded_file.name}`")
        embed_pdf_to_qdrant(save_path, collection_name=clean_collection_name(uploaded_file.name))
        st.rerun()

# 📌 Load embedded collections
embed_all_pdfs_in_folder()
collections = existing_qdrant_collections()

# 🔘 PDF Selector
selected_pdfs = st.multiselect("📑 Select one or more PDFs to query:", collections)

# 💬 Ask question
with st.form("chat_form", clear_on_submit=False):
    user_query = st.text_input("💬 Your question")
    submitted = st.form_submit_button("Ask")

# 🤖 Answer Logic
if submitted:
    if user_query.strip() and selected_pdfs:
        with st.spinner("Thinking..."):
            answer = ask_pdf(user_query, collections=selected_pdfs, top_k=6)

        st.markdown('<div class="user-msg">👤 <strong>You:</strong><br>' + user_query + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="bot-msg">🤖 <strong>Answer:</strong><br>' + answer + '</div>', unsafe_allow_html=True)
        st.markdown("📎 <strong>Used PDFs:</strong> " + " ".join([f"<span class='pdf-pill'>{c}</span>" for c in selected_pdfs]), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question and select at least one PDF.")
