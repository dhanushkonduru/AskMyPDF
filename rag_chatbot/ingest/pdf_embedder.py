import os
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384

def embed_pdf_to_qdrant(pdf_path, collection_name="pdf_collection"):
    from qdrant_client.http import models

    # Load and chunk PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed
    model = SentenceTransformer(MODEL_NAME)
    embeddings = [model.encode(chunk.page_content) for chunk in chunks]

    # Qdrant
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
    )

    # Prepare points with proper indexing
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "text": chunks[i].page_content,
                "source": os.path.basename(pdf_path)
            }
        )
        for i in range(len(chunks))
    ]

    # Upload points
    client.upload_points(
        collection_name=collection_name,
        points=points
    )

    print(f"✅ Embedded {len(points)} chunks into collection '{collection_name}'")
