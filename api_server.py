from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from pydantic import BaseModel
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://chgffkajpomobjfadaoofdfblamibhgk"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "BAAI/bge-large-en"
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)

client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
collection_name = "current_pdf"

@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form(...)
):
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    Qdrant.from_documents(
        texts,
        embeddings,
        collection_name=collection_name,
        url="http://localhost:6333",
        prefer_grpc=False,
    )
    return {"status": "uploaded and indexed", "collection_name": collection_name}

class QueryInput(BaseModel):
    query: str
    collection_name: str

@app.post("/query")
def query_pdf(body: QueryInput):
    print("Received query:", body.query)
    print("Collection name:", body.collection_name)
    if not client.collection_exists(body.collection_name):
        return [{"score": 0, "text": "No PDF collection found for that name.", "meta": {}}]

    db = Qdrant(
        client=client,
        collection_name=body.collection_name,
        embeddings=embeddings,
    )
    results = db.similarity_search_with_score(body.query, k=5)
    return [
        {"score": score, "text": doc.page_content}
        for doc, score in results
    ]

