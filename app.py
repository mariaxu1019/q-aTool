from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

qdrant_url = "http://localhost:6333"
collection_name = "gpt_db"

client = QdrantClient(
    url=qdrant_url,
    prefer_grpc=False,
)

print(client)
print("#####################")

db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

query = "what are the limitations of GPT-4?"

docs = db.similarity_search_with_score(query, k=5)
for i in docs:
    doc, score = i
    print(f"Score: {score}, Document: {doc.page_content}, Metadata: {doc.metadata}")
print("#####################")