import os
import chromadb
from chromadb.utils import embedding_functions
from fastembed import TextEmbedding

# 1. Create embedding model from fastembed
embbed_model = TextEmbedding()

# 2. Create chromaDB client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=None)

# 3. Create a function to query relavant documents
def query_documents(question, n_results=2):
    
    query_embedding = list(embbed_model.embed([question]))
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    print(results)
    for id,doc in enumerate(results["documents"][0]):
        doc_id = results["ids"][0][id]
        distance = results["distances"][0][id]
        print(f"ID: {doc_id}, Distance: {distance}")
        print(f"Found document chunk: {doc}")
        print("\n\n")
    # return results

query_documents("What is databricks?",n_results=3)