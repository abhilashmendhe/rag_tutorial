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

# 3. Load documents from the directory and store it in an array. Return the list of documents
def load_documents(dir_path):
    print("------------- Loading documents -------------")
    documents = []
    for filename in os.listdir(dir_path):
        # print(filename)
        with open(os.path.join(dir_path, filename)) as file:
            documents.append({"id":filename, "text":file.read()})
    
    return documents

# 4. create a split function that splits the text of each document for default chunk size 1000
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

documents = load_documents("./news-articles")
print(f"Loaded {len(documents)} documents.")
# print(len(split_text(documents[0]["text"])))
# print(documents[0])

# 5. Call split function on each document and store in a list of chunked documents
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("----- Splitting docs into chunks -----")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id":f"{doc['id']}_chunk{i+1}", "text":chunk})
        
# print(f"Split documents into {len(chunked_documents)} chunks")

# print(chunked_documents[0])

# 6. Create a function that returns the embedding of each chunked docs
def generate_embedding(text):
    embedding = embbed_model.embed(text)
    return list(embedding)

# 7. Generate embedding for the chunked documents
for doc in chunked_documents:
    print("------ Generating embeddings ------")    
    doc["embedding"] = generate_embedding(doc["text"])
    # break

# print(chunked_documents[0]["embedding"])

# 8. Upsert or insert documents inside chroma db

for doc in chunked_documents:
    print("---- Inserting chunks into db;;; ----")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=doc["embedding"]
    )
    
print("Completed inserting documents in chunked documents")