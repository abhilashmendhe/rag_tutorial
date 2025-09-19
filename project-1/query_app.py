from ollama import chat, ChatResponse
import chromadb
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
    return results

# 4. Invoke LLM 
def invoke_llm(prompt, model="llama3"):
    result = chat(
        model=model,
        messages=[{'role':'user', 'content': prompt}]
    )
    return result.message.content

# 5. RAG pipeline
def rag_answer(question, n_results=2, model="llama3"):
    results = query_documents(question, n_results=n_results)
    docs = results["documents"][0]

    # Step 2: build context prompt
    context = "\n".join(docs)
    prompt = f"""You are a helpful assistant. 
Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    # Step 3: call local LLM
    answer = invoke_llm(prompt, model=model)

    # Step 4: return
    return {
        "question": question,
        "retrieved_docs": docs,
        "answer": answer
    }

# Run a RAG query
result = rag_answer("What is databricks?", n_results=10, model="llama3")

print("\n🔎 Question:", result["question"])
print("\n📄 Retrieved docs:", result["retrieved_docs"])
print("\n🤖 Answer:", result["answer"])