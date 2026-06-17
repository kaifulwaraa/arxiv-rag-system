import os
from dotenv import load_dotenv
from groq import Groq
from retrieval import (
    load_all_chunks,
    naive_vector_search,
    hybrid_search,
    hybrid_rerank_search
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, chunks):
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    prompt = f"""You are an expert AI researcher. Answer the question based ONLY on the provided context from research papers.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

def ask(query, strategy="rerank", all_chunks=None):
    print(f"\nQuestion: {query}")
    print(f"Strategy: {strategy}")
    print("-" * 60)
    if strategy == "naive":
        chunks = naive_vector_search(query, top_k=5)
    elif strategy == "hybrid":
        chunks = hybrid_search(query, all_chunks, top_k=5)
    elif strategy == "rerank":
        chunks = hybrid_rerank_search(query, all_chunks, top_k=5)
    answer = generate_answer(query, chunks)
    print(f"Answer:\n{answer}")
    print(f"\nSources used:")
    for i, c in enumerate(chunks):
        print(f"  {i+1}. {c['source']} (score: {c['score']:.4f})")
    return {"query": query, "answer": answer, "chunks": chunks, "strategy": strategy}

if __name__ == "__main__":
    print("Loading chunks for BM25...")
    all_chunks = load_all_chunks()
    test_questions = [
        "What are the main limitations of RAG systems?",
        "What is the difference between naive RAG and advanced RAG?"
    ]
    for question in test_questions:
        ask(question, strategy="rerank", all_chunks=all_chunks)
        print("\n" + "=" * 60 + "\n")