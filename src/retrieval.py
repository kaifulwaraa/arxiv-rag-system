import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ----------------------------
# Initialize models
# ----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ----------------------------
# Initialize Qdrant client
# ----------------------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

COLLECTION_NAME = "arxiv_papers"

# ----------------------------
# Load all chunks for BM25
# ----------------------------
def load_all_chunks():
    all_chunks = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for r in results:
            all_chunks.append({
                "id": r.id,
                "text": r.payload["text"],
                "source": r.payload["source"]
            })
        if offset is None:
            break
    print(f"Loaded {len(all_chunks)} chunks for BM25")
    return all_chunks

# ----------------------------
# Strategy 1: Naive Vector Search
# ----------------------------
def naive_vector_search(query, top_k=5):
    query_embedding = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    ).points
    seen = set()
    unique_results = []
    for r in results:
        if r.payload["text"] not in seen:
            seen.add(r.payload["text"])
            unique_results.append({
                "text": r.payload["text"],
                "source": r.payload["source"],
                "score": r.score
            })
    return unique_results
# ----------------------------
# Strategy 2: Hybrid BM25 + Vector Search
# ----------------------------
def hybrid_search(query, all_chunks, top_k=5):
    # BM25 search
    tokenized_chunks = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Vector search — get more candidates for fusion
    query_embedding = embedder.encode(query).tolist()
    vector_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=20
    ).points
    

    # Map vector results to index positions
    vector_score_map = {}
    for r in vector_results:
        for i, chunk in enumerate(all_chunks):
            if chunk["id"] == r.id:
                vector_score_map[i] = r.score
                break

    # Reciprocal Rank Fusion
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    rrf_scores = {}

    for rank, idx in enumerate(bm25_ranking[:20]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 60)

    for rank, (idx, _) in enumerate(sorted(vector_score_map.items(), key=lambda x: x[1], reverse=True)):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 60)

    # Get top results
    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    seen = set()
    unique_results = []
    for i in top_indices:
        text = all_chunks[i]["text"]
        if text not in seen:
            seen.add(text)
            unique_results.append({
                "text": text,
                "source": all_chunks[i]["source"],
                "score": rrf_scores[i]
            })
    return unique_results

# ----------------------------
# Strategy 3: Hybrid + Cross-Encoder Reranking
# ----------------------------
def hybrid_rerank_search(query, all_chunks, top_k=5):
    # Get more candidates from hybrid search first
    candidates = hybrid_search(query, all_chunks, top_k=20)

    # Rerank with cross-encoder
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "text": c["text"],
            "source": c["source"],
            "score": float(score)
        }
        for c, score in reranked[:top_k]
    ]

# ----------------------------
# Test all three strategies
# ----------------------------
def test_retrieval():
    print("Loading all chunks for BM25...")
    all_chunks = load_all_chunks()

    test_query = "What are the main limitations of RAG systems?"

    print(f"\nQuery: {test_query}")
    print("=" * 60)

    print("\n--- Strategy 1: Naive Vector Search ---")
    results1 = naive_vector_search(test_query)
    for i, r in enumerate(results1):
        print(f"{i+1}. [{r['source']}] score={r['score']:.4f}")
        print(f"   {r['text'][:150]}...")

    print("\n--- Strategy 2: Hybrid BM25 + Vector ---")
    results2 = hybrid_search(test_query, all_chunks)
    for i, r in enumerate(results2):
        print(f"{i+1}. [{r['source']}] score={r['score']:.4f}")
        print(f"   {r['text'][:150]}...")

    print("\n--- Strategy 3: Hybrid + Reranking ---")
    results3 = hybrid_rerank_search(test_query, all_chunks)
    for i, r in enumerate(results3):
        print(f"{i+1}. [{r['source']}] score={r['score']:.4f}")
        print(f"   {r['text'][:150]}...")

if __name__ == "__main__":
    test_retrieval()
