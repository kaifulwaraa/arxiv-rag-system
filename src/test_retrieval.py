import sys
sys.path.append('src')
from retrieval import load_all_chunks, hybrid_rerank_search

chunks = load_all_chunks()
results = hybrid_rerank_search('quality threshold tau Vendi-RAG 0.85', chunks, top_k=6)

for r in results:
    print(r['source'])
    print(r['text'][:200])
    print('---')