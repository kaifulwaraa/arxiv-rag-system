# ArXiv RAG System

A production-grade Retrieval-Augmented Generation system that benchmarks 3 retrieval strategies over 20 ArXiv research papers, featuring an agentic layer, multimodal extraction, and domain-adapted embeddings.

## Demo

### Q&A Examples

![Question 1](question%201.jpg)

![Question 2](question%202.jpg)

### Benchmark

![Benchmark](benchmark.jpg)

### Full Demo

![Demo](Recording5-ezgif.com-reverse.gif)

## Architecture
User Question

↓

Question Classifier (SIMPLE / COMPARISON / MULTI_PART / COMPLEX / CROSS_PAPER)

↓

Query Planner → 2-5 targeted sub-queries

↓

Hybrid Retriever (BM25 + Vector Search + Cross-Encoder Reranking)

↓

Multimodal Chunks (Text + Tables + Figure Captions)

↓

Groq LLaMA 3.3 70B → Answer Generation

↓

Self-Check → Refine if incomplete (max 3 iterations)

↓

Final Answer

## Benchmark Results

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------|-------------|-----------------|-------------------|----------------|
| Naive Vector | 0.72 | 0.72 | 0.43 | 0.56 |
| Hybrid BM25+Vector | 0.96 | 0.48 | 0.21 | 0.36 |
| **Hybrid + Reranking** | **0.80** | **0.92** | **0.64** | **0.92** |

Reranking wins 3/4 metrics. Answer relevancy **+28%** and context recall **+64%** over naive baseline.

## Key Features

**3 Retrieval Strategies**
- Naive vector search
- Hybrid BM25 + vector with Reciprocal Rank Fusion
- Hybrid + cross-encoder reranking (best performance)

**Agentic Layer**
- Classifies questions into 5 types including CROSS_PAPER mode
- Plans and executes multiple targeted sub-queries
- Self-evaluates answers and refines with additional searches
- Max 3 iterations with smart placeholder detection

**Multimodal Extraction**
- Text chunks from 20 ArXiv PDFs (2,708 chunks)
- Table extraction with explicit cell labels using pdfplumber (59 chunks)
- Figure caption extraction with regex (148 chunks)
- All stored in Qdrant with chunk_type metadata

**Evaluation**
- Custom sequential RAGAS pipeline
- Evaluates faithfulness, answer relevancy, context precision, context recall
- Sequential calls with delays to avoid rate limits

## Tech Stack

- **LLM**: Groq LLaMA 3.3 70B Versatile
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2
- **Reranker**: cross-encoder ms-marco-MiniLM-L-6-v2
- **Vector DB**: Qdrant Cloud
- **Framework**: LangChain, Streamlit
- **PDF Processing**: PyMuPDF, pdfplumber
- **Evaluation**: RAGAS

## Project Structure
arxiv-rag-system/

├── src/

│   ├── ingestion.py          # Multimodal PDF ingestion

│   ├── retrieval.py          # 3 retrieval strategies

│   ├── rag_pipeline.py       # Answer generation

│   ├── evaluation.py         # RAGAS evaluation

│   ├── agent.py              # Agentic layer

│   └── app.py                # Streamlit UI

├── results/

│   ├── test_questions.json

│   └── benchmark_results.json

└── data/

└── processed/

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/kaifulwaraa/arxiv-rag-system.git
cd arxiv-rag-system
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file:
GROQ_API_KEY=your_key

QDRANT_URL=your_qdrant_url

QDRANT_API_KEY=your_qdrant_key

GEMINI_API_KEY=your_gemini_key

**5. Run the app**
```bash
streamlit run src/app.py
```

## What I Learned

- Fine-tuning `all-MiniLM-L6-v2` on 760 domain-specific pairs — learned why small models need 10k+ pairs and `MultipleNegativesRankingLoss` for meaningful improvement
- Hybrid retrieval consistently outperforms pure vector search on faithfulness
- Cross-encoder reranking provides the strongest signal for context recall
- Agentic query planning with CROSS_PAPER mode significantly improves multi-document synthesis

## Future Work

- Fine-tune with larger base model (bge-base-en-v1.5, 110M parameters) on 10k+ pairs
- Add knowledge graph for explicit cross-paper concept linking
- Expand to 50-80 papers
- Deploy on HuggingFace Spaces

## Author

**Kaif Ul Wara** — AI & Solutions Engineer  
[GitHub](https://github.com/kaifulwaraa)