from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load both models
print("Loading models...")
base_model = SentenceTransformer('all-MiniLM-L6-v2')
finetuned_model = SentenceTransformer('models/finetuned-embedding-v2')

# Test pairs — question vs correct chunk concept
# These simulate the cross-paper reasoning problem
test_pairs = [
    {
        "query": "what limitation does Telco-RAG identify in conventional RAG setups",
        "positive": "standard retrieval pipelines struggle with highly technical domain vocabulary and fail to retrieve domain specific content accurately",
        "negative": "the model was trained on general internet text and achieves good performance on standard benchmarks"
    },
    {
        "query": "how does RAG fail to address the real underlying problem",
        "positive": "conventional RAG systems retrieve based on surface level keyword matching and miss semantic relationships between technical concepts",
        "negative": "the dataset contains 10000 samples split into training validation and test sets with 80 20 ratio"
    },
    {
        "query": "what improvement does reranking provide over naive retrieval",
        "positive": "cross encoder reranking significantly improves precision by re scoring retrieved candidates using deeper semantic understanding",
        "negative": "the authors propose a new architecture based on transformer blocks with multi head attention mechanisms"
    },
    {
        "query": "how do AI tools improve task outputs without improving underlying thinking",
        "positive": "AI assistance enhances the appearance of critical thinking in outputs but does not necessarily develop the cognitive skills of the user",
        "negative": "the experiment was conducted over three months with participants from five different universities"
    }
]

print("\nComparing base vs fine-tuned model on cross-paper reasoning:\n")
print(f"{'Query':<55} {'Base':>8} {'Finetuned':>10} {'Winner':>8}")
print("-" * 85)

base_wins = 0
ft_wins = 0

for pair in test_pairs:
    query = pair["query"]
    positive = pair["positive"]
    negative = pair["negative"]

    # Base model scores
    base_q = base_model.encode([query])
    base_pos = base_model.encode([positive])
    base_neg = base_model.encode([negative])
    base_pos_score = cosine_similarity(base_q, base_pos)[0][0]
    base_neg_score = cosine_similarity(base_q, base_neg)[0][0]
    base_gap = base_pos_score - base_neg_score

    # Fine-tuned model scores
    ft_q = finetuned_model.encode([query])
    ft_pos = finetuned_model.encode([positive])
    ft_neg = finetuned_model.encode([negative])
    ft_pos_score = cosine_similarity(ft_q, ft_pos)[0][0]
    ft_neg_score = cosine_similarity(ft_q, ft_neg)[0][0]
    ft_gap = ft_pos_score - ft_neg_score

    winner = "FINETUNED" if ft_gap > base_gap else "BASE"
    if winner == "FINETUNED":
        ft_wins += 1
    else:
        base_wins += 1

    print(f"{query[:52]:<55} {base_gap:>8.4f} {ft_gap:>10.4f} {winner:>8}")

print("-" * 85)
print(f"\nBase model wins: {base_wins}/4")
print(f"Fine-tuned model wins: {ft_wins}/4")

if ft_wins > base_wins:
    print("\n✅ Fine-tuning improved cross-paper reasoning!")
elif ft_wins == base_wins:
    print("\n⚠️ Mixed results — fine-tuning helped on some, not others")
else:
    print("\n❌ Fine-tuning did not improve — need more training data")