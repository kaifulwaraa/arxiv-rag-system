import json
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Load existing chunks
print("Loading chunks...")
with open("data/processed/chunks_export.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

# Group chunks by source paper
from collections import defaultdict
chunks_by_paper = defaultdict(list)
for chunk in all_chunks:
    if chunk["chunk_type"] == "text" and len(chunk["text"]) > 200:
        chunks_by_paper[chunk["source"]].append(chunk)

papers = list(chunks_by_paper.keys())
print(f"Papers available: {len(papers)}")

cross_paper_pairs = []

# Generate pairs that explicitly connect concepts across papers
print("\nGenerating cross-paper training pairs...")
print("This targets the exact weakness in Question 4...\n")

# Step 1 — Find conceptually similar chunks from DIFFERENT papers
for i, paper1 in enumerate(papers):
    for paper2 in papers:
        if paper1 == paper2:
            continue
            
        # Take one chunk from paper1
        import random
        chunk1 = random.choice(chunks_by_paper[paper1][:5])
        chunk2 = random.choice(chunks_by_paper[paper2][:5])
        
        # Generate a question that BOTH chunks could answer
        prompt = f"""You are creating training data for a RAG system.

Given these two chunks from DIFFERENT research papers, generate 2 questions that require understanding BOTH chunks to answer fully.
The questions should use DIFFERENT vocabulary than the chunks themselves.
Return ONLY a JSON array of 2 question strings.

Chunk from paper 1 ({paper1}):
{chunk1['text'][:300]}

Chunk from paper 2 ({paper2}):
{chunk2['text'][:300]}

Generate questions that connect concepts from both. Use synonyms and paraphrases, not exact words from the chunks.
JSON array:"""

        try:
            result = call_llm(prompt)
            result = result.replace("```json", "").replace("```", "").strip()
            start = result.find("[")
            end = result.rfind("]") + 1
            if start != -1 and end > start:
                questions = json.loads(result[start:end])
                
                for question in questions:
                    if isinstance(question, str) and len(question) > 15:
                        # Both chunks are positives for this cross-paper question
                        cross_paper_pairs.append({
                            "anchor": question,
                            "positive": chunk1["text"],
                            "positive_source": paper1,
                            "chunk_type": "cross_paper"
                        })
                        cross_paper_pairs.append({
                            "anchor": question,
                            "positive": chunk2["text"],
                            "positive_source": paper2,
                            "chunk_type": "cross_paper"
                        })
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if len(cross_paper_pairs) >= 400:
            break
    
    print(f"  Paper {i+1}/{len(papers)} done — {len(cross_paper_pairs)} cross-paper pairs so far")
    time.sleep(2)  # Rate limit
    
    if len(cross_paper_pairs) >= 400:
        break

print(f"\nTotal cross-paper pairs: {len(cross_paper_pairs)}")

# Combine with original pairs
print("Loading original training pairs...")
with open("data/processed/training_pairs.json", "r", encoding="utf-8") as f:
    original_pairs = json.load(f)

combined = original_pairs + cross_paper_pairs
print(f"Combined total: {len(combined)} pairs")

with open("data/processed/training_pairs_v2.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print("Saved to data/processed/training_pairs_v2.json")