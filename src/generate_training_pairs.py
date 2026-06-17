import json
import random
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_questions_for_chunk(chunk_text: str, chunk_type: str, source: str) -> list:
    prompt = f"""You are creating training data for a RAG system about AI research papers.

Given this chunk from a research paper, generate 3 different questions that this chunk directly answers.
Questions should vary: one factual, one conceptual, one applied.
Return ONLY a JSON array of 3 strings, nothing else.

Chunk type: {chunk_type}
Source: {source}
Chunk text: {chunk_text[:600]}

Example output: ["What is X?", "How does Y work?", "Why does Z matter?"]

JSON array:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        questions = json.loads(result)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str) and len(q) > 10]
    except Exception as e:
        print(f"  Error generating questions: {e}")
    return []

def create_hard_negatives(all_chunks: list, current_source: str, n: int = 2) -> list:
    """Get chunks from DIFFERENT papers as hard negatives"""
    different_paper_chunks = [
        c for c in all_chunks 
        if c["source"] != current_source and c["chunk_type"] == "text" and len(c["text"]) > 100
    ]
    return random.sample(different_paper_chunks, min(n, len(different_paper_chunks)))

def main():
    print("Loading exported chunks...")
    with open("data/processed/chunks_export.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Filter good chunks — long enough to generate meaningful questions
    text_chunks = [c for c in all_chunks if c["chunk_type"] == "text" and len(c["text"]) > 200]
    table_chunks = [c for c in all_chunks if c["chunk_type"] == "table"]
    figure_chunks = [c for c in all_chunks if c["chunk_type"] == "figure_caption"]

    print(f"Usable text chunks: {len(text_chunks)}")
    print(f"Table chunks: {len(table_chunks)}")
    print(f"Figure chunks: {len(figure_chunks)}")

    # Sample chunks to generate pairs from
    # 80 text + 20 table + 20 figure = 120 chunks x 3 questions = ~360 pairs
    selected_text = random.sample(text_chunks, min(80, len(text_chunks)))
    selected_table = random.sample(table_chunks, min(20, len(table_chunks)))
    selected_figure = random.sample(figure_chunks, min(20, len(figure_chunks)))
    selected_chunks = selected_text + selected_table + selected_figure

    print(f"\nGenerating questions for {len(selected_chunks)} chunks...")
    print("This will take ~15-20 minutes due to Groq rate limits...\n")

    training_pairs = []
    
    for i, chunk in enumerate(selected_chunks):
        print(f"[{i+1}/{len(selected_chunks)}] {chunk['source'][:40]} ({chunk['chunk_type']})")
        
        questions = generate_questions_for_chunk(
            chunk["text"], 
            chunk["chunk_type"],
            chunk["source"]
        )
        
        for question in questions:
            # Positive pair — question matches this chunk
            pair = {
                "anchor": question,
                "positive": chunk["text"],
                "positive_source": chunk["source"],
                "chunk_type": chunk["chunk_type"]
            }
            
            # Add hard negatives — chunks from different papers
            negatives = create_hard_negatives(all_chunks, chunk["source"], n=2)
            pair["negatives"] = [n["text"] for n in negatives]
            
            training_pairs.append(pair)
        
        # Rate limit pause every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"  → {len(training_pairs)} pairs so far, pausing 5 seconds...")
            import time
            time.sleep(5)

    print(f"\nTotal training pairs generated: {len(training_pairs)}")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/training_pairs.json", "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)

    print("Saved to data/processed/training_pairs.json")
    
    # Show sample
    if training_pairs:
        print("\nSample pair:")
        sample = training_pairs[0]
        print(f"  Anchor: {sample['anchor']}")
        print(f"  Positive: {sample['positive'][:100]}...")
        print(f"  Negatives: {len(sample['negatives'])} chunks from other papers")

if __name__ == "__main__":
    main()