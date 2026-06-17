import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import load_all_chunks, hybrid_rerank_search

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()

def classify_question(question: str) -> str:
    prompt = f"""Classify this question into one of these types:
- SIMPLE: straightforward factual question needing one search
- COMPARISON: asks to compare two or more things
- MULTI_PART: has multiple distinct sub-questions in one
- COMPLEX: requires information from multiple sections or topics
- CROSS_PAPER: explicitly asks to connect or compare ideas across different papers

Question: {question}

Return ONLY one word: SIMPLE, COMPARISON, MULTI_PART, COMPLEX, or CROSS_PAPER"""
    result = call_llm(prompt).strip().upper()
    for valid in ["SIMPLE", "COMPARISON", "MULTI_PART", "COMPLEX", "CROSS_PAPER"]:
        if valid in result:
            return valid
    return "SIMPLE"

def plan_queries(question: str, question_type: str) -> list:
    if question_type == "SIMPLE":
        return [question]
    
    if question_type == "CROSS_PAPER":
        prompt = f"""This question requires connecting ideas across multiple research papers.
Break it into 4-5 specific search queries that will find relevant content from DIFFERENT papers.
Each query should target a different paper or a different angle of the same concept.
Use varied vocabulary — synonyms and paraphrases of the key concepts.
Return ONLY a JSON array of strings.

Question: {question}

Example: ["RAG pipeline limitations technical domains", "conventional retrieval system challenges", 
"AI tool impact on cognitive skills", "critical thinking AI assistance research"]

JSON array:"""
    else:
        prompt = f"""Break this question into 2-3 specific search queries.
Each query should target a different aspect of the question.
Return ONLY a JSON array of strings.

Question: {question}
Question type: {question_type}

JSON array:"""
    
    result = call_llm(prompt)
    try:
        result = result.replace("```json", "").replace("```", "").strip()
        start = result.find("[")
        end = result.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(result[start:end])
            if isinstance(queries, list) and len(queries) > 0:
                print(f"[Agent] Planned {len(queries)} sub-queries")
                return queries
    except:
        pass
    return [question]

def retrieve_chunks(queries: list, all_chunks: list) -> list:
    all_retrieved = []
    seen_texts = set()
    for query in queries:
        print(f"[Agent] Searching: {query[:60]}...")
        chunks = hybrid_rerank_search(query, all_chunks, top_k=4)
        for chunk in chunks:
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_retrieved.append(chunk)
    print(f"[Agent] Total unique chunks retrieved: {len(all_retrieved)}")
    return all_retrieved

def retrieve_cross_paper(queries: list, all_chunks: list) -> list:
    """Special retrieval for cross-paper questions — ensures chunks from multiple papers"""
    all_retrieved = []
    seen_texts = set()
    seen_sources = set()
    
    for query in queries:
        print(f"[Agent] Cross-paper search: {query[:60]}...")
        # Get more chunks per query for cross-paper
        chunks = hybrid_rerank_search(query, all_chunks, top_k=7)
        for chunk in chunks:
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_retrieved.append(chunk)
                seen_sources.add(chunk["source"])
    
    print(f"[Agent] Retrieved from {len(seen_sources)} different papers: {seen_sources}")
    print(f"[Agent] Total unique chunks: {len(all_retrieved)}")
    return all_retrieved

def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    prompt = f"""You are an expert AI researcher. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    return call_llm(prompt)

def generate_cross_paper_answer(question: str, chunks: list) -> str:
    """Special answer generation for cross-paper questions"""
    # Group chunks by source paper
    by_paper = {}
    for chunk in chunks:
        source = chunk["source"]
        if source not in by_paper:
            by_paper[source] = []
        by_paper[source].append(chunk["text"])
    
    # Build context organized by paper
    context_parts = []
    for paper, texts in by_paper.items():
        context_parts.append(f"=== From {paper} ===")
        context_parts.append("\n".join(texts[:2]))  # Max 2 chunks per paper
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are an expert AI researcher tasked with connecting ideas across multiple research papers.

The context below contains excerpts from DIFFERENT papers. Your job is to:
1. Identify what each paper says about the topic
2. Find the conceptual connections between papers
3. Synthesize a unified answer that draws from all relevant papers

Context from multiple papers:
{context}

Question: {question}

Instructions:
- Explicitly mention which paper each insight comes from
- Draw clear connections between the papers
- If papers discuss the same concept with different terminology, explain that
- Be specific with evidence from the context

Answer:"""
    return call_llm(prompt)

def is_answer_complete(question: str, answer: str) -> bool:
    prompt = f"""Review this answer carefully.

Question: {question}
Answer: {answer}

Check for these specific issues:
1. Placeholder phrases like "not specified", "not provided", "not mentioned"?
2. Vague references without actual content?
3. Missing specific values, formulas, or steps where required?
4. For cross-paper questions — does it actually reference multiple papers?

If ANY issues exist, answer "no". Otherwise answer "yes".

Answer (yes/no):"""
    result = call_llm(prompt).lower()
    complete = "yes" in result and "no" not in result
    print(f"[Agent] Answer complete: {complete}")
    return complete

def refine_queries(question: str, answer: str) -> list:
    prompt = f"""The answer below is incomplete.

Question: {question}
Incomplete answer: {answer}

Identify SPECIFIC missing pieces and generate 1-2 targeted search queries to find them.
Return ONLY a JSON array of search queries.

JSON array:"""
    result = call_llm(prompt)
    try:
        result = result.replace("```json", "").replace("```", "").strip()
        start = result.find("[")
        end = result.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(result[start:end])
            if isinstance(queries, list) and len(queries) > 0:
                return queries
    except:
        pass
    return [question]

def ask_agent(question: str, all_chunks: list) -> dict:
    print(f"\n{'='*60}")
    print(f"[Agent] Processing: {question[:80]}...")
    
    question_type = classify_question(question)
    queries = plan_queries(question, question_type)
    
    # Use special cross-paper retrieval if needed
    if question_type == "CROSS_PAPER":
        chunks = retrieve_cross_paper(queries, all_chunks)
        answer = generate_cross_paper_answer(question, chunks)
    else:
        chunks = retrieve_chunks(queries, all_chunks)
        answer = generate_answer(question, chunks)
    
    iterations = 1
    while iterations < 3:
        if is_answer_complete(question, answer):
            break
        print(f"[Agent] Refining — iteration {iterations + 1}")
        new_queries = refine_queries(question, answer)
        new_chunks = retrieve_chunks(new_queries, all_chunks)
        seen = {c["text"] for c in chunks}
        for c in new_chunks:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])
        answer = generate_answer(question, chunks)
        iterations += 1
    
    return {
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "sub_queries": queries,
        "chunks_used": len(chunks),
        "iterations": iterations
    }

if __name__ == "__main__":
    print("Loading chunks...")
    all_chunks = load_all_chunks()
    
    test_questions = [
        "What is the UCT algorithm, what is its formula, and how does RAG-Star use it for node selection?",
        "Both the Telco-RAG paper and the Critical Thinking paper discuss how AI systems can fail to address the real underlying problem. What limitation does Telco-RAG identify in conventional RAG setups, and what analogous concern does the Critical Thinking paper raise about AI tools improving task outputs?"
    ]
    
    for question in test_questions:
        result = ask_agent(question, all_chunks)
        print(f"\nQuestion Type: {result['question_type']}")
        print(f"Sub-queries: {result['sub_queries']}")
        print(f"Chunks used: {result['chunks_used']}")
        print(f"Iterations: {result['iterations']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n{'='*60}")