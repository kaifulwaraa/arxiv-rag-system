import os
import re
import arxiv
import fitz  # pymupdf
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import time

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ----------------------------
# Paths
# ----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Step 1: Download papers from ArXiv
# ----------------------------
def download_papers(query="RAG retrieval augmented generation LLM", max_results=20):
    print(f"Searching ArXiv for: {query}")
    import requests

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded = []
    for paper in client.results(search):
        paper_id = paper.entry_id.split("/")[-1]
        filename = paper_id.replace("/", "_") + ".pdf"
        filepath = RAW_DIR / filename

        if filepath.exists():
            print(f"Already exists: {filename}")
            downloaded.append(filepath)
            continue

        try:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}"
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
                downloaded.append(filepath)
            else:
                print(f"Failed: {filename} — status {response.status_code}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    print(f"\nTotal papers downloaded: {len(downloaded)}")
    return downloaded

# ----------------------------
# Step 2: Extract text from PDFs
# ----------------------------
def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

# ----------------------------
# Step 2b: Extract tables from PDFs
# ----------------------------
def extract_tables_from_pdf(filepath, paper_name):
    table_chunks = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    # Clean None values
                    cleaned = []
                    for row in table:
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        cleaned.append(cleaned_row)

                    headers = cleaned[0]
                    rows = cleaned[1:]

                    # Skip if headers are all empty
                    if all(h == "" for h in headers):
                        continue

                    # Build readable table text with explicit row/column labels
                    table_text = f"[TABLE from page {page_num}]\n"
                    table_text += "Headers: " + " | ".join(headers) + "\n"
                    table_text += "-" * 60 + "\n"
                    for row in rows:
                        if all(cell == "" for cell in row):
                            continue
                        labeled_cells = []
                        for col_idx, cell in enumerate(row):
                            if cell:
                                header = headers[col_idx] if col_idx < len(headers) and headers[col_idx] else f"Col{col_idx+1}"
                                labeled_cells.append(f"[{header}: {cell}]")
                        if labeled_cells:
                            table_text += " | ".join(labeled_cells) + "\n"

                    if len(table_text.strip()) > 100:
                        table_chunks.append({
                            "text": table_text.strip(),
                            "source": paper_name,
                            "chunk_type": "table"
                        })
                        print(f"  Extracted table {table_idx+1} from page {page_num}")

    except Exception as e:
        print(f"  Table extraction failed: {e}")

    return table_chunks
# ----------------------------
# Step 2c: Extract figure captions from PDFs
# ----------------------------
def extract_figures_from_pdf(filepath, paper_name):
    figure_chunks = []
    try:
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()

        # Match figure captions like "Figure 1:", "Fig. 2 —", etc.
        pattern = r'(Fig(?:ure)?\.?\s*\d+[a-z]?[\s\.:—\-]+[^\n]{20,300})'
        matches = re.findall(pattern, full_text, re.IGNORECASE)

        for idx, caption in enumerate(matches):
            caption = caption.strip()
            if len(caption) < 30:
                continue
            figure_chunks.append({
                "text": f"[FIGURE CAPTION]\n{caption}",
                "source": paper_name,
                "chunk_type": "figure_caption"
            })
            print(f"  Extracted figure caption {idx+1}: {caption[:80]}...")

    except Exception as e:
        print(f"  Figure extraction failed: {e}")

    return figure_chunks

# ----------------------------
# Step 3: Chunk the text
# ----------------------------
def chunk_text(text, chunk_size=512, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# ----------------------------
# Step 4: Store in Qdrant
# ----------------------------
def store_in_qdrant(chunks_data, collection_name="arxiv_papers"):
    """
    chunks_data: list of dicts with keys: text, source, chunk_type
    """
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name}")

    points = []
    for chunk in chunks_data:
        if not chunk["text"].strip():
            continue
        embedding = embedder.encode(chunk["text"]).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_type": chunk.get("chunk_type", "text")
            }
        ))

    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            print(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Batch failed, retrying in 3 seconds... {e}")
            time.sleep(3)
            client.upsert(collection_name=collection_name, points=batch)

    print(f"Stored {len(points)} chunks")
    return len(points)

# ----------------------------
# Reset collection (fresh start)
# ----------------------------
def reset_collection(collection_name="arxiv_papers"):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Created fresh collection: {collection_name}")

# ----------------------------
# Main Pipeline
# ----------------------------
def run_ingestion():
    print("=== Starting Multimodal Ingestion Pipeline ===\n")
    print("This will reset your Qdrant collection and re-index all papers")
    print("with text + tables + figure captions.\n")

    confirm = input("Type 'yes' to continue: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    # Reset collection for fresh start
    print("\nResetting Qdrant collection...")
    reset_collection()

    # Download papers
    papers = download_papers(max_results=20)
    if not papers:
        print("No papers found in data/raw/. Check your files.")
        return

    # Clear processed tracker
    processed_file = PROCESSED_DIR / "processed.txt"
    if processed_file.exists():
        processed_file.unlink()

    total_text_chunks = 0
    total_table_chunks = 0
    total_figure_chunks = 0

    for paper_path in papers:
        print(f"\n{'='*50}")
        print(f"Processing: {paper_path.name}")
        print(f"{'='*50}")

        all_chunks = []

        # --- Text ---
        print("  Extracting text...")
        text = extract_text_from_pdf(paper_path)
        if text.strip():
            text_chunks = chunk_text(text)
            for chunk in text_chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": paper_path.name,
                    "chunk_type": "text"
                })
            total_text_chunks += len(text_chunks)
            print(f"  Text chunks: {len(text_chunks)}")

        # --- Tables ---
        print("  Extracting tables...")
        table_chunks = extract_tables_from_pdf(paper_path, paper_path.name)
        all_chunks.extend(table_chunks)
        total_table_chunks += len(table_chunks)
        print(f"  Table chunks: {len(table_chunks)}")

        # --- Figure captions ---
        print("  Extracting figure captions...")
        figure_chunks = extract_figures_from_pdf(paper_path, paper_path.name)
        all_chunks.extend(figure_chunks)
        total_figure_chunks += len(figure_chunks)
        print(f"  Figure caption chunks: {len(figure_chunks)}")

        # --- Store ---
        if all_chunks:
            stored = store_in_qdrant(all_chunks)
            print(f"  ✅ {stored} total chunks stored for {paper_path.name}")

        with open(processed_file, "a") as f:
            f.write(paper_path.name + "\n")

    # Final summary
    total = total_text_chunks + total_table_chunks + total_figure_chunks
    print(f"\n{'='*50}")
    print("=== INGESTION COMPLETE ===")
    print(f"Text chunks:           {total_text_chunks}")
    print(f"Table chunks:          {total_table_chunks}")
    print(f"Figure caption chunks: {total_figure_chunks}")
    print(f"Total chunks indexed:  {total}")
    print(f"{'='*50}")
    print("\nUpdate the chunk count in app.py sidebar with the new total.")

if __name__ == "__main__":
    run_ingestion()