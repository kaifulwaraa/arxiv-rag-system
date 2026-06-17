import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

print("Exporting chunks from Qdrant...")

all_points = []
offset = None
batch_size = 100

while True:
    results, next_offset = client.scroll(
        collection_name="arxiv_papers",
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    for point in results:
        all_points.append({
            "id": str(point.id),
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", ""),
            "chunk_type": point.payload.get("chunk_type", "text")
        })
    
    print(f"  Fetched {len(all_points)} chunks so far...")
    
    if next_offset is None:
        break
    offset = next_offset

print(f"\nTotal chunks exported: {len(all_points)}")

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/chunks_export.json", "w", encoding="utf-8") as f:
    json.dump(all_points, f, indent=2, ensure_ascii=False)

print("Saved to data/processed/chunks_export.json")

# Quick stats
text_chunks = sum(1 for p in all_points if p["chunk_type"] == "text")
table_chunks = sum(1 for p in all_points if p["chunk_type"] == "table")
figure_chunks = sum(1 for p in all_points if p["chunk_type"] == "figure_caption")
print(f"\nBreakdown:")
print(f"  Text chunks: {text_chunks}")
print(f"  Table chunks: {table_chunks}")
print(f"  Figure chunks: {figure_chunks}")