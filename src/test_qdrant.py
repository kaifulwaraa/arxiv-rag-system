from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print(f"URL: {url}")
print(f"API Key: {api_key[:10]}...")

client = QdrantClient(url=url, api_key=api_key)
collections = client.get_collections()
print("Connected successfully!")
print(collections)