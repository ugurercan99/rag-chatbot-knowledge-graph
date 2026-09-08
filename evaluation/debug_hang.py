import sys, os
from app import chatbot
import chromadb
from chromadb.utils import embedding_functions

print("Testing chromadb client...")
client = chromadb.PersistentClient(path=chatbot.OUTPUT_DIR)
print("Client created.")
collections = client.list_collections()
print("Collections listed:", collections)

try:
    print("Getting collection...")
    print("Imports inside embedding function...")
    collection = client.get_collection(
        name="langchain",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )
    print("Collection retrieved.")
except Exception as e:
    print("Error:", e)
