import os
import psycopg2
import requests
import json
import numpy as np

# Configuration
#SERVER_URL = 'http://localhost:8080/embed'
DB_CONFIG = {
    'dbname': os.environ.get('DB_NAME', 'market_simulation'),
    'user': os.environ.get('DB_USER', 'db_user'),
    'password': os.environ.get('DB_PASSWORD', 'db_pwd@123'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5433')
}

# Sample documents
documents = [
    "Artificial Intelligence is transforming the world.",
    "Machine Learning enables computers to learn from data.",
    "Deep Learning is a subset of Machine Learning.",
    "Natural Language Processing allows machines to understand human language.",
    "Neural networks are the foundation of Deep Learning."
]

# Function to get embedding for a document
def get_embedding(text):
    response = requests.post(
        SERVER_URL,
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'inputs': text})
    )
    print(f"Response Status: {response.status_code}")
    print(f"Response Content: {response.text}")
    
    if response.status_code == 200:
        try:
            embedding = response.json()
            if isinstance(embedding, list):
                return embedding[0]  
            elif isinstance(embedding, dict) and 'embedding' in embedding:
                return embedding['embedding']
            else:
                print("Unexpected response format:", embedding)
                return None
        except ValueError as e:
            print(f"Failed to parse JSON response: {e}")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Connect to the database
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# Ensure the vector extension is enabled
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# Create the new table with vector(768)
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory_embeddings_768 (
    id SERIAL PRIMARY KEY,
    agent_id UUID,
    embedding vector(768),
    memory_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

# Insert documents and their embeddings into the new table
for doc in documents:
    embedding = get_embedding(doc)
    if embedding and len(embedding) == 768:
        agent_id = '00000000-0000-0000-0000-000000000000'
        memory_data = {'text': doc}
        cursor.execute("""
            INSERT INTO memory_embeddings_768 (agent_id, embedding, memory_data)
            VALUES (%s, %s, %s::jsonb)
        """, (agent_id, embedding, json.dumps(memory_data)))
        conn.commit()
    else:
        print("Embedding size mismatch or retrieval error.")

# Query for similar documents
query_text = "What is deep learning?"
query_embedding = get_embedding(query_text)

if query_embedding and len(query_embedding) == 768:
    cursor.execute("""
    SELECT id, memory_data, embedding <-> %s::vector AS distance
    FROM memory_embeddings_768
    ORDER BY distance
    LIMIT 3
    """, (query_embedding,))
    results = cursor.fetchall()
    print("\nTop 3 similar documents:")
    for result in results:
        print(f"ID: {result[0]}, Memory: {result[1]['text']}, Distance: {result[2]}")

# Clean up
cursor.close()
conn.close()
