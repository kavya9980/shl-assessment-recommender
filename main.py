import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load assessment data
with open("assessments.json", "r") as f:
    assessments = json.load(f)

# Load precomputed embeddings
embeddings = np.load("embeddings.npy")

# Extract descriptions safely
corpus = [a.get("description", "") for a in assessments]

# FastAPI app setup
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

# API route
@app.post("/recommend")
async def recommend(query_request: QueryRequest):
    query_embedding = get_embedding(query_request.query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-query_request.top_k:][::-1]
    results = [assessments[i] for i in top_indices]
    return results

# Embedding helper
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
