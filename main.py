from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# Allow CORS for all origins (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variable for API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data and embeddings
with open("assessments.json", "r") as f:
    data = json.load(f)

embedding_matrix = np.load("embeddings.npy")

texts = [item.get("name", "") + " - " + item.get("description", "") for item in data]

@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API is running."}

@app.post("/recommend")
async def recommend(request: Request):
    payload = await request.json()
    query = payload.get("query", "")

    if not query:
        return {"error": "Query cannot be empty."}

    # Generate embedding for query
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]

    # Get top 10 indices
    top_indices = similarities.argsort()[::-1][:10]

    recommendations = []
    for idx in top_indices:
        item = data[idx]
        recommendations.append({
            "name": item.get("name", item.get("Assessment Name", "")),
            "description": item.get("description", item.get("Description", "")),
            "job_levels": item.get("job_levels", item.get("Job Levels", [])),
            "duration_minutes": item.get("duration_minutes", item.get("Assessment Length (Minutes)", "")),
            "remote_testing": item.get("remote_testing", item.get("Remote Testing Support", "")),
            "test_type": item.get("test_type", item.get("Test Type", "")),
        })

    return {"results": recommendations}
