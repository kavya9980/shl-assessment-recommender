import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import openai

# Setup OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load assessment data and embeddings
with open("assessments.json", "r", encoding="utf-8") as f:
    assessments = json.load(f)

embeddings = np.load("embeddings.npy")

# Setup FastAPI app
app = FastAPI()

# CORS (allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/recommend")
async def recommend(request: QueryRequest):
    try:
        # Embed the query using OpenAI
        response = openai.Embedding.create(
            input=[request.query],
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(response["data"][0]["embedding"])

        # Compute cosine similarity
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(embeddings, query_embedding) / norms
        top_indices = np.argsort(similarities)[::-1][:request.top_k]

        # Return top K results
        recommendations = [assessments[i] for i in top_indices]
        return {"recommendations": recommendations}

    except Exception as e:
        print("Error occurred:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
