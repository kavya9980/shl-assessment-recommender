from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

# Load assessment data
with open("assessments.json", "r") as f:
    assessments = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings
corpus = [a["description"] for a in assessments]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
async def recommend(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=request.top_k)
    hits = hits[0]

    results = []
    for hit in hits:
        item = assessments[hit['corpus_id']]
        results.append({
            "name": item["name"],
            "url": item["url"],
            "description": item["description"],
            "duration": item.get("duration"),
            "adaptive": item.get("adaptive"),
            "remote": item.get("remote"),
            "test_type": item.get("test_type")
        })

    return {"results": results}
