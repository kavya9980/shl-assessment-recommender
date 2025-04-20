import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
import uvicorn
import os

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load assessment data
with open("assessments.json", "r", encoding="utf-8") as f:
    assessments = json.load(f)

# Handle 'description' and 'Description' keys
corpus = []
valid_indices = []
for i, a in enumerate(assessments):
    desc = a.get("description") or a.get("Description")
    if desc:
        corpus.append(desc)
        valid_indices.append(i)

# Precompute embeddings
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# FastAPI app
app = FastAPI()

# Enable CORS
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
        query_embedding = model.encode(request.query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=request.top_k)[0]

        results = []
        for hit in hits:
            item = assessments[valid_indices[hit['corpus_id']]]
            results.append({
                "name": item.get("name") or item.get("Assessment Name"),
                "description": item.get("description") or item.get("Description"),
                "duration": item.get("duration_minutes") or item.get("Assessment Length (Minutes)"),
                "remote": item.get("remote_testing") or item.get("Remote Testing Support"),
                "adaptive": item.get("adaptive") or item.get("IRT Support"),
                "test_type": item.get("test_type") or item.get("Test Type"),
                "url": item.get("url", "#")
            })

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
