from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Ollama API")

OLLAMA_URL = "http://localhost:11434/api/generate"

class PromptRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5-coder:7b"  # change to your local model

@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": request.model,
            "prompt": request.prompt,
            "stream": False
        })
        response.raise_for_status()
        return {"response": response.json().get("response", "")}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

@app.get("/")
def root():
    return {"status": "running"}
