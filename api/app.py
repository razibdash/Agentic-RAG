from fastapi import FastAPI, Request
from pydantic import BaseModel
from graph import compiled_graph
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic RAG API")

# Optional: Allow frontend apps to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        result = compiled_graph.invoke({"question": request.question})
        return {
            "question": request.question,
            "answer": result.get("answer", "No answer found."),
            "meta": result
        }
    except Exception as e:
        return {"error": str(e)}
