from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables from .env if present (local dev)
load_dotenv()

# Now get the API key from environment (works for both prod and local)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set in environment variables.")

client = Groq(api_key=api_key)
app = FastAPI()
origins = [
    "http://localhost:3000",  # Allow React dev server
    "http://localhost",       # Optional: allow other localhost origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Only allow listed origins
    allow_credentials=True,
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)

class QuizRequest(BaseModel):
    text: str

@app.post("/generate-quiz")
async def generate_quiz(req: QuizRequest):
    paragraphs = [p.strip() for p in req.text.split('\n\n') if p.strip()]
    if not paragraphs:
        return {"quiz": []}

    joined_paragraphs = "\n\n---\n\n".join(paragraphs)
    prompt = f"""
You are an intelligent exam assistant. From the paragraphs below, generate 2 multiple-choice questions (MCQs) **for each paragraph**.

Each MCQ must have:
- A question based on facts in the paragraph
- Four options (A, B, C, D)
- Exactly one correct answer
- Clearly labeled answer (e.g., Answer: A)

Format strictly like:
Question: ...
Options:
A ...
B ...
C ...
D ...
Answer: ...

Paragraphs:
{joined_paragraphs}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048
    )

    # Parse the response into a list of questions (optional: improve formatting as needed)
    quiz_text = response.choices[0].message.content.strip()
    return {"quiz": quiz_text}

# To run the server:
# uvicorn your_filename:app --reload --port 8000