from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://quizoq.netlify.app",
    "http://localhost"
]
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set in environment variables.")

client = Groq(api_key=api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    text: str
    num_questions: int = 4
    difficulty: str = "medium"
    num_options: int = 4
    question_type: str = "mcq"  # "mcq", "truefalse", "mixed"

@app.get("/")
def ping():
    return {"ping": "pong"}

@app.post("/generate-quiz")
async def generate_quiz(req: QuizRequest):
    paragraphs = [p.strip() for p in req.text.split('\n\n') if p.strip()]
    if not paragraphs:
        return {"quiz": []}

    joined_paragraphs = "\n\n---\n\n".join(paragraphs)

    # Build prompt based on question_type and num_options
    if req.question_type == "truefalse":
        qtype_str = "true/false questions"
        format_str = """
Question: ...
Options:
A) True
B) False
Answer: ...
"""
    elif req.question_type == "mcq":
        qtype_str = f"multiple-choice questions (MCQs) with {req.num_options} options"
        options_letters = [chr(65 + i) for i in range(req.num_options)]
        options_format = '\n'.join([f"{l}) ..." for l in options_letters])
        format_str = f"""
Question: ...
Options:
{options_format}
Answer: ...
"""
    elif req.question_type == "mcq_multiple":
        qtype_str = "multiple-choice questions (MCQs) with multiple correct answers. Indicate all correct answers."
        options_letters = [chr(65 + i) for i in range(req.num_options)]
        options_format = '\n'.join([f"{l}) ..." for l in options_letters])
        format_str = f"""
Question: ...
Options:
{options_format}
Answer: ... (e.g., A, C)
"""
    elif req.question_type == "fillblanks":
        qtype_str = f"fill in the blank questions."
        format_str = """
Question: ... (The blank should be represented by an underscore or bracketed empty space like [_] or [])
Answer: ...
"""
    elif req.question_type == "faq":
        qtype_str = "frequently asked questions (FAQ)."
        format_str = """
Question: ...
Answer: ...
"""
    elif req.question_type == "short":
        qtype_str = "short answer questions."
        format_str = """
Question: ...
Answer: ...
"""
    elif req.question_type == "higherorder":
        qtype_str = "higher-order thinking questions requiring reasoning and synthesis."
        format_str = """
Question: ...
Answer: ...
"""
    else:
        qtype_str = f"a mix of MCQs (with {req.num_options} options) and true/false questions"
        options_letters = [chr(65 + i) for i in range(req.num_options)]
        options_format = '\n'.join([f"{l}) ..." for l in options_letters])
        format_str = f"""
Question: ...
Options (for MCQ):
{options_format}
or
Options (for True/False):
A) True
B) False
Answer: ...
"""

    prompt = f"""
You are an intelligent exam assistant. From the paragraphs below, generate a total of {req.num_questions} {qtype_str} at **{req.difficulty}** difficulty level. Distribute the questions across the paragraphs as evenly as possible.

Each question must have:
- A question based on facts in the paragraph
- Options as specified below (if applicable)
- Exactly one correct answer (unless the question type is 'mcq_multiple')
- Clearly labeled answer(s) (e.g., Answer: A or Answer: A, C)

Format strictly like:
{format_str}

Paragraphs:
{joined_paragraphs}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048
    )

    quiz_text = response.choices[0].message.content.strip()
    return {"quiz": quiz_text}

# To run the server:
# uvicorn app:app --reload --port 8000