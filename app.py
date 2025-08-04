from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from datetime import timedelta, datetime
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://user:password@localhost:5432/your_database_name"  # Replace with your actual credentials
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    email = Column(String(120), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    google_id = Column(String(255), unique=True, nullable=True)

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    price = Column(Integer, nullable=False)

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    subscription_status = Column(Boolean, default=False)

# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://quizoq.netlify.app",
    "http://localhost",
    "https://weegek.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set in environment variables.")
client = Groq(api_key=api_key)

# Security
SECRET_KEY = os.getenv("SECRET_KEY") or "your-secret-key"  # Replace with a strong secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID") or "your-google-client-id" # Replace with your Google Client ID

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

class QuizRequest(BaseModel):
    text: str
    num_questions: int = 4
    difficulty: str = "medium"
    num_options: int = 4
    question_type: str = "mcq"  # "mcq", "truefalse", "mixed"

# Password Hashing
def hash_password(password: str):
    return generate_password_hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return check_password_hash(hashed_password, plain_password)

# JWT Functions
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user

# API Endpoints
@app.get("/")
def ping():
    return {"ping": "pong"}

@app.post("/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user with name
    hashed_password = hash_password(user.password)
    db_user = User(
        name=user.name,
        email=user.email, 
        password_hash=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token_data = {"sub": db_user.email}
    access_token = create_access_token(data=access_token_data)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token_data = {"sub": user.email}
    access_token = create_access_token(data=access_token_data, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

class GoogleLoginRequest(BaseModel):
    token_id: str

@app.post("/google-login", response_model=Token)
async def google_login(login_request: GoogleLoginRequest, db: Session = Depends(get_db)):
    token_id = login_request.token_id
    try:
        idinfo = id_token.verify_oauth2_token(token_id, google_requests.Request(), GOOGLE_CLIENT_ID)
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise HTTPException(status_code=401, detail="Wrong issuer")

        email = idinfo.get('email')
        google_id = idinfo.get('sub')
        name = idinfo.get('name')

        # First try to find user by Google ID
        user = db.query(User).filter(User.google_id == google_id).first()
        
        # If not found by Google ID, try to find by email
        if not user:
            user = db.query(User).filter(User.email == email).first()
            
            # If user exists with this email but no Google ID, update with Google ID
            if user and not user.google_id:
                user.google_id = google_id
                if name and not user.name:  # Update name if not set
                    user.name = name
                db.commit()
            # If no user exists with this email, create a new one
            elif not user:
                hashed_password = hash_password(google_id) # Using Google ID as password for OAuth users
                user = User(email=email, name=name, google_id=google_id, password_hash=hashed_password)
                db.add(user)
                db.commit()
                db.refresh(user)

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token_data = {"sub": user.email}
        access_token = create_access_token(data=access_token_data, expires_delta=access_token_expires)
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")

@app.post("/generate-quiz")
async def generate_quiz(req: QuizRequest, current_user: User = Depends(get_current_active_user)):
    input_limit = 1000  # Set the input limit for non-subscribed users

    if not current_user.subscription_status and len(req.text) > input_limit:
        raise HTTPException(status_code=403, detail=f"Input text limited to {input_limit} characters for non-subscribed users.")

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