
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from config import UPLOAD_DIR, RESULT_DIR
from routers import watercolor_router

app = FastAPI(title="AI Style Transformer - Watercolor")

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure media directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Routers
app.include_router(watercolor_router.router, prefix="/api", tags=["watercolor"])

@app.get("/")
def root():
    return {"message": "AI Style Transformer backend running!"}
