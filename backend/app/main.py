from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SignSetu API", version="0.1.0")

# Allow the React frontend (running on port 5173) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "SignSetu API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}