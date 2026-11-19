from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .collect import router as collect_router
from .inference import SignPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = SignPredictor()
        print("Model loaded.")
    except Exception as err:  # keep the API alive even if the model is missing
        app.state.predictor = None
        print(f"Model NOT loaded: {err}")
    yield


app = FastAPI(title="SignSetu API", version="0.3.0", lifespan=lifespan)

# Allow the React frontend (running on port 5173) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collect_router)


class PredictRequest(BaseModel):
    frames: list[list[list[list[float]]]]


@app.get("/")
def root():
    return {"message": "SignSetu API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = app.state.predictor.predict(req.frames)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    if result is None:
        return {"label": None, "confidence": 0.0, "top3": []}
    return result