import os, tempfile, shutil
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ---------------- Config ----------------
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")            # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")        # int8 (CPU friendly), float16/float32 if you have GPU
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="Faster-Whisper STT API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup for speed
model: WhisperModel | None = None

@app.on_event("startup")
def _load_model():
    global model
    model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)
    # You’ll see a download the first time; afterwards it uses cache

class TranscribeParams(BaseModel):
    task: str = "transcribe"      # or "translate" to English
    language: Optional[str] = None # e.g., "en", "es"; None = auto-detect
    vad_filter: bool = True        # basic voice activity detection

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscribeResponse(BaseModel):
    language: str
    duration: float
    text: str
    segments: List[Segment]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...), task: str = "transcribe", language: str | None = None, vad_filter: bool = True):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Save upload to a temporary file so faster-whisper/ffmpeg can read it
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        segments_iter, info = model.transcribe(
            tmp_path,
            task=task,
            language=language,
            vad_filter=vad_filter,
        )

        segments = []
        texts = []
        for seg in segments_iter:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
            texts.append(seg.text)

        return TranscribeResponse(
            language=info.language,
            duration=info.duration,
            text=" ".join(t.strip() for t in texts).strip(),
            segments=[Segment(**s) for s in segments],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
