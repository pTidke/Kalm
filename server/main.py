# main.py
import json
import random
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# =========================
# -------- Config ---------
# =========================
BASE_DIR = Path(__file__).parent

# Put a HF model id in a local ".model" file to override (e.g., "google/gemma-2-2b-it")
GEN_MODEL_ID = (
    BASE_DIR.joinpath(".model").read_text().strip()
    if BASE_DIR.joinpath(".model").exists()
    else "Qwen/Qwen2.5-1.5B-Instruct"
)

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

KB_PATH = BASE_DIR / "KB.json"
INTENTS_PATH = BASE_DIR / "intents.json"

TOP_K_PASSAGES = 6
SIMILARITY_FLOOR = 0.20   # if the top KB sim is below this, we won't force KB context
INTENT_MIN_SCORE = 0.18   # if below this, tag as "neutral"

MAX_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.9

# Prefer Apple GPU (MPS) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =========================
# ------ Data models ------
# =========================
class Msg(BaseModel):
    role: str               # "user" | "assistant" | "system"
    content: str

class ChatIn(BaseModel):
    messages: List[Msg]
    n: Optional[int] = None
    seed: Optional[int] = None

class KBEntry(BaseModel):
    name: str
    description_type: str
    description: str


# =========================
# ------- Safety ----------
# =========================
def crisis_check(text: str) -> bool:
    t = text.lower()
    keywords = [
        "suicide", "kill myself", "end it all", "self-harm", "overdose",
        "cut myself", "no reason to live", "want to die", "hurt myself",
    ]
    return any(k in t for k in keywords)


# =========================
# -- Load KB & intents ----
# =========================
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())

KB_RAW: List[KBEntry] = [KBEntry(**row) for row in load_json(KB_PATH)]
INTENTS: Dict[str, List[str]] = load_json(INTENTS_PATH)


# =========================
# ---- Embedding model ----
# =========================
embedder = SentenceTransformer(EMBED_MODEL_ID)
try:
    embedder.to(DEVICE)
except Exception:
    # Older sentence_transformers may ignore .to(device); CPU fallback is fine.
    pass

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    v = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return v.astype(np.float32)

# Build KB embeddings
KB_TEXTS: List[str] = [
    f"{row.name} | {row.description_type.replace('_', ' ')} | {row.description}"
    for row in KB_RAW
]
KB_EMB: np.ndarray = embed_texts(KB_TEXTS)  # (N, d)


# =========================
# ----- Intent proto ------
# =========================
def build_intent_prototypes() -> Dict[str, np.ndarray]:
    protos: Dict[str, np.ndarray] = {}
    for intent, examples in INTENTS.items():
        if not examples:
            continue
        embs = embed_texts(examples)  # already L2-normalized
        proto = embs.mean(axis=0)
        proto /= (np.linalg.norm(proto) + 1e-8)
        protos[intent] = proto
    return protos

PROTOTYPES: Dict[str, np.ndarray] = build_intent_prototypes()


# =========================
# ------ Retrieval --------
# =========================
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # expects a & b to be row-normalized
    return a @ b.T

def retrieve_kb(user_text: str, top_k: int = TOP_K_PASSAGES):
    q = embed_texts([user_text])  # (1, d)
    sims = cosine(q, KB_EMB)[0]   # (N,)
    idx = np.argsort(-sims)[:top_k]
    results = [(KB_RAW[i], float(sims[i])) for i in idx]
    top_sim = float(sims[idx[0]]) if len(idx) else 0.0
    return results, top_sim

def group_by_topic(results: List[Tuple[KBEntry, float]]) -> Dict[str, List[Tuple[KBEntry, float]]]:
    buckets: Dict[str, List[Tuple[KBEntry, float]]] = {}
    for r, s in results:
        buckets.setdefault(r.name, []).append((r, s))
    return buckets

def pick_best_topic(results: List[Tuple[KBEntry, float]]) -> Tuple[str, List[Tuple[KBEntry, float]]]:
    by = group_by_topic(results)
    best_name, best_score, best_rows = "General Mental Health", -1.0, []
    for name, rows in by.items():
        m = max(s for _, s in rows)
        if m > best_score:
            best_name, best_score, best_rows = name, m, rows
    return best_name, best_rows

def select_sections(rows: List[Tuple[KBEntry, float]], key: str, n: int = 2) -> List[str]:
    pool = [r.description for r, _ in rows if r.description_type == key]
    return pool[:max(0, n)]


# =========================
# -------- Intent ---------
# =========================
def pick_intent(text: str) -> Tuple[str, float]:
    if not PROTOTYPES:
        return "neutral", 0.0
    u = embed_texts([text])[0]  # normalized
    best, best_s = "neutral", -1.0
    for k, proto in PROTOTYPES.items():
        s = float(np.dot(u, proto))
        if s > best_s:
            best, best_s = k, s
    if best_s < INTENT_MIN_SCORE:
        return "neutral", best_s
    return best, best_s


# =========================
# ---- Generation model ---
# =========================
def _load_gen_model_and_tokenizer(model_id: str):
    # On MPS/CUDA prefer fp16 for speed; fall back to fp32 if needed.
    dtype_try = torch.float16 if DEVICE.type in ("mps", "cuda") else torch.float32
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_try,
            trust_remote_code=True,
        )
        mdl.to(DEVICE)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        # Fallback to fp32 (some models don’t love fp16 on MPS/older devices)
        if dtype_try != torch.float32:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            mdl.to(DEVICE)
            mdl.eval()
            return tok, mdl
        raise e

gen_tokenizer, gen_model = _load_gen_model_and_tokenizer(GEN_MODEL_ID)

def apply_chat_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    # Use the tokenizer's chat template if available
    if getattr(gen_tokenizer, "chat_template", None):
        return gen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # Fallback: simple tags
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content'].strip()}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)

def generate_chat(
    messages: List[Dict[str, str]],
    n: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    prompt = apply_chat_template(messages, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = gen_model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_TOKENS,
        num_return_sequences=max(1, n),
        pad_token_id=gen_tokenizer.eos_token_id,
        eos_token_id=gen_tokenizer.eos_token_id,
    )

    gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    texts = gen_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return [t.strip() for t in texts]


# =========================
# ---- Prompt building ----
# =========================
SYSTEM_STYLE = (
    "You are a supportive, non-judgmental mental health companion. "
    "Be concise (120–180 words), warm, and practical. Avoid medical diagnoses or labels. "
    "If user expresses imminent risk, advise contacting local emergency services or 988 (U.S.). "
    "Use the KB Context (if present) for psychoeducation and coping tips. "
    "Ask one gentle follow-up question when appropriate."
)

def build_messages(history: List[Msg], kb_snippets: List[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_STYLE}]
    if kb_snippets:
        kb_text = "KB Context:\n- " + "\n- ".join(s.strip() for s in kb_snippets if s.strip())
        msgs.append({"role": "system", "content": kb_text})

    for m in history:
        r = "user" if m.role == "user" else ("assistant" if m.role == "assistant" else "system")
        msgs.append({"role": r, "content": m.content})
    return msgs


# =========================
# -------- FastAPI --------
# =========================
app = FastAPI()

# CORS for your Expo app/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can lock this down to your LAN IP / dev domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "gen_model": GEN_MODEL_ID,
        "embed_model": EMBED_MODEL_ID,
        "kb_items": len(KB_RAW),
        "intents": list(INTENTS.keys()),
    }

@app.post("/chat")
def chat(body: ChatIn):
    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_user_msgs = [m for m in body.messages if m.role == "user"]
    if not last_user_msgs:
        raise HTTPException(status_code=400, detail="Need at least one user message")

    last_text = last_user_msgs[-1].content.strip()
    if not last_text:
        raise HTTPException(status_code=400, detail="Empty user message")

    # Crisis short-circuit
    if crisis_check(last_text):
        return {
            "reply": (
                "I’m really sorry you’re feeling this way. "
                "If you might be in immediate danger, please call your local emergency number "
                "or 988 in the U.S. If you can, consider reaching out to someone you trust. "
                "I’m here to listen."
            ),
            "intent": "crisis",
            "alternates": [],
            "topic": "crisis",
            "kb_used": False,
            "kb_top_sim": 0.0,
        }

    # Intent tag (optional)
    intent, _ = pick_intent(last_text)

    # RAG: retrieve KB for the latest turn
    results, top_sim = retrieve_kb(last_text, TOP_K_PASSAGES)
    kb_snips: List[str] = []
    topic = "general"
    if results and top_sim >= SIMILARITY_FLOOR:
        topic, rows = pick_best_topic(results)
        kb_snips.extend(select_sections(rows, "description", 1))
        kb_snips.extend(select_sections(rows, "psychoeducation_point", 1))
        kb_snips.extend(
            select_sections(rows, "coping_strategies", 1)
            or select_sections(rows, "evidence_based_exercise", 1)
        )
        kb_snips.extend(select_sections(rows, "referral_options", 1))

    # Build messages with full history + KB context
    n = body.n or 1
    msgs = build_messages(body.messages, kb_snips)
    seed = body.seed

    variants = generate_chat(msgs, n=n, seed=seed)
    reply = variants[0]
    alternates = variants[1:] if len(variants) > 1 else []

    return {
        "reply": reply,
        "alternates": alternates,
        "intent": intent,
        "topic": topic,
        "kb_used": bool(kb_snips),
        "kb_top_sim": round(float(top_sim), 4),
    }


# -------- Optional admin endpoints --------
@app.post("/admin/reload_intents")
def reload_intents():
    global INTENTS, PROTOTYPES
    INTENTS = load_json(INTENTS_PATH)
    PROTOTYPES = build_intent_prototypes()
    return {"ok": True, "intents": list(INTENTS.keys())}

@app.post("/admin/reload_kb")
def reload_kb():
    global KB_RAW, KB_TEXTS, KB_EMB
    KB_RAW = [KBEntry(**row) for row in load_json(KB_PATH)]
    KB_TEXTS = [
        f"{row.name} | {row.description_type.replace('_', ' ')} | {row.description}"
        for row in KB_RAW
    ]
    KB_EMB = embed_texts(KB_TEXTS)
    return {"ok": True, "kb_items": len(KB_RAW)}


# -------- Optional: /transcribe (server-side STT) --------
# Import is inside the handler so your app still starts even if the package isn't installed.
# @app.post("/transcribe")
# async def transcribe(file: UploadFile = File(...)):
#     try:
#         from faster_whisper import WhisperModel  # lazy import
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"ASR backend not installed. Please `pip install faster-whisper soundfile` ({e})",
#         )

#     suffix = Path(file.filename).suffix or ".m4a"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     try:
#         asr_model_name = os.getenv("ASR_MODEL", "base.en")
#         device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#         model = WhisperModel(asr_model_name, device=device)
#         segments, _info = model.transcribe(tmp_path, beam_size=1)
#         text = "".join(seg.text for seg in segments).strip()
#         return {"text": text}
#     finally:
#         try:
#             os.unlink(tmp_path)
#         except Exception:
#             pass

# --- imports at top of file ---
import tempfile, os
from fastapi.responses import JSONResponse

# --- add a simple cache so we don't re-load the model every request ---
_WHISPER = None

def get_whisper_model():
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(f"ASR backend not installed. pip install faster-whisper soundfile  ({e})")

    model_name = os.getenv("ASR_MODEL", "base.en")
    # IMPORTANT: don't pass "mps" here. Use "cuda" if available else "auto" (falls back to cpu).
    device = "cuda" if torch.cuda.is_available() else "auto"
    # Reasonable defaults: int8 on CPU, float16 on CUDA
    compute_type = "float16" if device == "cuda" else "int8"
    _WHISPER = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _WHISPER

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # save temp file
    suffix = Path(file.filename).suffix or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        try:
            model = get_whisper_model()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "asr_not_ready", "detail": str(e)})

        # transcribe
        segments, info = model.transcribe(tmp_path, beam_size=1, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        return {"text": text}
    except Exception as e:
        # surface the true reason instead of a generic 500
        return JSONResponse(status_code=400, content={"error": "transcribe_failed", "detail": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass



# Local dev entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
