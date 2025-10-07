# main.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import random
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import uuid, re

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
BASE_DIR = Path(__file__).parent

GEN_MODEL_ID = (
    BASE_DIR.joinpath(".model").read_text().strip()
    if BASE_DIR.joinpath(".model").exists()
    else "Qwen/Qwen2.5-1.5B-Instruct"
)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

KB_PATH = BASE_DIR / "KB.json"
INTENTS_PATH = BASE_DIR / "intents.json"

TOP_K_PASSAGES = 6
SIMILARITY_FLOOR = 0.20
INTENT_MIN_SCORE = 0.18
MAX_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.9

# WebSocket origins (relaxed for dev)
WS_ALLOW_ALL = os.getenv("WS_ALLOW_ALL", "1") == "1"
WS_ALLOWED = {o.strip() for o in os.getenv("WS_ALLOWED", "").split(",") if o.strip()}
WS_ALLOWED |= {
    None, "null",
    "http://localhost:8081", "http://127.0.0.1:8081",
    "http://localhost:19006", "http://127.0.0.1:19006",
}

# Device pref
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ---------------- FastAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Schemas ----------------
class Msg(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    messages: List[Msg]
    n: Optional[int] = None
    seed: Optional[int] = None

class KBEntry(BaseModel):
    name: str
    description_type: str
    description: str

# ---------------- Safety ----------------
def crisis_check(text: str) -> bool:
    t = text.lower()
    keywords = [
        "suicide","kill myself","end it all","self-harm","overdose",
        "cut myself","no reason to live","want to die","hurt myself"
    ]
    return any(k in t for k in keywords)

# ---------------- Load KB & Intents ----------------
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())

KB_RAW: List[KBEntry] = [KBEntry(**row) for row in load_json(KB_PATH)]
INTENTS: Dict[str, List[str]] = load_json(INTENTS_PATH)

# ---------- Live ASR (micro-batch) utilities ----------

ASR_SESSIONS: dict[str, dict] = {}  # sid -> {"text": str}

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")

def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s or "")

def _norm(tok: str) -> str:
    # normalize for overlap matching
    return tok.lower()

def _detok(tokens: list[str]) -> str:
    # simple detokenizer to keep spacing readable
    out = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t)
            continue
        prev = tokens[i - 1]
        # no space before closing punct, or after opening bracket
        if t in ".,!?;:%)]}" or prev in "([{":
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)

def merge_transcripts(prev: str, new: str, max_overlap: int = 12) -> str:
    """Word-level suffix/prefix merge to avoid dupes and missing spaces."""
    prev = (prev or "").strip()
    new = (new or "").strip()
    if not prev:
        return new
    if not new:
        return prev

    pw = _tokenize(prev)
    nw = _tokenize(new)
    pn = [_norm(t) for t in pw]
    nn = [_norm(t) for t in nw]

    kmax = min(max_overlap, len(pw), len(nw))
    k = 0
    for j in range(kmax, 0, -1):
        if pn[-j:] == nn[:j]:
            k = j
            break

    remainder = nw[k:]
    if not remainder:
        return prev

    # Decide if we need a space between prev and remainder text
    need_space = (prev and prev[-1].isalnum() and remainder[0].isalnum())
    return prev + (" " if need_space else "") + _detok(remainder)

# ---------------- Embeddings ----------------
embedder = SentenceTransformer(EMBED_MODEL_ID)
try:
    embedder.to(DEVICE)
except Exception:
    pass

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    v = embedder.encode(
        texts, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    return v.astype(np.float32)

KB_TEXTS: List[str] = [
    f"{row.name} | {row.description_type.replace('_', ' ')} | {row.description}"
    for row in KB_RAW
]
KB_EMB: np.ndarray = embed_texts(KB_TEXTS)

def build_intent_prototypes() -> Dict[str, np.ndarray]:
    protos: Dict[str, np.ndarray] = {}
    for intent, examples in INTENTS.items():
        if not examples: continue
        embs = embed_texts(examples)
        proto = embs.mean(axis=0)
        proto /= (np.linalg.norm(proto) + 1e-8)
        protos[intent] = proto
    return protos

PROTOTYPES: Dict[str, np.ndarray] = build_intent_prototypes()

# ---------------- Retrieval ----------------
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def retrieve_kb(user_text: str, top_k: int = TOP_K_PASSAGES):
    q = embed_texts([user_text])
    sims = cosine(q, KB_EMB)[0]
    idx = np.argsort(-sims)[:top_k]
    results = [(KB_RAW[i], float(sims[i])) for i in idx]
    top_sim = float(sims[idx[0]]) if len(idx) else 0.0
    return results, top_sim

def group_by_topic(results: List[Tuple[KBEntry, float]]):
    by = {}
    for r, s in results:
        by.setdefault(r.name, []).append((r, s))
    return by

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

# ---------------- Intent ----------------
def pick_intent(text: str) -> Tuple[str, float]:
    if not PROTOTYPES: return "neutral", 0.0
    u = embed_texts([text])[0]
    best, best_s = "neutral", -1.0
    for k, proto in PROTOTYPES.items():
        s = float(np.dot(u, proto))
        if s > best_s:
            best, best_s = k, s
    if best_s < INTENT_MIN_SCORE:
        return "neutral", best_s
    return best, best_s

# ---------------- Generation ----------------
torch_dtype = torch.float16 if DEVICE.type in ("mps", "cuda") else torch.float32
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, use_fast=True)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID, torch_dtype=torch_dtype)
gen_model.to(DEVICE)
gen_model.eval()

SYSTEM_STYLE = (
    "You are a supportive, non-judgmental mental health companion. "
    "Be concise (120–180 words), warm, and practical. Avoid medical diagnoses. "
    "If the user expresses imminent risk, advise contacting emergency services/988 (U.S.). "
    "Use KB Context (if present) for psychoeducation and coping tips. "
    "Ask one gentle follow-up question when appropriate."
)

# -- Chat template helpers -----------------------------------------------------
def _template_qwen(messages: List[Dict[str, str]], add_generation_prompt=True) -> str:
    parts = []
    for m in messages:
        role = m["role"] if m["role"] in ("system","user","assistant") else "user"
        parts.append(f"<|im_start|>{role}\n{(m['content'] or '').strip()}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)

def _template_llama3(messages: List[Dict[str, str]], add_generation_prompt=True) -> str:
    text = "<|begin_of_text|>"
    for m in messages:
        role = m["role"]
        content = (m["content"] or "").strip()
        header = "system" if role == "system" else ("assistant" if role == "assistant" else "user")
        text += f"<|start_header_id|>{header}<|end_header_id|>\n{content}<|eot_id|>"
    if add_generation_prompt:
        text += "<|start_header_id|>assistant<|end_header_id|>\n"
    return text

def _template_gemma2(messages: List[Dict[str, str]], add_generation_prompt=True) -> str:
    parts = []
    for m in messages:
        who = "model" if m["role"] == "assistant" else ("user" if m["role"] != "system" else "system")
        parts.append(f"<start_of_turn>{who}\n{(m['content'] or '').strip()}<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)

def _template_mistral(messages: List[Dict[str, str]], add_generation_prompt=True) -> str:
    sys_blocks = [m["content"].strip() for m in messages if m["role"] == "system"]
    sys = "\n".join(sys_blocks) if sys_blocks else ""
    chunks = []
    if sys:
        chunks.append(f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n")
    for m in messages:
        if m["role"] == "user":
            chunks.append(m["content"].strip() + " [/INST] ")
        elif m["role"] == "assistant":
            chunks.append(m["content"].strip() + " ")
    if add_generation_prompt:
        chunks.append("")
    return "".join(chunks)

def apply_chat_template(messages: List[Dict[str, str]], add_generation_prompt=True) -> str:
    tmpl = getattr(gen_tokenizer, "chat_template", None)
    if tmpl:
        return gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    m = GEN_MODEL_ID.lower()
    if "qwen" in m or "tinyllama" in m or "chatml" in m:
        return _template_qwen(messages, add_generation_prompt)
    if "llama" in m:
        return _template_llama3(messages, add_generation_prompt)
    if "gemma" in m:
        return _template_gemma2(messages, add_generation_prompt)
    if "mistral" in m:
        return _template_mistral(messages, add_generation_prompt)

    text = []
    for msg in messages:
        text.append(f"{msg['role']}:\n{msg['content'].strip()}\n\n")
    if add_generation_prompt:
        text.append("assistant:\n")
    return "".join(text)

# -- Echo trimming -------------------------------------------------------------
ECHO_MARKERS = [
    "assistant:\n", "assistant\n", "Assistant:\n", "ASSISTANT:\n",
    "<|assistant|>", "<|im_start|>assistant\n", "<start_of_turn>model\n",
]

def trim_prompt_echo(full_text: str) -> str:
    s = full_text.lstrip("\n")  # keep spaces, only drop leading newlines
    low = s.lower()

    cut_idx = None
    for marker in ECHO_MARKERS:
        p = low.rfind(marker.lower())
        if p != -1:
            cut_idx = p + len(marker)
    if cut_idx is not None:
        return s[cut_idx:]  # don't strip spaces

    if low.startswith("system") or low.startswith("user"):
        parts = s.split("\n")
        lows = [p.lower() for p in parts]
        if "assistant" in lows:
            i = lows.index("assistant")
            return "\n".join(parts[i+1:])

    return s

# -- Build messages ------------------------------------------------------------
def build_messages(history: List[Msg], kb_snippets: List[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_STYLE}]
    if kb_snippets:
        kb_text = "KB Context:\n- " + "\n- ".join(s.strip() for s in kb_snippets if s.strip())
        msgs.append({"role": "system", "content": kb_text})
    for m in history:
        r = "user" if m.role == "user" else ("assistant" if m.role == "assistant" else "system")
        msgs.append({"role": r, "content": m.content})
    return msgs

def generate_text_stream(messages: List[Dict[str, str]], seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    prompt = apply_chat_template(messages, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        gen_tokenizer,
        skip_special_tokens=True,
        decode_kwargs={
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,  # keep spaces exactly
        },
    )

    def _run():
        gen_model.generate(
            **inputs,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            max_new_tokens=MAX_TOKENS,
            streamer=streamer,
            pad_token_id=gen_tokenizer.eos_token_id,
            eos_token_id=gen_tokenizer.eos_token_id,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Echo filter: buffer until we detect the assistant marker once
    started = False
    buffer = ""
    for chunk in streamer:
        if not started:
            buffer += chunk
            trimmed_now = trim_prompt_echo(buffer)
            if trimmed_now and trimmed_now != buffer:
                started = True
                yield trimmed_now
            elif any(m in buffer.lower() for m in [mk.lower() for mk in ECHO_MARKERS]):
                continue
        else:
            yield chunk

# ---------------- REST ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "gen_model": GEN_MODEL_ID,
        "embed_model": EMBED_MODEL_ID,
        "kb_items": len(KB_RAW),
        "intents": list(INTENTS.keys())
    }

@app.post("/chat")
def chat(body: ChatIn):
    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="Need at least one user message")

    last_text = user_msgs[-1].content.strip()
    if crisis_check(last_text):
        return {
            "reply": (
                "I’m really sorry you’re feeling this way. "
                "If you might be in immediate danger, please call your local emergency number "
                "or 988 in the U.S. If you can, reach out to someone you trust. I’m here to listen."
            ),
            "intent": "crisis",
            "alternates": [],
            "topic": "crisis"
        }

    intent, _ = pick_intent(last_text)

    results, top_sim = retrieve_kb(last_text, TOP_K_PASSAGES)
    kb_snips: List[str] = []
    topic = "general"
    if results:
        topic, rows = pick_best_topic(results)
        if top_sim >= SIMILARITY_FLOOR:
            kb_snips.extend(select_sections(rows, "description", 1))
            kb_snips.extend(select_sections(rows, "psychoeducation_point", 1))
            kb_snips.extend(select_sections(rows, "coping_strategies", 1) or select_sections(rows, "evidence_based_exercise", 1))
            kb_snips.extend(select_sections(rows, "referral_options", 1))

    msgs = build_messages(body.messages, kb_snips)
    chunks = list(generate_text_stream(msgs, seed=body.seed))
    text = trim_prompt_echo("".join(chunks))

    return {
        "reply": text,
        "alternates": [],
        "intent": intent,
        "topic": topic,
        "kb_used": bool(kb_snips),
        "kb_top_sim": round(float(top_sim), 4),
    }

# ---------------- WebSocket (Streaming) ----------------
from fastapi import Query

@app.post("/asr/session")
def asr_open_session():
    sid = uuid.uuid4().hex[:12]
    ASR_SESSIONS[sid] = {"text": ""}
    return {"sid": sid}

@app.post("/asr/append")
async def asr_append(
    sid: str = Query(...),
    seq: int = Query(0, ge=0),
    final: bool = Query(False),
    file: UploadFile = File(...)
):
    if not WHISPER_READY or WHISPER is None:
        raise HTTPException(status_code=500, detail="Whisper model not initialized. Install faster-whisper and ffmpeg.")
    sess = ASR_SESSIONS.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown ASR session")

    try:
        import tempfile, os as _os
        suffix = Path(file.filename or "clip").suffix or ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name

        segments, info = WHISPER.transcribe(tmp_path, beam_size=1)
        clip_text = "".join(seg.text for seg in segments).strip()
        _os.unlink(tmp_path)

        merged = merge_transcripts(sess["text"], clip_text)
        sess["text"] = merged

        if final:
            # finalize + cleanup
            text = sess["text"].strip()
            ASR_SESSIONS.pop(sid, None)
            return {"sid": sid, "text": text, "final": True}

        return {"sid": sid, "text": sess["text"], "final": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if not (WS_ALLOW_ALL or origin in WS_ALLOWED):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        init = await websocket.receive_json()
        raw_messages = init.get("messages", [])
        seed = init.get("seed")

        hist = [Msg(**m) for m in raw_messages]
        user_msgs = [m for m in hist if m.role == "user"]
        last_text = user_msgs[-1].content.strip() if user_msgs else ""

        if not last_text:
            await websocket.send_json({"type": "error", "message": "No user text"})
            await websocket.close(code=1003)
            return

        if crisis_check(last_text):
            await websocket.send_json({
                "type": "done",
                "text": (
                    "I’m really sorry you’re feeling this way. "
                    "If you might be in immediate danger, please call your local emergency number "
                    "or 988 in the U.S. If you can, reach out to someone you trust. I’m here to listen."
                ),
                "intent": "crisis",
                "topic": "crisis"
            })
            await websocket.close()
            return

        intent, _ = pick_intent(last_text)

        results, top_sim = retrieve_kb(last_text, TOP_K_PASSAGES)
        kb_snips: List[str] = []
        topic = "general"
        if results:
            topic, rows = pick_best_topic(results)
            if top_sim >= SIMILARITY_FLOOR:
                kb_snips.extend(select_sections(rows, "description", 1))
                kb_snips.extend(select_sections(rows, "psychoeducation_point", 1))
                kb_snips.extend(select_sections(rows, "coping_strategies", 1) or select_sections(rows, "evidence_based_exercise", 1))
                kb_snips.extend(select_sections(rows, "referral_options", 1))

        msgs = build_messages(hist, kb_snips)

        started = False
        buffer = ""
        for chunk in generate_text_stream(msgs, seed=seed):
            if not started:
                buffer += chunk
                trimmed_now = trim_prompt_echo(buffer)
                if trimmed_now and trimmed_now != buffer:
                    started = True
                    await websocket.send_json({"type": "delta", "text": trimmed_now})
                elif any(m in buffer.lower() for m in [mk.lower() for mk in ECHO_MARKERS]):
                    continue
            else:
                await websocket.send_json({"type": "delta", "text": chunk})

        await websocket.send_json({
            "type": "done",
            "intent": intent,
            "topic": topic,
            "kb_used": bool(kb_snips),
            "kb_top_sim": round(float(top_sim), 4),
        })
        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            await websocket.close(code=1011)

# ---------------- Transcription ----------------
WHISPER_READY = False
WHISPER = None
try:
    from faster_whisper import WhisperModel
    model_size = os.getenv("WHISPER_SIZE", "base")  # tiny|base|small|medium|large-v3
    w_device = "cuda" if torch.cuda.is_available() else ("cpu" if not torch.backends.mps.is_available() else "cpu")
    WHISPER = WhisperModel(model_size, device=w_device, compute_type="int8")
    WHISPER_READY = True
except Exception:
    WHISPER_READY = False

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not WHISPER_READY or WHISPER is None:
        raise HTTPException(status_code=500, detail="Whisper model not initialized. Install faster-whisper and ffmpeg.")

    # Toggle via env: ENGLISH_ONLY=1 (default) forces English
    ENGLISH_ONLY = os.getenv("ENGLISH_ONLY", "1") == "1"
    try:
        import tempfile, os as _os
        suffix = Path(file.filename or "audio").suffix or ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name

        # Force English transcription. If the input isn't English, Whisper will not
        # auto-switch languages; it’ll try to decode as English (often resulting in
        # blanks/garble instead of picking another language).
        segments, info = WHISPER.transcribe(
            tmp_path,
            language=("en" if ENGLISH_ONLY else None),
            task="transcribe",
            beam_size=1,
            vad_filter=True,  # trims long silences for nicer UX
        )

        text = "".join(seg.text for seg in segments).strip()

        # Optional hard rejection if Whisper still thinks language != en with high confidence.
        # (Note: when language="en" is set, info.language is typically "en", but we keep this guard.)
        lang = getattr(info, "language", "en")
        lang_p = float(getattr(info, "language_probability", 0.0) or 0.0)

        if ENGLISH_ONLY and lang != "en" and lang_p >= 0.60:
            # Strict mode: refuse non-English confidently detected audio
            _os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="English only: detected non-English speech.")

        _os.unlink(tmp_path)
        return {"text": text, "language": lang, "language_probability": lang_p}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- Admin --------
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
