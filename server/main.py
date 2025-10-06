# main.py
import json
import random
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
GEN_MODEL_ID = Path(__file__).parent.joinpath(".model").read_text().strip() \
    if Path(__file__).parent.joinpath(".model").exists() else "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "KB.json"
INTENTS_PATH = BASE_DIR / "intents.json"

TOP_K_PASSAGES = 6
SIMILARITY_FLOOR = 0.20
INTENT_MIN_SCORE = 0.18
MAX_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.9

# ---------------- Data models ----------------
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

# ---------------- Load KB & intents ----------------
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())

KB_RAW: List[KBEntry] = [KBEntry(**row) for row in load_json(KB_PATH)]
INTENTS: Dict[str, List[str]] = load_json(INTENTS_PATH)

# ---------------- Embedding model ----------------
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

# Build KB embeddings
KB_TEXTS: List[str] = [
    f"{row.name} | {row.description_type.replace('_', ' ')} | {row.description}"
    for row in KB_RAW
]
KB_EMB: np.ndarray = embed_texts(KB_TEXTS)

# Intent prototypes
def build_intent_prototypes() -> Dict[str, np.ndarray]:
    protos: Dict[str, np.ndarray] = {}
    for intent, examples in INTENTS.items():
        if not examples: 
            continue
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

def group_by_topic(results):
    by = {}
    for r, s in results:
        by.setdefault(r.name, []).append((r, s))
    return by

def pick_best_topic(results):
    by = group_by_topic(results)
    best_name, best_score, best_rows = "General Mental Health", -1.0, []
    for name, rows in by.items():
        m = max(s for _, s in rows)
        if m > best_score:
            best_name, best_score, best_rows = name, m, rows
    return best_name, best_rows

def select_sections(rows, key: str, n: int = 2):
    pool = [r.description for r, _ in rows if r.description_type == key]
    return pool[:max(0, n)]

# ---------------- Intent ----------------
def pick_intent(text: str) -> Tuple[str, float]:
    if not PROTOTYPES:
        return "neutral", 0.0
    u = embed_texts([text])[0]
    best, best_s = "neutral", -1.0
    for k, proto in PROTOTYPES.items():
        s = float(np.dot(u, proto))
        if s > best_s:
            best, best_s = k, s
    if best_s < INTENT_MIN_SCORE:
        return "neutral", best_s
    return best, best_s

# ---------------- Generation model ----------------
torch_dtype = torch.float16 if DEVICE.type in ("mps", "cuda") else torch.float32
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, use_fast=True)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID, torch_dtype=torch_dtype)
gen_model.to(DEVICE)
gen_model.eval()

def apply_chat_template(messages: List[Dict[str, str]], add_generation_prompt=True) -> str:
    if getattr(gen_tokenizer, "chat_template", None):
        return gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    text = []
    for m in messages:
        text.append(f"<|{m['role']}|>\n{m['content'].strip()}\n")
    if add_generation_prompt:
        text.append("<|assistant|>\n")
    return "".join(text)

SYSTEM_STYLE = (
    "You are a supportive, non-judgmental mental health companion. "
    "Be concise (120–180 words), warm, and practical. Avoid medical diagnoses. "
    "If user expresses imminent risk, advise contacting local emergency services or 988 (U.S.). "
    "Use the KB Context when present. Ask one gentle follow-up question when appropriate."
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

def generate_chat(messages: List[Dict[str, str]], n: int = 1, seed: Optional[int] = None) -> List[str]:
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
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

def generate_chat_stream(messages: List[Dict[str, str]], seed: Optional[int] = None):
    """Yields text chunks as they are generated."""
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    prompt = apply_chat_template(messages, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        gen_tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_TOKENS,
        pad_token_id=gen_tokenizer.eos_token_id,
        eos_token_id=gen_tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=gen_model.generate, kwargs=gen_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text

# ---------------- FastAPI ----------------
app = FastAPI()

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

    last_user_msgs = [m for m in body.messages if m.role == "user"]
    if not last_user_msgs:
        raise HTTPException(status_code=400, detail="Need at least one user message")

    last_text = last_user_msgs[-1].content.strip()
    if not last_text:
        raise HTTPException(status_code=400, detail="Empty user message")

    if crisis_check(last_text):
        return {
            "reply": ("I’m really sorry you’re feeling this way. If you might be in immediate danger, "
                      "please call your local emergency number or 988 in the U.S. "
                      "If you can, reach out to someone you trust. I’m here to listen."),
            "intent": "crisis",
            "alternates": [],
            "topic": "crisis"
        }

    intent, _ = pick_intent(last_text)
    results, top_sim = retrieve_kb(last_text, TOP_K_PASSAGES)

    kb_snips: List[str] = []
    topic = "general"
    if results and top_sim >= SIMILARITY_FLOOR:
        topic, rows = pick_best_topic(results)
        kb_snips.extend(select_sections(rows, "description", 1))
        kb_snips.extend(select_sections(rows, "psychoeducation_point", 1))
        kb_snips.extend(select_sections(rows, "coping_strategies", 1) or
                        select_sections(rows, "evidence_based_exercise", 1))
        kb_snips.extend(select_sections(rows, "referral_options", 1))

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
        "kb_top_sim": round(float(top_sim), 4)
    }

# -------- WebSocket streaming --------
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        # Expect: {"messages":[{role,content},...], "seed": optional int}
        payload = await ws.receive_json()
        msgs_in = [Msg(**m) for m in payload.get("messages", [])]
        seed = payload.get("seed")

        # Validation
        if not msgs_in:
            await ws.send_json({"type": "error", "error": "No messages"})
            await ws.close()
            return

        last_user = next((m for m in reversed(msgs_in) if m.role == "user"), None)
        if not last_user or not last_user.content.strip():
            await ws.send_json({"type": "error", "error": "Need a user message"})
            await ws.close()
            return

        last_text = last_user.content.strip()

        # Crisis short-circuit
        if crisis_check(last_text):
            await ws.send_json({"type": "delta", "text": (
                "I’m really sorry you’re feeling this way. If you might be in immediate danger, "
                "please call your local emergency number or 988 in the U.S. "
                "If you can, reach out to someone you trust. I’m here to listen."
            )})
            await ws.send_json({"type": "done", "intent": "crisis", "topic": "crisis"})
            await ws.close()
            return

        # Intent + KB
        intent, _ = pick_intent(last_text)
        results, top_sim = retrieve_kb(last_text, TOP_K_PASSAGES)
        kb_snips: List[str] = []
        topic = "general"
        if results and top_sim >= SIMILARITY_FLOOR:
            topic, rows = pick_best_topic(results)
            kb_snips.extend(select_sections(rows, "description", 1))
            kb_snips.extend(select_sections(rows, "psychoeducation_point", 1))
            kb_snips.extend(select_sections(rows, "coping_strategies", 1) or
                            select_sections(rows, "evidence_based_exercise", 1))
            kb_snips.extend(select_sections(rows, "referral_options", 1))

        msgs = build_messages(msgs_in, kb_snips)

        # Stream tokens
        for chunk in generate_chat_stream(msgs, seed=seed):
            # throttle tiny chunks a bit for UI smoothness
            if not chunk:
                await asyncio.sleep(0.01)
                continue
            await ws.send_json({"type": "delta", "text": chunk})

        await ws.send_json({"type": "done", "intent": intent, "topic": topic})
        await ws.close()

    except WebSocketDisconnect:
        # Client dropped; nothing to do (generation thread will finish on its own)
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

# -------- Admin reload --------
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
