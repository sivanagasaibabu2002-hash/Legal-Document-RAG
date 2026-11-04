# legal_rag.py — Retrieval-augmented Q&A over legal documents
# Primary retrieval: FAISS vector index (persistent local files)
# Fallback retrieval: in-memory TF-IDF
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

import numpy as np
from ollama import chat as ollama_chat

try:
    from nltk.tokenize import sent_tokenize as _sent_tokenize
except ImportError:
    _sent_tokenize = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

OLLAMA_MODEL = "llama3.2:3b"
print(f"[legal_rag][DEBUG] RAG generation LLM (Ollama): model={OLLAMA_MODEL!r}", flush=True)
CHUNK_TARGET = 900
MAX_DOC_CHARS = 100_000
# Balanced tuning: keep latency controlled while preserving multi-clause coverage.
TOP_K_RETRIEVAL = 4
# Tighter budget for small LLMs: less noise, fewer unrelated clauses in-window.
MAX_CONTEXT_CHARS = 2200
FAISS_DIR = os.path.join(os.path.dirname(__file__), ".faiss_legal_rag")
# Bump when chunking/retrieval selection changes so stale FAISS files are not reused.
RAG_INDEX_VERSION = "v5-clause-split-inline-numbering"
_embedding_model = None
_index_lock = threading.Lock()
_indexed_doc_ids = set()
_faiss_import_failed = faiss is None
_tfidf_cache = {}
_tfidf_cache_lock = threading.Lock()


def _normalize_whitespace(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_broken_text(text: str) -> str:
    import re

    if not text:
        return ""

    # Add space between lowercase and uppercase letters
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Add space between letters and digits
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)

    # Recover missing spaces in long OCR-fused English tokens
    # so retrieval chunks are indexed with readable text.
    fused_pat = re.compile(r"\b[A-Za-z]{14,}\b")
    text = fused_pat.sub(lambda m: _split_fused_token(m.group(0)), text)

    # Normalize horizontal spaces (preserve line boundaries for better chunking)
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _window_chunks(text: str, size: int, step: int) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks if chunks else [text[:size]]


def _split_clause_blocks(text: str) -> List[str]:
    """
    Split legal documents into clause-like blocks when numbered clauses exist
    (e.g., '1.', '2.', ...). This improves retrieval for clause-specific questions.
    """
    t = text or ""
    if not t:
        return []
    # Normalize obvious OCR spacing around clause markers first:
    # "1 . The" -> "1. The"
    t = re.sub(r"(?<!\d)(\d{1,2})\s*\.\s*", r"\1. ", t)

    # Split before numbered clauses even when they appear mid-line, e.g.
    # "... otherwise. 2. The husband ..."
    # "... rights or otherwise 2. The husband ..."
    boundary = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=[.;:]))\s*(?=\d{1,2}\.\s)")
    parts = boundary.split(t)
    parts = [p.strip() for p in parts if p and p.strip()]

    # Keep only numbered clause chunks (drop preamble recitals for strict legal QA retrieval).
    clause_pat = re.compile(r"^\d{1,2}\.\s")
    blocks = [p for p in parts if clause_pat.match(p)]

    # If no numbered clauses are detected, fall back to previous behavior.
    if not blocks:
        return [p for p in parts if len(p) >= 40]

    # Ensure one logical clause per chunk by cutting off at next inline marker if any survived.
    cleaned: List[str] = []
    seen = set()
    for b in blocks:
        m = re.search(r"\s(?=\d{1,2}\.\s)", b[3:])
        chunk = b[: m.start() + 3].strip() if m else b.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            cleaned.append(chunk)
    return cleaned


def chunk_document(text: str, max_chars: int = CHUNK_TARGET) -> List[str]:
    """Split document into overlapping segments for retrieval."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Clause-aware path for legal docs:
    # one clause = one chunk (no cross-clause merging).
    clause_blocks = _split_clause_blocks(text)
    if len(clause_blocks) >= 4:
        out: List[str] = []
        seen = set()
        for block in clause_blocks:
            cc = block.strip()
            if not cc:
                continue
            # Keep clause boundaries strict. If a single clause is very large,
            # split only within that clause (no mixing with other clauses).
            if len(cc) > max_chars:
                for part in _window_chunks(cc, max_chars, max_chars):
                    pp = part.strip()
                    if pp and pp not in seen:
                        seen.add(pp)
                        out.append(pp)
                continue
            if cc not in seen:
                seen.add(cc)
                out.append(cc)
        if out:
            return out

    try:
        if _sent_tokenize is not None:
            sents = [s.strip() for s in _sent_tokenize(text) if s and s.strip()]
        else:
            raise RuntimeError("nltk optional")
    except Exception:
        sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]

    if not sents:
        return _window_chunks(text, max_chars, max_chars - 200)

    avg = len(text) / max(len(sents), 1)
    if len(sents) == 1 or avg > max_chars * 0.85:
        return _window_chunks(text, max_chars, max(200, max_chars // 3))

    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    overlap_tail = 2

    for s in sents:
        add_len = len(s) + (1 if buf else 0)
        if cur + add_len > max_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf = buf[-overlap_tail:] if len(buf) > overlap_tail else buf[-1:]
            cur = sum(len(x) + 1 for x in buf)
        buf.append(s)
        cur += add_len
    if buf:
        chunks.append(" ".join(buf).strip())

    out: List[str] = []
    seen = set()
    for c in chunks:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out if out else [text[:max_chars]]


def retrieve_relevant_chunks(
    question: str,
    chunks: Sequence[str],
    top_k_scan: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> List[str]:
    """Multilingual-friendly retrieval: char n-gram TF-IDF + cosine similarity."""
    chunks = [c for c in chunks if c and str(c).strip()]
    if not chunks:
        return []
    q = (question or "").strip()
    if not q:
        return list(chunks[: min(3, len(chunks))])

    if len(chunks) == 1:
        return [chunks[0]]

    cache_key = hashlib.sha256(
        (RAG_INDEX_VERSION + "||" + "\n\n".join(chunks)).encode("utf-8", errors="ignore")
    ).hexdigest()
    with _tfidf_cache_lock:
        cached = _tfidf_cache.get(cache_key)
    if cached is None:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        try:
            doc_mat = vectorizer.fit_transform(chunks)
        except ValueError:
            return list(chunks[: min(top_k_scan, len(chunks))])
        with _tfidf_cache_lock:
            # Tiny LRU behavior to cap memory on long sessions.
            if len(_tfidf_cache) >= 6:
                _tfidf_cache.pop(next(iter(_tfidf_cache)))
            _tfidf_cache[cache_key] = (vectorizer, doc_mat)
    else:
        vectorizer, doc_mat = cached
    try:
        q_mat = vectorizer.transform([q])
        sims = cosine_similarity(q_mat, doc_mat).flatten()
    except ValueError:
        return list(chunks[: min(top_k_scan, len(chunks))])

    order = np.argsort(-sims)
    base_idx = [int(i) for i in order[: min(len(order), max(top_k_scan * 3, top_k_scan))]]
    candidates = [chunks[i] for i in base_idx]
    cand_scores = [float(sims[i]) for i in base_idx]
    selected = _select_diverse_relevant_chunks(
        candidates,
        scores=cand_scores,
        max_chunks=top_k_scan,
        max_context_chars=max_context_chars,
    )
    return selected if selected else [chunks[int(order[0])]]


def _select_compact_context_chunks(
    chunks: Sequence[str],
    max_chunks: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> List[str]:
    """
    Keep only compact, high-value chunks:
    - remove empty/duplicate fragments
    - enforce max chunk count and total context budget
    """
    out: List[str] = []
    seen = set()
    total = 0
    for c in chunks:
        if not c:
            continue
        s = str(c).strip()
        if not s or s in seen:
            continue
        if total + len(s) > max_context_chars and out:
            break
        out.append(s)
        seen.add(s)
        total += len(s)
        if len(out) >= max_chunks or total >= max_context_chars:
            break
    return out


def _select_diverse_relevant_chunks(
    chunks: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    max_chunks: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    lambda_rel: float = 0.72,
    near_duplicate_threshold: float = 0.93,
) -> List[str]:
    """
    MMR-like selector:
    - keeps relevance to query (scores)
    - discourages redundancy via chunk-to-chunk cosine similarity
    """
    base = _select_compact_context_chunks(chunks, max_chunks=max_chunks * 3, max_context_chars=max_context_chars * 2)
    if not base:
        return []
    if len(base) == 1:
        return base

    rel = np.array(list(scores[: len(base)]) if scores else [1.0] * len(base), dtype=np.float32)
    if rel.size != len(base):
        rel = np.ones(len(base), dtype=np.float32)
    if np.max(rel) > np.min(rel):
        rel = (rel - np.min(rel)) / (np.max(rel) - np.min(rel))
    else:
        rel = np.ones_like(rel)

    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=1.0, sublinear_tf=True)
        mat = vec.fit_transform(base)
        pair = cosine_similarity(mat, mat)
    except ValueError:
        pair = np.eye(len(base), dtype=np.float32)

    chosen: List[int] = []
    budget = 0
    remaining = set(range(len(base)))

    while remaining and len(chosen) < max_chunks and budget < max_context_chars:
        if not chosen:
            nxt = max(remaining, key=lambda i: float(rel[i]))
        else:
            def mmr(i: int) -> float:
                max_sim = max(float(pair[i, j]) for j in chosen) if chosen else 0.0
                if max_sim >= near_duplicate_threshold:
                    return -1e9
                return lambda_rel * float(rel[i]) - (1.0 - lambda_rel) * max_sim
            nxt = max(remaining, key=mmr)
            if mmr(nxt) < -1e8:
                break
        txt = base[nxt]
        if budget + len(txt) > max_context_chars and chosen:
            break
        chosen.append(nxt)
        remaining.remove(nxt)
        budget += len(txt)

    return [base[i] for i in chosen] if chosen else _select_compact_context_chunks(base, max_chunks=max_chunks, max_context_chars=max_context_chars)


def _faiss_index_path(doc_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{doc_id}.index")


def _faiss_meta_path(doc_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{doc_id}.json")


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SentenceTransformer is not None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _faiss_available() -> bool:
    global faiss, _faiss_import_failed
    if _faiss_import_failed:
        return False
    if faiss is None:
        try:
            import faiss as _faiss
            faiss = _faiss
        except Exception:
            _faiss_import_failed = True
            return False
    return True


def _doc_id_from_text(doc_text: str) -> str:
    payload = (RAG_INDEX_VERSION + "\n" + (doc_text or "")).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _ensure_indexed_in_vector_db(doc_text: str, chunks: Sequence[str]) -> Optional[str]:
    """
    Index chunks once per document hash into FAISS.
    Returns doc_id when vector index is ready, else None.
    """
    if not _faiss_available():
        return None
    embedder = _get_embedding_model()
    if embedder is None:
        return None

    doc_id = _doc_id_from_text(doc_text)
    if doc_id in _indexed_doc_ids:
        return doc_id

    with _index_lock:
        if doc_id in _indexed_doc_ids:
            return doc_id

        os.makedirs(FAISS_DIR, exist_ok=True)
        idx_path = _faiss_index_path(doc_id)
        meta_path = _faiss_meta_path(doc_id)

        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            chunk_list = [c for c in chunks if c and str(c).strip()]
            if not chunk_list:
                return None

            embeddings = embedder.encode(
                chunk_list,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            emb = np.asarray(embeddings, dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] == 0:
                return None

            index = faiss.IndexFlatIP(int(emb.shape[1]))
            index.add(emb)
            faiss.write_index(index, idx_path)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunk_list}, f, ensure_ascii=False)

        _indexed_doc_ids.add(doc_id)
        return doc_id


def _retrieve_with_vector_db(doc_text: str, question: str, chunks: Sequence[str]) -> Optional[List[str]]:
    """Try FAISS vector retrieval. Return None if unavailable."""
    if not question.strip():
        return list(chunks[: min(3, len(chunks))])

    doc_id = _ensure_indexed_in_vector_db(doc_text, chunks)
    embedder = _get_embedding_model()
    if not doc_id or embedder is None or not _faiss_available():
        return None

    idx_path = _faiss_index_path(doc_id)
    meta_path = _faiss_meta_path(doc_id)
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None

    try:
        index = faiss.read_index(idx_path)
    except Exception:
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        chunk_list = meta.get("chunks", [])
    except Exception:
        return None
    if not chunk_list:
        return None

    q_emb = embedder.encode(
        [question],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    q_arr = np.asarray(q_emb, dtype=np.float32)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(1, -1)

    k = min(max(TOP_K_RETRIEVAL, 1), len(chunk_list))
    try:
        _dist, ids = index.search(q_arr, k)
    except Exception:
        return None

    ids_row = ids[0] if len(ids) > 0 else []
    dist_row = _dist[0] if len(_dist) > 0 else []
    candidates = []
    cand_scores = []
    base_idx = [int(i) for i in ids_row if 0 <= int(i) < len(chunk_list)]
    score_map = {}
    for pos, i in enumerate(ids_row):
        ii = int(i)
        if 0 <= ii < len(chunk_list):
            score_map[ii] = float(dist_row[pos]) if pos < len(dist_row) else 0.0
    for ii in base_idx:
        c = chunk_list[ii]
        if not c:
            continue
        candidates.append(c)
        cand_scores.append(score_map.get(ii, 0.0))
    selected = _select_diverse_relevant_chunks(
        candidates,
        scores=cand_scores,
        max_chunks=TOP_K_RETRIEVAL,
        max_context_chars=MAX_CONTEXT_CHARS,
    )
    return selected if selected else None


def _build_messages(context: str, question: str, history: Optional[Sequence[dict]]) -> list:
    system = (
        "You are a legal-document question-answering assistant. Use ONLY the provided DOCUMENT "
        "EXCERPTS for the current question.\n\n"
        "Rules you must follow:\n\n"
        "1. Full-context analysis\n\n"
        "* Carefully review all relevant excerpt segments before answering.\n"
        "* Do not rely on a single sentence or clause when additional relevant information "
        "exists.\n\n"
        "2. Complete and multi-part answering\n\n"
        "* Address every part of the question explicitly.\n"
        "* If the question contains multiple components, ensure each component is answered fully.\n"
        "* Include all relevant conditions, exceptions, and limitations mentioned in the excerpts.\n\n"
        "3. Question-specific response\n\n"
        "* Treat each question independently.\n"
        "* Do not reuse or copy answers from previous questions.\n"
        "* Ignore prior assistant outputs and rely only on the current question and provided "
        "excerpts.\n\n"
        "4. No over-summarization\n\n"
        "* Prioritize correctness and completeness over brevity.\n"
        "* Do not omit important legal conditions or qualifying details.\n\n"
        "5. Strict grounding\n\n"
        "* Use only the information present in the provided excerpts.\n"
        "* Do not infer, assume, or introduce any facts not explicitly stated.\n"
        "* Do not interpret, summarize implications, or draw conclusions beyond what is explicitly "
        "stated.\n"
        "* Do not explain what any clause indicates, suggests, or means.\n"
        "* Do not use phrases such as 'this indicates', 'this suggests', or 'this means'.\n"
        "* Only extract and restate the exact facts from the document excerpts.\n"
        "* Never invent numbers, dates, durations, names, or clause details.\n"
        "* If any information is missing or unclear, explicitly state:\n"
        "  \"Not specified in the provided document excerpts.\"\n\n"
        "6. Clause relevance and separation\n\n"
        "* Use only those parts of the excerpts that are directly relevant to the question.\n"
        "* Do not combine or mix information from unrelated clauses.\n"
        "* Ensure that each statement in the answer is supported by a clearly relevant part of "
        "the excerpts.\n"
        "* Do not assign roles, responsibilities, or rights unless explicitly stated in the "
        "excerpts.\n\n"
        "Output requirements:\n\n"
        "* Structure the answer clearly and logically.\n"
        "* Use separate sentences or bullet points when appropriate.\n"
        "* If the question has multiple parts, answer each part in a clearly separated line or "
        "point.\n"
        "* Do not merge unrelated points into a single sentence.\n"
        "* Ensure each statement directly corresponds to a specific part of the question.\n"
        "* Keep statements factual, precise, and grounded in the excerpts.\n"
        "* If excerpt text appears to have missing spaces due to extraction/OCR artifacts, restore "
        "readable word spacing for clarity without changing meaning or adding new facts.\n"
        "* Use the same language as the question when appropriate."
    )
    messages = [{"role": "system", "content": system}]
    # Keep only recent USER turns to reduce answer drift from earlier assistant outputs.
    if history:
        recent_users = []
        for turn in history[-8:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role == "user" and content:
                recent_users.append(content[:2000])
        for q_prev in recent_users[-3:]:
            messages.append({"role": "user", "content": f"Previous user question for context: {q_prev}"})
    user_block = (
        "DOCUMENT EXCERPTS (retrieved segments; order is not necessarily document order):\n\n"
        f"{context}\n\n---\nQUESTION:\n{question}"
    )
    messages.append({"role": "user", "content": user_block[:15000]})
    return messages


def _extract_potential_fact_tokens(text: str) -> set:
    """
    Extract high-risk hallucination tokens:
    - amounts/currency/percent
    - dates
    - durations/years/months
    - standalone long numbers
    """
    s = text or ""
    tokens = set()

    patterns = [
        r"(?:Rs\.?|INR|₹|USD|\$)\s*[\d,]+(?:\.\d+)?",
        r"[\d,]+(?:\.\d+)?\s*(?:%|percent)",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:\d{4})\b",
        r"\b\d+\s*(?:years?|months?|days?)\b",
        r"\b\d{3,}\b",
    ]
    for p in patterns:
        for m in re.findall(p, s, flags=re.IGNORECASE):
            tok = re.sub(r"\s+", " ", str(m).strip())
            if tok:
                tokens.add(tok.lower())
    return tokens


def _find_unsupported_tokens(answer: str, context: str) -> List[str]:
    ans_tokens = _extract_potential_fact_tokens(answer)
    ctx_tokens = _extract_potential_fact_tokens(context)
    unsupported = [t for t in ans_tokens if t not in ctx_tokens]
    return unsupported[:8]


def _generate_answer_with_guardrails(messages: list) -> str:
    resp = ollama_chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.1, "num_predict": 320},
    )
    out = (getattr(resp.message, "content", None) or "").strip()
    return out if out else "(The model returned an empty answer.)"


def _repair_answer_if_needed(question: str, answer: str, context: str) -> str:
    # Fast path: skip repair unless answer includes risky factual tokens
    # (numbers/dates/currency/durations) that can hallucinate.
    if not _extract_potential_fact_tokens(answer):
        return answer
    unsupported = _find_unsupported_tokens(answer, context)
    if not unsupported:
        return answer

    repair_messages = [
        {
            "role": "system",
            "content": (
                "Revise the answer so every factual token is grounded in DOCUMENT EXCERPTS. "
                "Do not add new facts. If information is absent, explicitly say "
                "'Not specified in the provided document excerpts.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT EXCERPTS:\n{context[:12000]}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"DRAFT ANSWER:\n{answer}\n\n"
                f"UNSUPPORTED TOKENS FOUND:\n{', '.join(unsupported)}\n\n"
                "Return corrected final answer only."
            ),
        },
    ]
    try:
        repaired = _generate_answer_with_guardrails(repair_messages)
        # If still unsupported, prefer conservative original with explicit caution.
        if _find_unsupported_tokens(repaired, context):
            return (
                f"{answer}\n\nNote: Some numeric/date details may be absent in the provided excerpts. "
                "Treat unspecified values as not provided in the document excerpts."
            )
        return repaired
    except Exception:
        return answer


_FUSED_WORD_LEXICON = {
    "a", "access", "account", "actions", "after", "against", "agreement", "all", "allowance", "and", "any",
    "apart", "are", "arisen", "as", "authority", "be", "because", "before", "between", "by", "called", "can",
    "case", "chaste", "child", "children", "claim", "claims", "clause", "conjugal", "consent", "contained",
    "contract", "could", "custody", "date", "days", "death", "demand", "demands", "differences", "discharge",
    "disputes", "document", "duplicate", "during", "each", "either", "entitled", "except", "exception",
    "executed", "expressly", "facts", "for", "from", "further", "gave", "giving", "guardian", "guardianship",
    "has", "have", "having", "he", "her", "here", "hereafter", "herein", "hers", "him", "his", "husband",
    "if", "in", "incurred", "indemnified", "indemnifies", "intend", "is", "it", "its", "keep", "keeps",
    "legal", "liable", "liabilities", "liability", "life", "lifetime", "limitations", "live", "living",
    "made", "maintenance", "majority", "marriage", "may", "months", "namely", "no", "not", "notice",
    "notwithstanding", "now", "obligation", "of", "on", "or", "other", "otherwise", "out", "over", "part",
    "parties", "party", "pay", "payable", "payment", "present", "presents", "prior", "proceeding", "provided",
    "question", "reconciliation", "relevant", "resident", "respectively", "responsibilities", "responsibility",
    "restitution", "return", "revoked", "right", "rights", "same", "separate", "separately", "settlement",
    "shall", "she", "society", "specified", "state", "stated", "states", "stop", "such", "support", "that",
    "the", "their", "them", "there", "thereafter", "therein", "thereis", "thereof", "this", "times", "to",
    "together", "under", "unless", "use", "void", "want", "wants", "was", "were", "whereas", "wife", "will",
    "witnesses", "witnesseth", "with", "without", "years",
}


def _split_fused_token(token: str) -> str:
    raw = token or ""
    core = re.sub(r"[^A-Za-z]", "", raw)
    if len(core) < 14:
        return token

    s = core.lower()
    n = len(s)
    best: List[Optional[tuple]] = [None] * (n + 1)
    best[0] = (0, 0, [])
    max_word_len = max(len(w) for w in _FUSED_WORD_LEXICON)

    for i in range(n):
        if best[i] is None:
            continue
        score_i, covered_i, parts_i = best[i]
        upper = min(n, i + max_word_len)
        for j in range(i + 1, upper + 1):
            piece = s[i:j]
            is_word = piece in _FUSED_WORD_LEXICON
            score_j = score_i + (8 if is_word else -3 * len(piece))
            covered_j = covered_i + (len(piece) if is_word else 0)
            cand = (score_j, covered_j, parts_i + [piece])
            cur = best[j]
            if cur is None or cand[0] > cur[0] or (cand[0] == cur[0] and cand[1] > cur[1]):
                best[j] = cand

    if best[n] is None:
        return token
    _, covered, parts = best[n]
    if covered / max(1, n) < 0.75 or len(parts) < 3:
        return token
    # Guardrail: reject noisy segmentations like "m a in t a in".
    if any(len(p) == 1 and p not in {"a", "i"} for p in parts):
        return token
    short_ratio = sum(1 for p in parts if len(p) <= 2) / max(1, len(parts))
    if short_ratio > 0.34:
        return token
    if len(parts) > max(8, n // 3):
        return token

    spaced = " ".join(parts)
    if raw and raw[0].isupper():
        spaced = spaced[:1].upper() + spaced[1:]
    return spaced


def _restore_spacing_artifacts(answer: str) -> str:
    """
    Fix common OCR/extraction artifacts where many words are fused together.
    Only reformats spacing; does not add facts.
    """
    if not answer:
        return answer

    pattern = re.compile(r"\b[A-Za-z]{14,}\b")
    return pattern.sub(lambda m: _split_fused_token(m.group(0)), answer)


def rag_answer(
    document_text: str,
    question: str,
    history: Optional[Sequence[dict]] = None,
) -> str:
    """
    Run retrieval over document_text, then answer with Groq LLM.

    history: optional list of {"role": "user"|"assistant", "content": "..."} for short multi-turn.
    """
    doc = _normalize_whitespace(document_text or "")
    doc = _fix_broken_text(doc)
    if len(doc) < 25:
        return "The document is empty or too short to search. Upload a PDF with extractable text."

    q = (question or "").strip()
    if not q:
        return "Please enter a question."

    doc = doc[:MAX_DOC_CHARS]
    chunks = chunk_document(doc)
    if not chunks:
        return "Could not process the document text."

    relevant = _retrieve_with_vector_db(doc, q, chunks)
    if not relevant:
        relevant = retrieve_relevant_chunks(q, chunks)
    # Compact context selection reduces LLM input size and latency.
    relevant = _select_compact_context_chunks(relevant, TOP_K_RETRIEVAL, MAX_CONTEXT_CHARS)
    context = "\n\n---\n\n".join(relevant)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    messages = _build_messages(context, q, history)
    try:
        out = _generate_answer_with_guardrails(messages)
        out = _repair_answer_if_needed(q, out, context)
        out = _restore_spacing_artifacts(out)
        return out
    except Exception as e:
        return f"Could not reach Ollama model ({OLLAMA_MODEL}). Is it running locally? Details: {e}"


__all__ = [
    "OLLAMA_MODEL",
    "rag_answer",
    "chunk_document",
    "retrieve_relevant_chunks",
]
