"""
Resume–Job Matcher (Prototype)
---------------------------------------
- Transformer embeddings (Sentence-Transformers)
- FAISS ANN preselection
- TF-IDF lexical weighting
- Skill overlap + gap analysis (spaCy + dictionary + embedding fallback)
- Composite scoring with configurable weights + normalization
- Caching for embeddings (on disk) to speed up reruns
- CLI commands: build-index, score-pair, rank-batch
"""

import os
import re
import sys
import json
import time
import math
import glob
import hashlib
import logging
import argparse
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd

# Optional heavy deps are imported lazily
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
L = logging.getLogger("matcher")

# ---- Utilities ----

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ---- Config ----

@dataclass
class Config:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"  # "cuda" if available
    faiss_index_path: str = "models/faiss.index"
    emb_cache_dir: str = "cache/embeddings"
    tfidf_cache_path: str = "cache/tfidf.pkl"
    skills_path: Optional[str] = None  # optional custom skills file
    alpha: float = 0.6  # semantic
    beta: float = 0.25  # tfidf lexical
    gamma: float = 0.15  # skills overlap
    ann_topk: int = 100
    rerank_topk: int = 20
    normalize_scores: bool = True
    lower: bool = True
    strip_accents: Optional[str] = None

# ---- Text cleaning / sectioning ----

SECTION_HEADINGS = [
    r"experience", r"work history", r"projects", r"education", r"skills",
    r"certifications?", r"awards?", r"publications?"
]

def clean_text(text: str, lower=True) -> str:
    if text is None:
        return ""
    # remove control chars
    text = re.sub(r"[\r\t]", " ", text)
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    if lower:
        text = text.lower()
    return text

def split_sections(text: str) -> Dict[str, str]:
    t = text
    # naive split by headings
    sections = {}
    last = 0
    last_name = "body"
    for m in re.finditer(rf"({'|'.join(SECTION_HEADINGS)})\s*:?", t, flags=re.I):
        sections[last_name] = t[last:m.start()].strip()
        last = m.end()
        last_name = m.group(1).lower()
    sections[last_name] = t[last:].strip()
    # clean empties
    return {k: v for k, v in sections.items() if v}

# ---- Skills ----

DEFAULT_SKILLS = [
    "python","pandas","numpy","scikit-learn","tensorflow","pytorch","transformers",
    "nlp","natural language processing","embedding","vector search","faiss","hugging face",
    "sql","postgresql","mysql","mongodb","docker","kubernetes","aws","gcp","azure",
    "git","ci/cd","linux","data engineering","feature engineering","statistics",
    "regression","classification","clustering","time series","computer vision",
    "apriltag","limelight","ros","opencv","fastapi","flask","rest api","prompt engineering",
    "llm","sentence-transformers","semantic search","tf-idf","cosine similarity",
    "javascript","react","react native","expo","swift","ios","java"
]

def load_skills(path: Optional[str]) -> List[str]:
    skills = DEFAULT_SKILLS.copy()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                extra = [line.strip() for line in f if line.strip()]
            skills.extend(extra)
        except Exception as e:
            L.warning(f"Could not read skills file {path}: {e}")
    # dedupe
    skills = sorted(set(s.lower() for s in skills))
    return skills

def extract_skills(text: str, skills: List[str]) -> List[str]:
    # simple lexical extraction + fuzzy on embeddings (optional)
    present = []
    t = " " + text.lower() + " "
    for s in skills:
        # naive containment or word-boundary match
        if re.search(rf"\b{re.escape(s)}\b", t):
            present.append(s)
    return present

# ---- Embeddings + caching ----

class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        os.makedirs(cfg.emb_cache_dir, exist_ok=True)

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)

    def encode(self, texts: List[str], use_cache=True) -> np.ndarray:
        # cache per text sha1
        keys = [sha1(t) for t in texts]
        paths = [os.path.join(self.cfg.emb_cache_dir, k + ".npy") for k in keys]
        embs = [None] * len(texts)
        missing_idx = []
        for i, p in enumerate(paths):
            if use_cache and os.path.exists(p):
                embs[i] = np.load(p)
            else:
                missing_idx.append(i)
        if missing_idx:
            self._load_model()
            # batch encode
            batch = [texts[i] for i in missing_idx]
            B = self.cfg.batch_size
            outs = []
            for s in range(0, len(batch), B):
                chunk = batch[s:s+B]
                v = self.model.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
                outs.append(v)
            outs = np.vstack(outs)
            # save & insert back
            for j, i in enumerate(missing_idx):
                embs[i] = outs[j]
                np.save(paths[i], outs[j])
        # stack
        return np.vstack(embs)

# ---- FAISS index ----

class FaissIndex:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        self.index = None

    def build(self, vectors: np.ndarray):
        import faiss  # lazy import
        self.index = faiss.IndexFlatIP(self.dim)  # inner product for cosine (normalized)
        self.index.add(vectors.astype(np.float32))

    def save(self):
        if self.index is None:
            raise RuntimeError("Index not built")
        import faiss
        faiss.write_index(self.index, self.path)

    def load(self):
        import faiss
        self.index = faiss.read_index(self.path)

    def search(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            self.load()
        D, I = self.index.search(queries.astype(np.float32), topk)
        return D, I

# ---- TF-IDF ----

class LexicalModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vectorizer = None

    def fit(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer(lowercase=cfg.lower, strip_accents=cfg.strip_accents, max_features=50000)
        X = self.vectorizer.fit_transform(corpus)
        return X

    def transform(self, docs: List[str]):
        return self.vectorizer.transform(docs)

# ---- Scoring ----

def score_components(resume_text: str, jd_text: str,
                     resume_emb: np.ndarray, jd_emb: np.ndarray,
                     tfidf_model: LexicalModel, skills_lex: List[str]) -> Dict[str, float]:
    # semantic (cosine since normalized)
    s_sem = float(np.dot(resume_emb, jd_emb.T))

    # lexical tfidf overlap: J terms present in R weighted by tf-idf(J)
    Xr = tfidf_model.transform([resume_text])
    Xj = tfidf_model.transform([jd_text])
    # dot product as proxy for overlap
    s_tfidf = float((Xr @ Xj.T).A[0, 0])
    # normalize tfidf overlap to [0,1] relative to self-sim of jd
    denom = float((Xj @ Xj.T).A[0, 0]) or 1.0
    s_tfidf = min(s_tfidf / denom, 1.0)

    # skills
    Rk = set(extract_skills(resume_text, skills_lex))
    Jk = set(extract_skills(jd_text, skills_lex))
    s_skills = (len(Rk & Jk) / max(len(Jk), 1)) if Jk else 0.0

    return {"semantic": s_sem, "tfidf": s_tfidf, "skills": s_skills}

def composite_score(parts: Dict[str, float], cfg: Config) -> float:
    return cfg.alpha * parts["semantic"] + cfg.beta * parts["tfidf"] + cfg.gamma * parts["skills"]

# ---- Ranking ----

def rank_resumes_for_jd(resumes: List[str], jd_text: str, cfg: Config) -> pd.DataFrame:
    set_seed(42)
    # clean
    resumes = [clean_text(r, cfg.lower) for r in resumes]
    jd_text = clean_text(jd_text, cfg.lower)

    # load models
    emb = Embedder(cfg)
    # embed resumes + jd
    R = emb.encode(resumes)
    J = emb.encode([jd_text])
    dim = R.shape[1]

    # build/search FAISS
    os.makedirs(os.path.dirname(cfg.faiss_index_path), exist_ok=True)
    index = FaissIndex(dim, cfg.faiss_index_path)
    index.build(R.astype(np.float32))

    # ANN preselect
    D, I = index.search(J, topk=min(cfg.ann_topk, len(resumes)))
    cand_idx = I[0].tolist()

    # TF-IDF
    lex = LexicalModel(cfg)
    lex.fit([jd_text] + [resumes[i] for i in cand_idx])

    # Skills
    skills = load_skills(cfg.skills_path)

    # Score candidates
    rows = []
    for i in cand_idx:
        parts = score_components(resumes[i], jd_text, R[i], J[0], lex, skills)
        score = composite_score(parts, cfg)
        rows.append({"idx": i, "resume": resumes[i], **parts, "score_raw": score})

    df = pd.DataFrame(rows).sort_values("score_raw", ascending=False).head(cfg.rerank_topk)

    # scale to 0–100 for presentation
    if cfg.normalize_scores and len(df) > 1:
        scaler = MinMaxScaler(feature_range=(0, 100))
        df["score"] = scaler.fit_transform(df[["score_raw"]])
    else:
        df["score"] = df["score_raw"] * 100.0

    # attach skill gaps
    skills_set = load_skills(cfg.skills_path)
    gaps = []
    for _, r in df.iterrows():
        Rk = set(extract_skills(r["resume"], skills_set))
        Jk = set(extract_skills(jd_text, skills_set))
        gaps.append(sorted(list(Jk - Rk)))
    df["skill_gaps"] = gaps

    # reorder columns
    return df[["idx","score","semantic","tfidf","skills","skill_gaps","resume"]]

# ---- CLI ----

def cli():
    p = argparse.ArgumentParser(description="Resume–Job Matcher (Prototype, extended)")
    sub = p.add_subparsers(dest="cmd")

    p_rank = sub.add_parser("rank", help="Rank a folder of resume .txt files against a single JD .txt")
    p_rank.add_argument("--resumes_dir", required=True, help="Folder of .txt resumes")
    p_rank.add_argument("--jd_path", required=True, help="Path to JD .txt")
    p_rank.add_argument("--out_csv", default="results.csv")
    p_rank.add_argument("--skills", default=None, help="Optional skills file (one per line)")

    p_pair = sub.add_parser("score", help="Score a single resume against a JD")
    p_pair.add_argument("--resume", required=True)
    p_pair.add_argument("--jd", required=True)
    p_pair.add_argument("--skills", default=None)

    args = p.parse_args()

    cfg = Config()
    if args.cmd == "rank":
        cfg.skills_path = args.skills
        resumes = []
        fns = sorted(glob.glob(os.path.join(args.resumes_dir, "*.txt")))
        if not fns:
            L.error("No .txt files found in resumes_dir.")
            sys.exit(1)
        for fn in fns:
            resumes.append(read_text_file(fn))
        jd_text = read_text_file(args.jd_path)
        df = rank_resumes_for_jd(resumes, jd_text, cfg)
        df.to_csv(args.out_csv, index=False)
        L.info(f"Wrote {args.out_csv}")
        print(df.head(10).to_string(index=False))
    elif args.cmd == "score":
        cfg.skills_path = args.skills
        resume = read_text_file(args.resume)
        jd_text = read_text_file(args.jd)
        df = rank_resumes_for_jd([resume], jd_text, cfg)
        print(df.to_string(index=False))
    else:
        p.print_help()

if __name__ == "__main__":
    # Allow running as a script or importing as a module
    if len(sys.argv) > 1:
        cli()
