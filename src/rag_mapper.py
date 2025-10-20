import os, glob, yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Keep track of root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Normalize YAML mapping entries into a flat, uniform structure.
# Accepts a YAML object (dict with "mappings" key or list of items),
# extracts fields (source, target, description, synonyms, examples),
# cleans and joins them into text, and yields dictionaries like:
# {"source": ..., "target": ..., "text": "..."} for valid mappings.
def _iter_yaml_items(yaml_obj):
    if isinstance(yaml_obj, dict) and "mappings" in yaml_obj:
        items = yaml_obj["mappings"]
    elif isinstance(yaml_obj, list):
        items = yaml_obj
    else:
        return
    for it in items:
        if not isinstance(it, dict):
            continue
        src = str(it.get("source", "")).strip()
        tgt = str(it.get("target", "")).strip()
        desc = " ".join(map(str, it.get("description", []) if isinstance(it.get("description"), list) else [it.get("description","")])).strip()
        syns = " ".join(map(str, it.get("synonyms", []) or [])).strip()
        exs  = " ".join(map(str, it.get("examples", []) or [])).strip()
        if not tgt:
            continue
        yield {"source": src, "target": tgt, "text": " | ".join([x for x in [src, syns, desc, exs] if x])}

# Recursively load all YAML files under the given directory and extract mapping data.
# Uses _iter_yaml_items() to normalize each YAML file's mappings, collecting text for embeddings
# and corresponding target labels. Returns (docs, labels) lists for downstream training or indexing.
# Raises ValueError if no valid mappings are found.
def load_yaml_corpus(yaml_dir):
    docs, labels = [], []
    for path in glob.glob(os.path.join(BASE_DIR, yaml_dir, "**/*.y*ml"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if obj is None:
            continue
        for item in _iter_yaml_items(obj):
            text = item["text"] if item["text"] else item["target"]
            docs.append(text)
            labels.append(item["target"])
    if not docs:
        raise ValueError("No mappings found in YAML directory.")
    return docs, labels

# Build a text descriptor summarizing a DataFrame column.
# Combines the column header, a numeric ratio estimate, and up to N sample values
# into a single string useful for feature description or schema embedding.
# Returns a compact representation like: "header:col_name numeric_ratio:0.75 values: ..."
def build_column_descriptor(header, series, sample_rows=200):
    s = series.dropna().astype(str).head(sample_rows).tolist()
    sample = " ".join(s)
    meta = f"header:{header}"
    try:
        numeric_ratio = pd.to_numeric(series.dropna(), errors="coerce").notna().mean()
    except Exception:
        numeric_ratio = 0.0
    hint = f" numeric_ratio:{numeric_ratio:.2f}"
    return f"{meta} {hint} values: {sample}"

# RAGMapper: a lightweight retrieval-augmented mapping model.
# Combines TF-IDF character n-gram vectorization, nearest-neighbor search,
# and optional logistic regression classification to suggest label mappings.

# Core methods:
# - __init__: sets up vectorizer, neighbor model, classifier, and label encoder.
# - fit(docs, labels): trains all components and stores vectorized knowledge base.
# - retrieve(query_vec): finds nearest neighbor documents by cosine similarity.
# - suggest(column_text): encodes query, retrieves similar entries, optionally
#   refines with classifier probabilities, and ranks label suggestions by score.
class RAGMapper:
    def __init__(self, n_neighbors=8, max_features=50000):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=max_features)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.clf = LogisticRegression(max_iter=2000)
        self.le = LabelEncoder()
        self._fitted = False

    def fit(self, docs, labels):
        X = self.vec.fit_transform(docs)
        y = self.le.fit_transform(labels)
        self._has_classifier = len(self.le.classes_) >= 2
        if self._has_classifier:
            self.clf.fit(X, y)
        self.nn.fit(X)
        self._kb_X = X
        self._kb_docs = docs
        self._kb_labels = labels
        self._fitted = True
        return self

    def retrieve(self, query_vec, top_k=8):
        dists, idx = self.nn.kneighbors(query_vec, n_neighbors=min(top_k, self._kb_X.shape[0]))
        return dists[0], idx[0]

    def suggest(self, column_text, k_retrieve=8, k_return=5, restrict_to_retrieved=True):
        assert self._fitted
        q = self.vec.transform([column_text])
        dists, idx = self.retrieve(q, top_k=k_retrieve)
        retrieved_labels = [self._kb_labels[i] for i in idx]
        sims = 1.0 - dists
        agg = {}
        for lab, s in zip(retrieved_labels, sims):
            agg[lab] = max(agg.get(lab, 0.0), float(s))
        clf_scores = {}
        if getattr(self, "_has_classifier", False):
            proba = self.clf.predict_proba(q)[0]
            for lab_idx, p in enumerate(proba):
                clf_scores[self.le.inverse_transform([lab_idx])[0]] = float(p)
            if restrict_to_retrieved:
                subset = {lab: clf_scores.get(lab, 0.0) for lab in agg.keys()}
                s = sum(subset.values())
                clf_scores = {lab: (subset.get(lab,0.0)/s if s>0 else 0.0) for lab in agg.keys()}
        out = []
        for lab in agg.keys():
            r = agg[lab]
            c = clf_scores.get(lab, 0.0) if getattr(self, "_has_classifier", False) else 0.0
            score = 0.6*r + 0.4*c if getattr(self, "_has_classifier", False) else r
            out.append((lab, score, r, c))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k_return]
    
# Serialize and save the trained RAGMapper components to disk as a .npz file.
# Stores vectorizer, nearest-neighbor model, optional classifier, label encoder,
# and knowledge base docs/labels for later reuse or inference loading.
# Returns the output file path.
def save_index(mapper: RAGMapper, path="rag_index.npz"):
    joblib.dump({
        "vec": mapper.vec,
        "nn": mapper.nn,
        "clf": mapper.clf if getattr(mapper, "_has_classifier", False) else None,
        "le": mapper.le,
        "kb_docs": mapper._kb_docs,
        "kb_labels": mapper._kb_labels
    }, path)
    return path

# Load a saved RAGMapper model from disk and reconstruct its state.
# Restores vectorizer, nearest-neighbor model, optional classifier, label encoder,
# and knowledge base data, then rebuilds the vectorized document matrix.
# Marks the mapper as fitted and returns the ready-to-use RAGMapper instance.
def load_index(path="rag_index.npz") -> RAGMapper:
    bundle = joblib.load(path)
    m = RAGMapper()
    m.vec = bundle["vec"]
    m.nn = bundle["nn"]
    m.clf = bundle["clf"] if bundle["clf"] is not None else m.clf
    m.le  = bundle["le"]
    m._kb_docs = bundle["kb_docs"]
    m._kb_labels = bundle["kb_labels"]
    m._kb_X = m.vec.transform(m._kb_docs)
    m._has_classifier = bundle["clf"] is not None
    m._fitted = True
    return m