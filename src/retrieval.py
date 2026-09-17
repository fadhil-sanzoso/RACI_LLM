import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def split_roles(cell):
    if pd.isna(cell):
        return []
    return [r.strip() for r in str(cell).split(";") if r.strip()]


def normalize_role(s):
    s = str(s).strip()
    return re.sub(r"\s+", " ", s)


def expand_aliases(text):
    for pattern, full in config.ALIAS_MAP.items():
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    return text


def resolve_entity(text, candidates, threshold=None):
    threshold = threshold if threshold is not None else config.ENTITY_MATCH_THRESHOLD
    match = rf_process.extractOne(text, candidates, scorer=fuzz.token_sort_ratio)
    return (match[0], match[1]) if match and match[1] >= threshold else (None, 0)


def semantic_search(query, embed_client, embeddings, embedded_ids, df, top_k=5):
    resp = embed_client.embeddings.create(model=config.EMBED_MODEL, input=[query])
    q_vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(q_vec, embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        pid = embedded_ids[i]
        results.append({
            "process_id": pid,
            "process_name": df.loc[pid, "PROSES BISNIS"],
            "similarity": float(sims[i]),
            "semantic_document": df.loc[pid, "Semantic_Document"],
        })
    return results


def get_process_vector(process_id, embeddings, embedded_ids):
    """Look up a process's already-computed embedding instead of calling the embed API again --
    saves cost/latency on every request in the deployed app, since embeddings.npy already
    has every process embedded once during the Colab export step."""
    matches = np.where(embedded_ids == process_id)[0]
    if len(matches) == 0:
        return None
    return embeddings[matches[0]].reshape(1, -1)


def query_by_role(structured_df, role_name, raci_type=None, limit=15):
    mask = structured_df["role_normalized"].str.contains(role_name, case=False, na=False)
    if raci_type:
        mask &= structured_df["raci_type"] == raci_type
    matches = structured_df[mask]
    return matches.head(limit), len(matches)


def query_by_process(df, process_name):
    mask = df["PROSES BISNIS"].str.contains(process_name, case=False, na=False)
    return df[mask]
