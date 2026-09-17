import re

import numpy as np
import streamlit as st
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def match_jabatan_to_tupoksi(role_name, known_jabatan_tupoksi):
    match = rf_process.extractOne(role_name, known_jabatan_tupoksi, scorer=fuzz.token_sort_ratio)
    return match[0] if match and match[1] >= config.JABATAN_MATCH_THRESHOLD else None


def is_tupoksi_query(question):
    return any(re.search(p, question, re.IGNORECASE) for p in config.TUPOKSI_KEYWORDS)


def get_tupoksi_for_role(role_name, known_jabatan_tupoksi, jabatan_tugas):
    matched = match_jabatan_to_tupoksi(role_name, known_jabatan_tupoksi)
    if not matched:
        return None, None
    return matched, jabatan_tugas.get(matched, [])


@st.cache_data(show_spinner=False)
def _embed_duties(_embed_client, duties_tuple):
    """Leading underscore on _embed_client tells Streamlit's cache not to hash the client
    object (API clients aren't reliably hashable/serializable). duties_tuple (a tuple of
    strings) is the real cache key.

    This cache lives in memory for the lifetime of the running app instance and resets on
    redeploy/restart -- a deliberate tradeoff vs. bundling a fully precomputed pickle of every
    jabatan's duty embeddings, which for ~79k duty rows would likely be too large to
    comfortably commit to a plain git repo. In practice this just means the first question
    about a given jabatan after a fresh restart costs one extra embedding call.
    """
    resp = _embed_client.embeddings.create(model=config.EMBED_MODEL, input=list(duties_tuple))
    return np.array([item.embedding for item in resp.data], dtype=np.float32)


def get_tugas_embeddings(jabatan_name, embed_client, jabatan_tugas):
    duties = jabatan_tugas.get(jabatan_name, [])
    if not duties:
        return None
    return _embed_duties(embed_client, tuple(duties))


def get_role_alignment(role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi, top_n=3):
    matched = match_jabatan_to_tupoksi(role_name, known_jabatan_tupoksi)
    if not matched:
        return {
            "found": False,
            "message": (
                f"Tidak ditemukan data tugas pokok untuk '{role_name}' "
                f"(kemungkinan level Direktur/GM, belum tersedia di katalog ini)."
            ),
        }
    duty_vecs = get_tugas_embeddings(matched, embed_client, jabatan_tugas)
    if duty_vecs is None:
        return {"found": False, "message": f"Jabatan '{matched}' ditemukan tapi tidak memiliki data tugas pokok."}
    duties = jabatan_tugas[matched]
    sims = cosine_similarity(proc_vec, duty_vecs)[0]
    ranked = sorted(zip(duties, sims), key=lambda x: -x[1])[:top_n]
    return {
        "found": True,
        "matched_jabatan": matched,
        "top_duties": [(d, float(s)) for d, s in ranked],
        "avg_similarity": float(np.mean(sims)),
        "total_duties": len(duties),
    }


def format_alignment_block(role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi, top_n=3):
    result = get_role_alignment(role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi, top_n=top_n)
    if not result["found"]:
        return f"  [Tugas Pokok - {role_name}] {result['message']}"
    total, shown = result["total_duties"], len(result["top_duties"])
    lines = [
        f"  [Tugas Pokok - {result['matched_jabatan']}] "
        f"(rata-rata similarity dari semua {total} tugas: {result['avg_similarity']:.2f}; "
        f"menampilkan {shown} tugas paling relevan dari total {total})"
    ]
    for duty, score in result["top_duties"]:
        lines.append(f"    - ({score:.2f}) {duty}")
    return "\n".join(lines)
