import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

from . import config


@st.cache_resource(show_spinner=False)
def get_chat_client():
    return OpenAI(api_key=config.CHAT_API_KEY, base_url=config.BASE_URL_LLM)


@st.cache_resource(show_spinner=False)
def get_embed_client():
    return OpenAI(api_key=config.EMBED_API_KEY, base_url=config.BASE_URL_EMBED)


def _require_file(path, hint):
    if not os.path.exists(path):
        st.error(
            f"Data file not found: `{path}`.\n\n{hint}\n\n"
            f"Run `scripts/export_data_for_streamlit.py` at the end of your Colab notebook "
            f"to generate it, then place it in this app's `data/` folder. See README.md."
        )
        st.stop()
    return path


@st.cache_data(show_spinner="Loading RACI process data...")
def load_df():
    path = _require_file(
        os.path.join(config.DATA_DIR, "raci_processes.parquet"),
        "This should be your raw RACI table (with the Semantic_Document column), exported to parquet.",
    )
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading exploded RACI role table...")
def load_structured_df():
    path = _require_file(
        os.path.join(config.DATA_DIR, "structured_raci.parquet"),
        "This is the exploded R/A/C/I -> one-row-per-role table built in your Colab notebook.",
    )
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading process embeddings...")
def load_embeddings():
    emb_path = _require_file(
        os.path.join(config.DATA_DIR, "embeddings.npy"),
        "Semantic_Document embeddings (bge-m3), one row per RACI process.",
    )
    ids_path = _require_file(
        os.path.join(config.DATA_DIR, "embedding_process_ids.npy"),
        "The matching process_id (df index) for each embedding row.",
    )
    embeddings = np.load(emb_path)
    embedded_ids = np.load(ids_path)
    return embeddings, embedded_ids


@st.cache_data(show_spinner="Loading Tupoksi (job duties) data...")
def load_jabatan_tugas():
    path = _require_file(
        os.path.join(config.DATA_DIR, "jabatan_tugas.pkl"),
        "Dict of {sebutan_normalized: [list of Tugas Pokok strings]} built in your Colab notebook.",
    )
    with open(path, "rb") as f:
        jabatan_tugas = pickle.load(f)
    return jabatan_tugas


@st.cache_data(show_spinner=False)
def get_known_roles(structured_df):
    return structured_df["role_normalized"].unique().tolist()


@st.cache_data(show_spinner=False)
def get_known_processes(df):
    return df["PROSES BISNIS"].unique().tolist()


@st.cache_data(show_spinner=False)
def get_known_jabatan(jabatan_tugas):
    return list(jabatan_tugas.keys())


@st.cache_data(show_spinner=False)
def get_level1_lookup(df):
    level1_df = df[df["LEVEL"] == "Lv.1"]
    return dict(zip(
        level1_df["HIERARCHY ID"].astype(str).str.split(".").str[0],
        level1_df["PROSES BISNIS"],
    ))
