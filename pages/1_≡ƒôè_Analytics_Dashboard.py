import pandas as pd
import plotly.express as px
import streamlit as st
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process

from src import config
from src.ask_engine import build_process_context
from src.alignment import format_alignment_block
from src.data_loader import (
    get_embed_client,
    get_known_jabatan,
    get_known_processes,
    get_known_roles,
    get_level1_lookup,
    load_df,
    load_embeddings,
    load_jabatan_tugas,
    load_structured_df,
)
from src.retrieval import query_by_process

st.set_page_config(page_title="Analytics Dashboard - RACI PLN", page_icon="📊", layout="wide")
st.title("📊 Analytics Dashboard — RACI & Tupoksi PLN")

try:
    df = load_df()
    structured_df = load_structured_df()
    embeddings, embedded_ids = load_embeddings()
    jabatan_tugas = load_jabatan_tugas()
    known_roles = get_known_roles(structured_df)
    known_processes = get_known_processes(df)
    known_jabatan_tupoksi = get_known_jabatan(jabatan_tugas)
    level1_lookup = get_level1_lookup(df)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# --- Overview metrics ---
st.subheader("Ringkasan")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Proses Bisnis", len(df))
col2.metric("Role Unik (RACI)", len(known_roles))
col3.metric("Jabatan dengan Tupoksi", len(known_jabatan_tupoksi))
col4.metric("Total Baris Tugas Pokok", sum(len(v) for v in jabatan_tugas.values()))

st.divider()

# --- RACI type distribution ---
st.subheader("Distribusi Tipe RACI")
raci_counts = structured_df["raci_type"].value_counts().reindex(["R", "A", "C", "I"]).fillna(0)
raci_counts.index = raci_counts.index.map(config.RACI_LABELS)
fig1 = px.bar(raci_counts, text_auto=True, color=raci_counts.index,
              color_discrete_sequence=px.colors.qualitative.Safe)
fig1.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah baris")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- Top roles by RACI type ---
st.subheader("Role Paling Banyak Terlibat")
c1, c2 = st.columns([1, 3])
with c1:
    raci_pick = st.selectbox("Tipe RACI", ["R", "A", "C", "I"], format_func=lambda x: config.RACI_LABELS[x])
    top_n = st.slider("Jumlah role ditampilkan", 5, 30, 15)
subset = structured_df[structured_df["raci_type"] == raci_pick]
top_roles = subset["role_normalized"].value_counts().head(top_n).sort_values()
fig2 = px.bar(top_roles, orientation="h", text_auto=True)
fig2.update_layout(
    showlegend=False, yaxis_title="", xaxis_title="Jumlah proses",
    height=max(400, top_n * 28),
)
with c2:
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Process count by Level 1 category ---
st.subheader("Jumlah Proses per Kategori (Level 1)")
tmp = structured_df.drop_duplicates(subset=["process_id"]).copy()
tmp["root"] = tmp["hierarchy_id"].astype(str).str.split(".").str[0]
cat_counts = tmp.groupby("root").size()
cat_counts.index = cat_counts.index.map(lambda r: level1_lookup.get(r, f"Kategori {r}"))
cat_counts = cat_counts.sort_values(ascending=False)
fig3 = px.bar(cat_counts, text_auto=True)
fig3.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah proses", xaxis_tickangle=-40)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Tupoksi coverage gap ---
st.subheader("Cakupan Tupoksi per Role RACI")
st.caption(
    "Role yang muncul di RACI matrix tapi TIDAK memiliki data Tugas Pokok yang cocok "
    "(misalnya level Direktur/GM, yang memang belum tersedia di katalog Tupoksi ini)."
)


@st.cache_data(show_spinner="Menghitung cakupan tupoksi...")
def compute_tupoksi_coverage(known_roles_tuple, known_jabatan_tuple, threshold):
    rows = []
    for role in known_roles_tuple:
        match = rf_process.extractOne(role, known_jabatan_tuple, scorer=rf_fuzz.token_sort_ratio)
        found = bool(match and match[1] >= threshold)
        rows.append({
            "role": role,
            "tupoksi_ditemukan": found,
            "jabatan_cocok": match[0] if found else None,
            "skor_kecocokan": round(match[1], 1) if match else 0.0,
        })
    return pd.DataFrame(rows)


coverage_df = compute_tupoksi_coverage(
    tuple(known_roles), tuple(known_jabatan_tupoksi), config.JABATAN_MATCH_THRESHOLD
)
gap_df = coverage_df[~coverage_df["tupoksi_ditemukan"]].sort_values("role")

gc1, gc2 = st.columns(2)
gc1.metric("Role dengan data Tupoksi", int(coverage_df["tupoksi_ditemukan"].sum()))
gc2.metric("Role TANPA data Tupoksi (gap)", int((~coverage_df["tupoksi_ditemukan"]).sum()))

with st.expander(f"Lihat {len(gap_df)} role tanpa data Tupoksi"):
    st.dataframe(gap_df[["role"]].reset_index(drop=True), use_container_width=True)
    st.download_button(
        "Unduh daftar gap (CSV)",
        gap_df.to_csv(index=False).encode("utf-8"),
        file_name="tupoksi_coverage_gap.csv",
        mime="text/csv",
    )

st.divider()

# --- On-demand alignment checker ---
st.subheader("Cek Kesesuaian Tugas Pokok (per Proses)")
st.caption("Memanggil API embedding hanya saat tombol ditekan — tidak dihitung otomatis untuk seluruh data.")

process_pick = st.selectbox(
    "Pilih proses bisnis", options=known_processes, index=None, placeholder="Ketik untuk mencari proses..."
)

if process_pick and st.button("Cek kesesuaian tugas pokok"):
    embed_client = get_embed_client()
    row = query_by_process(df, process_pick).iloc[0]
    context, role_candidates, proc_vec = build_process_context(row, embeddings, embedded_ids)
    st.markdown(f"**Proses:** {row['PROSES BISNIS']} (ID: {row['HIERARCHY ID']})")
    if proc_vec is None:
        st.warning("Embedding untuk proses ini tidak ditemukan di cache — coba proses lain.")
    else:
        seen = set()
        for role_name, code in role_candidates:
            if role_name in seen:
                continue
            seen.add(role_name)
            st.markdown(f"**[{config.RACI_LABELS[code]}] {role_name}**")
            block = format_alignment_block(role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi)
            st.text(block)
