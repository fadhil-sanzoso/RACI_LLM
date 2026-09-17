import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
        .main > div { padding-top: 1.2rem; }

        .dash-header {
            background: linear-gradient(135deg, #0B3C5D 0%, #145DA0 45%, #2E9CCA 100%);
            padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
            box-shadow: 0 8px 26px rgba(11, 60, 93, 0.28);
        }
        .dash-header h1 { color:#fff; margin:0; font-size:1.6rem; font-weight:700; letter-spacing:-0.3px; }
        .dash-header p  { color:rgba(255,255,255,0.82); margin:.35rem 0 0 0; font-size:.93rem; }

        div[data-testid="stMetric"] {
            background:#fff; border:1px solid rgba(11,60,93,.10); border-radius:14px;
            padding:.9rem 1.1rem; box-shadow:0 2px 10px rgba(0,0,0,.04);
            transition:transform .15s ease, box-shadow .15s ease;
        }
        div[data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 6px 18px rgba(11,60,93,.13); }
        div[data-testid="stMetricLabel"] { font-weight:600; color:#4a5568; font-size:.85rem; }
        div[data-testid="stMetricValue"] { color:#0B3C5D; }

        .sec-title {
            font-size:1.05rem; font-weight:700; color:#0B3C5D;
            margin:.3rem 0 .6rem 0; padding-bottom:.3rem;
            border-bottom:3px solid #2E9CCA; display:inline-block;
        }
        .sub-note { color:#64748b; font-size:.85rem; margin:-.2rem 0 .8rem 0; }

        .insight-box {
            background:#F8FAFC; border-left:4px solid #2E9CCA; border-radius:8px;
            padding:.8rem 1rem; margin:.6rem 0 1rem 0; font-size:.9rem; color:#334155;
        }
        .insight-warn { background:#FEF6F3; border-left-color:#E76F51; }
        .insight-good { background:#F1FAF6; border-left-color:#2A9D8F; }

        button[data-baseweb="tab"] { border-radius:10px 10px 0 0; font-weight:600; }
        div[data-baseweb="tab-list"] { gap:4px; }
        div[data-testid="stExpander"] { border-radius:12px; border:1px solid rgba(11,60,93,.12); }
        .stButton>button { border-radius:10px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dash-header">
        <h1>📊 Analytics Dashboard — RACI &amp; Tupoksi PLN</h1>
        <p>Audit tata kelola matriks RACI: kejelasan akuntabilitas, konsentrasi beban peran,
        kompleksitas proses, dan cakupan tugas pokok jabatan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================
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

NAVY, BLUE, SKY, AMBER, CORAL, TEAL = "#0B3C5D", "#145DA0", "#57C4E5", "#F4A259", "#E76F51", "#2A9D8F"
RACI_COLORS = {"Responsible": BLUE, "Accountable": NAVY, "Consulted": SKY, "Informed": AMBER}


# ============================================================
# DERIVED ANALYTICS
# ============================================================
@st.cache_data(show_spinner="Menyiapkan analitik...")
def build_process_profile(sdf):
    """Satu baris per proses: jumlah R/A/C/I, total peran, kategori Level 1."""
    prof = (
        sdf.pivot_table(index="process_id", columns="raci_type",
                        values="role_normalized", aggfunc="count")
        .reindex(columns=["R", "A", "C", "I"])
        .fillna(0)
        .astype(int)
    )
    prof.columns = [f"n_{c}" for c in prof.columns]
    prof["total_peran"] = prof.sum(axis=1)
    prof["beban_koordinasi"] = prof["n_C"] + prof["n_I"]

    meta = sdf.drop_duplicates(subset=["process_id"]).set_index("process_id")
    prof["hierarchy_id"] = meta["hierarchy_id"].astype(str)
    prof["root"] = prof["hierarchy_id"].str.split(".").str[0]
    return prof.reset_index()


@st.cache_data(show_spinner=False)
def build_role_profile(sdf):
    """Satu baris per role: keterlibatan per tipe RACI."""
    rp = (
        sdf.pivot_table(index="role_normalized", columns="raci_type",
                        values="process_id", aggfunc="nunique")
        .reindex(columns=["R", "A", "C", "I"])
        .fillna(0)
        .astype(int)
    )
    rp.columns = [f"n_{c}" for c in rp.columns]
    rp["total"] = rp.sum(axis=1)
    rp["rasio_akuntabel"] = np.where(rp["total"] > 0, rp["n_A"] / rp["total"], 0)
    rp["rasio_eksekusi"] = np.where(rp["total"] > 0, rp["n_R"] / rp["total"], 0)
    return rp.reset_index().rename(columns={"role_normalized": "role"})


@st.cache_data(show_spinner=False)
def build_process_name_lookup(_df):
    """Peta HIERARCHY ID -> nama proses, dipakai agar tabel temuan mudah dibaca."""
    try:
        return dict(zip(_df["HIERARCHY ID"].astype(str), _df["PROSES BISNIS"].astype(str)))
    except Exception:
        return {}


proc_prof = build_process_profile(structured_df)
role_prof = build_role_profile(structured_df)
name_lookup = build_process_name_lookup(df)
proc_prof["nama_proses"] = proc_prof["hierarchy_id"].map(name_lookup).fillna("—")
proc_prof["kategori"] = proc_prof["root"].map(lambda r: level1_lookup.get(r, f"Kategori {r}"))

# --- Governance flags ---
proc_prof["tanpa_akuntabel"] = proc_prof["n_A"] == 0
proc_prof["akuntabel_ganda"] = proc_prof["n_A"] > 1
proc_prof["tanpa_pelaksana"] = proc_prof["n_R"] == 0
proc_prof["sehat"] = (proc_prof["n_A"] == 1) & (proc_prof["n_R"] >= 1)

total_proc = len(proc_prof)
n_sehat = int(proc_prof["sehat"].sum())
health_pct = (n_sehat / total_proc * 100) if total_proc else 0

# ============================================================
# TOP-LEVEL KPI
# ============================================================
st.markdown('<p class="sec-title">🧭 Ringkasan Eksekutif</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Proses Bisnis", f"{total_proc:,}")
k2.metric("Role Unik (RACI)", f"{len(known_roles):,}")
k3.metric(
    "Proses Sehat", f"{health_pct:.0f}%",
    help="Proses dengan tepat satu Accountable dan minimal satu Responsible.",
)
k4.metric(
    "Rata-rata Peran / Proses", f"{proc_prof['total_peran'].mean():.1f}",
    help="Semakin tinggi, semakin banyak pihak terlibat dalam satu proses.",
)
k5.metric("Jabatan dengan Tupoksi", f"{len(known_jabatan_tupoksi):,}")

st.write("")

tab_health, tab_role, tab_proc, tab_matrix, tab_tupoksi, tab_check = st.tabs([
    "🩺 Kesehatan RACI",
    "👥 Beban & Konsentrasi Peran",
    "🗂️ Struktur & Kompleksitas Proses",
    "🔥 Matriks Peran × Kategori",
    "📋 Cakupan Tupoksi",
    "🔎 Cek Kesesuaian",
])

# ============================================================
# TAB 1 — GOVERNANCE HEALTH
# ============================================================
with tab_health:
    st.markdown('<p class="sec-title">Audit Akuntabilitas</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-note">Prinsip dasar RACI: setiap proses idealnya punya '
        '<b>tepat satu Accountable</b> (satu pemilik keputusan) dan <b>minimal satu Responsible</b> '
        '(pelaksana). Penyimpangan dari dua aturan ini adalah indikasi ambiguitas tata kelola.</p>',
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns([1, 1.6])

    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_pct,
            number={"suffix": "%", "font": {"size": 40, "color": NAVY}},
            title={"text": "Skor Kesehatan RACI", "font": {"size": 14, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": TEAL if health_pct >= 70 else (AMBER if health_pct >= 40 else CORAL)},
                "steps": [
                    {"range": [0, 40], "color": "#FDEDE8"},
                    {"range": [40, 70], "color": "#FEF6E8"},
                    {"range": [70, 100], "color": "#EAF7F3"},
                ],
                "threshold": {"line": {"color": NAVY, "width": 3}, "thickness": .8, "value": 90},
            },
        ))
        gauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)

    with g2:
        issues = pd.Series({
            "Tanpa Accountable": int(proc_prof["tanpa_akuntabel"].sum()),
            "Accountable Ganda": int(proc_prof["akuntabel_ganda"].sum()),
            "Tanpa Responsible": int(proc_prof["tanpa_pelaksana"].sum()),
        }).sort_values()
        fig = px.bar(issues, orientation="h", text_auto=True,
                     color=issues.index,
                     color_discrete_sequence=[AMBER, CORAL, "#C1121F"])
        fig.update_traces(marker_line_width=0, textposition="outside")
        fig.update_layout(
            showlegend=False, xaxis_title="Jumlah proses bermasalah", yaxis_title="",
            height=260, margin=dict(t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Temuan Utama", font=dict(size=14, color="#475569")),
        )
        st.plotly_chart(fig, use_container_width=True)

    worst = issues.idxmax() if issues.max() > 0 else None
    if worst:
        st.markdown(
            f'<div class="insight-box insight-warn">💡 <b>Temuan terbesar:</b> {issues.max()} proses '
            f'masuk kategori <b>{worst}</b>. Kategori ini biasanya jadi prioritas perbaikan pertama '
            f'karena langsung berdampak pada kejelasan siapa yang mengambil keputusan.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight-box insight-good">✅ Tidak ditemukan pelanggaran aturan dasar RACI '
            'pada dataset ini.</div>', unsafe_allow_html=True,
        )

    st.markdown("**Daftar proses bermasalah**")
    issue_pick = st.radio(
        "Filter temuan", ["Tanpa Accountable", "Accountable Ganda", "Tanpa Responsible"],
        horizontal=True, label_visibility="collapsed",
    )
    flag_col = {
        "Tanpa Accountable": "tanpa_akuntabel",
        "Accountable Ganda": "akuntabel_ganda",
        "Tanpa Responsible": "tanpa_pelaksana",
    }[issue_pick]

    issue_df = (
        proc_prof[proc_prof[flag_col]]
        [["hierarchy_id", "nama_proses", "kategori", "n_R", "n_A", "n_C", "n_I", "total_peran"]]
        .rename(columns={
            "hierarchy_id": "ID", "nama_proses": "Proses Bisnis", "kategori": "Kategori",
            "n_R": "R", "n_A": "A", "n_C": "C", "n_I": "I", "total_peran": "Total Peran",
        })
        .sort_values("ID")
        .reset_index(drop=True)
    )
    st.dataframe(issue_df, use_container_width=True, height=320)
    if len(issue_df):
        st.download_button(
            f"⬇️ Unduh {len(issue_df)} temuan ({issue_pick}) — CSV",
            issue_df.to_csv(index=False).encode("utf-8"),
            file_name=f"temuan_{flag_col}.csv", mime="text/csv",
        )

    st.divider()
    st.markdown("**Kesehatan RACI per Kategori (Level 1)**")
    st.caption("Membantu mengidentifikasi area organisasi mana yang tata kelolanya paling perlu dibenahi.")
    by_cat = (
        proc_prof.groupby("kategori")
        .agg(total=("sehat", "size"), sehat=("sehat", "sum"))
        .assign(persen_sehat=lambda d: (d["sehat"] / d["total"] * 100).round(1))
        .sort_values("persen_sehat")
        .reset_index()
    )
    fig = px.bar(by_cat, x="persen_sehat", y="kategori", orientation="h", text="persen_sehat",
                 color="persen_sehat", color_continuous_scale=[CORAL, AMBER, TEAL],
                 range_color=[0, 100], custom_data=["total", "sehat"])
    fig.update_traces(
        texttemplate="%{text}%", textposition="outside", marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Sehat: %{customdata[1]} dari %{customdata[0]} proses<extra></extra>",
    )
    fig.update_layout(
        height=max(320, len(by_cat) * 32), xaxis_title="% proses sehat", yaxis_title="",
        coloraxis_showscale=False, margin=dict(t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_range=[0, 108],
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — ROLE LOAD & CONCENTRATION
# ============================================================
with tab_role:
    st.markdown('<p class="sec-title">Beban & Konsentrasi Peran</p>', unsafe_allow_html=True)

    top_a = role_prof.nlargest(1, "n_A")
    if len(top_a):
        r = top_a.iloc[0]
        share = r["n_A"] / max(proc_prof["n_A"].sum(), 1) * 100
        st.markdown(
            f'<div class="insight-box">💡 <b>{r["role"]}</b> memegang akuntabilitas (A) pada '
            f'<b>{int(r["n_A"])} proses</b> — sekitar {share:.1f}% dari seluruh penugasan Accountable. '
            f'Konsentrasi tinggi pada satu peran berarti proses tersebut rentan menjadi '
            f'<i>bottleneck</i> pengambilan keputusan.</div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Beban Akuntabilitas Tertinggi**")
        st.caption("Siapa yang paling banyak jadi pemilik keputusan.")
        n_show = st.slider("Jumlah role", 5, 25, 12, key="slider_a")
        top_acc = role_prof.nlargest(n_show, "n_A").sort_values("n_A")
        fig = px.bar(top_acc, x="n_A", y="role", orientation="h", text="n_A",
                     color_discrete_sequence=[NAVY])
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            height=max(340, n_show * 30), xaxis_title="Jumlah proses sebagai Accountable",
            yaxis_title="", margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Keterlibatan Total Tertinggi**")
        st.caption("Peran yang paling sering muncul di matriks, dalam bentuk apa pun.")
        n_show2 = st.slider("Jumlah role", 5, 25, 12, key="slider_tot")
        top_tot = role_prof.nlargest(n_show2, "total").sort_values("total")
        melted = top_tot.melt(
            id_vars="role", value_vars=["n_R", "n_A", "n_C", "n_I"],
            var_name="tipe", value_name="jumlah",
        )
        melted["tipe"] = melted["tipe"].str[-1].map(config.RACI_LABELS)
        fig = px.bar(melted, x="jumlah", y="role", color="tipe", orientation="h",
                     color_discrete_map=RACI_COLORS,
                     category_orders={"role": top_tot["role"].tolist()})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            height=max(340, n_show2 * 30), barmode="stack",
            xaxis_title="Jumlah proses", yaxis_title="", legend_title="",
            legend=dict(orientation="h", y=-0.12), margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Profil Peran: Pengambil Keputusan vs Pelaksana**")
    st.caption(
        "Setiap titik adalah satu role. Sumbu X = jumlah proses yang ia laksanakan (R), "
        "sumbu Y = jumlah proses yang ia pertanggungjawabkan (A). "
        "Kanan-bawah = pelaksana murni, kiri-atas = pengarah murni, kanan-atas = beban ganda berat."
    )
    min_inv = st.slider("Tampilkan role dengan minimal keterlibatan", 1, 30, 3, key="slider_scatter")
    sc = role_prof[role_prof["total"] >= min_inv].copy()
    sc["beban_total"] = sc["total"]
    fig = px.scatter(
        sc, x="n_R", y="n_A", size="beban_total", color="rasio_akuntabel",
        hover_name="role", color_continuous_scale=[SKY, BLUE, NAVY],
        size_max=38, custom_data=["total", "n_C", "n_I"],
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        hovertemplate=(
            "<b>%{hovertext}</b><br>Responsible: %{x}<br>Accountable: %{y}"
            "<br>Consulted: %{customdata[1]} · Informed: %{customdata[2]}"
            "<br>Total keterlibatan: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=480, xaxis_title="Jumlah proses sebagai Responsible (pelaksana)",
        yaxis_title="Jumlah proses sebagai Accountable (pemilik)",
        coloraxis_colorbar=dict(title="Rasio<br>Akuntabel"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Lihat tabel lengkap profil peran"):
        show = role_prof.sort_values("total", ascending=False).rename(columns={
            "role": "Role", "n_R": "R", "n_A": "A", "n_C": "C", "n_I": "I",
            "total": "Total", "rasio_akuntabel": "Rasio A", "rasio_eksekusi": "Rasio R",
        })
        show["Rasio A"] = (show["Rasio A"] * 100).round(1)
        show["Rasio R"] = (show["Rasio R"] * 100).round(1)
        st.dataframe(show.reset_index(drop=True), use_container_width=True, height=380)
        st.download_button(
            "⬇️ Unduh profil peran (CSV)",
            show.to_csv(index=False).encode("utf-8"),
            file_name="profil_peran_raci.csv", mime="text/csv",
        )

# ============================================================
# TAB 3 — PROCESS STRUCTURE & COMPLEXITY
# ============================================================
with tab_proc:
    st.markdown('<p class="sec-title">Struktur & Kompleksitas Proses</p>', unsafe_allow_html=True)

    avg_coord = proc_prof["beban_koordinasi"].mean()
    heavy = proc_prof[proc_prof["beban_koordinasi"] > proc_prof["beban_koordinasi"].quantile(.9)]
    st.markdown(
        f'<div class="insight-box">💡 Rata-rata satu proses melibatkan <b>{avg_coord:.1f} pihak</b> '
        f'yang hanya dikonsultasikan (C) atau diinformasikan (I). '
        f'Terdapat <b>{len(heavy)} proses</b> dengan beban koordinasi di 10% tertinggi — '
        f'kandidat utama untuk penyederhanaan alur agar keputusan tidak melambat.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown("**Jumlah Proses per Kategori (Level 1)**")
        cat_counts = proc_prof["kategori"].value_counts().sort_values(ascending=True)
        fig = px.bar(cat_counts, orientation="h", text_auto=True,
                     color_discrete_sequence=[BLUE])
        fig.update_traces(marker_line_width=0, textposition="outside")
        fig.update_layout(
            height=max(340, len(cat_counts) * 30), showlegend=False,
            xaxis_title="Jumlah proses", yaxis_title="", margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Distribusi Tipe RACI**")
        raci_counts = (
            structured_df["raci_type"].value_counts()
            .reindex(["R", "A", "C", "I"]).fillna(0)
        )
        raci_counts.index = raci_counts.index.map(config.RACI_LABELS)
        fig = px.pie(values=raci_counts.values, names=raci_counts.index, hole=.58,
                     color=raci_counts.index, color_discrete_map=RACI_COLORS)
        fig.update_traces(textinfo="percent", textfont_size=13,
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(
            height=300, legend=dict(orientation="h", y=-0.08),
            margin=dict(t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Sebaran Jumlah Peran per Proses**")
        fig = px.histogram(proc_prof, x="total_peran", nbins=25,
                           color_discrete_sequence=[SKY])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            height=220, xaxis_title="Jumlah peran dalam satu proses", yaxis_title="Frekuensi",
            margin=dict(t=10, b=10), bargap=.05,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Peta Kompleksitas Proses**")
    st.caption(
        "Sumbu X = jumlah pelaksana & pemilik (R+A), sumbu Y = jumlah pihak yang dikonsultasikan "
        "atau diinformasikan (C+I). Proses di kanan-atas melibatkan banyak pihak dari kedua sisi."
    )
    plot_df = proc_prof.copy()
    plot_df["inti"] = plot_df["n_R"] + plot_df["n_A"]
    fig = px.scatter(
        plot_df, x="inti", y="beban_koordinasi", color="kategori",
        hover_name="nama_proses", size="total_peran", size_max=26,
        custom_data=["hierarchy_id", "n_R", "n_A", "n_C", "n_I"],
        color_discrete_sequence=px.colors.qualitative.Safe, opacity=.75,
    )
    fig.update_traces(
        marker=dict(line=dict(width=.5, color="white")),
        hovertemplate=(
            "<b>%{hovertext}</b><br>ID: %{customdata[0]}"
            "<br>R: %{customdata[1]} · A: %{customdata[2]} · "
            "C: %{customdata[3]} · I: %{customdata[4]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=520, xaxis_title="Pelaksana + Pemilik (R+A)",
        yaxis_title="Dikonsultasikan + Diinformasikan (C+I)",
        legend=dict(title="", orientation="h", y=-0.18, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 20 proses dengan keterlibatan peran terbanyak"):
        top_complex = (
            proc_prof.nlargest(20, "total_peran")
            [["hierarchy_id", "nama_proses", "kategori", "n_R", "n_A", "n_C", "n_I", "total_peran"]]
            .rename(columns={
                "hierarchy_id": "ID", "nama_proses": "Proses Bisnis", "kategori": "Kategori",
                "n_R": "R", "n_A": "A", "n_C": "C", "n_I": "I", "total_peran": "Total Peran",
            })
            .reset_index(drop=True)
        )
        st.dataframe(top_complex, use_container_width=True)

# ============================================================
# TAB 4 — ROLE × CATEGORY HEATMAP
# ============================================================
with tab_matrix:
    st.markdown('<p class="sec-title">Matriks Peran × Kategori</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-note">Menunjukkan di area mana setiap peran paling banyak terlibat. '
        'Baris yang pekat di banyak kolom menandakan peran lintas-fungsi; '
        'baris yang pekat di satu kolom menandakan peran yang sangat terspesialisasi.</p>',
        unsafe_allow_html=True,
    )

    hc1, hc2 = st.columns([1, 1])
    with hc1:
        heat_type = st.selectbox(
            "Tipe RACI yang dipetakan", ["Semua", "R", "A", "C", "I"],
            format_func=lambda x: "Semua tipe" if x == "Semua" else config.RACI_LABELS[x],
        )
    with hc2:
        heat_n = st.slider("Jumlah role teratas", 10, 40, 20, key="slider_heat")

    sdf = structured_df.copy()
    sdf["root"] = sdf["hierarchy_id"].astype(str).str.split(".").str[0]
    sdf["kategori"] = sdf["root"].map(lambda r: level1_lookup.get(r, f"Kategori {r}"))
    if heat_type != "Semua":
        sdf = sdf[sdf["raci_type"] == heat_type]

    if sdf.empty:
        st.info("Tidak ada data untuk filter ini.")
    else:
        top_roles_heat = sdf["role_normalized"].value_counts().head(heat_n).index.tolist()
        pivot = (
            sdf[sdf["role_normalized"].isin(top_roles_heat)]
            .pivot_table(index="role_normalized", columns="kategori",
                         values="process_id", aggfunc="nunique")
            .fillna(0)
            .loc[top_roles_heat]
        )
        fig = px.imshow(
            pivot, aspect="auto", text_auto=True,
            color_continuous_scale=["#F8FAFC", SKY, BLUE, NAVY],
        )
        fig.update_traces(textfont_size=10)
        fig.update_layout(
            height=max(420, heat_n * 26), xaxis_title="", yaxis_title="",
            xaxis_tickangle=-35, coloraxis_colorbar=dict(title="Proses"),
            margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Kolaborator Terdekat**")
    st.caption("Peran lain yang paling sering muncul bersama dalam proses yang sama.")
    focus_role = st.selectbox(
        "Pilih role", options=sorted(role_prof["role"].tolist()),
        index=None, placeholder="Ketik untuk mencari role...",
    )
    if focus_role:
        procs_with_role = set(
            structured_df.loc[structured_df["role_normalized"] == focus_role, "process_id"]
        )
        co = (
            structured_df[
                structured_df["process_id"].isin(procs_with_role)
                & (structured_df["role_normalized"] != focus_role)
            ]
            .groupby("role_normalized")["process_id"].nunique()
            .nlargest(12).sort_values()
        )
        if co.empty:
            st.info("Tidak ditemukan peran lain yang berbagi proses dengan role ini.")
        else:
            fig = px.bar(co, orientation="h", text_auto=True, color_discrete_sequence=[TEAL])
            fig.update_traces(marker_line_width=0, textposition="outside")
            fig.update_layout(
                height=max(320, len(co) * 30), showlegend=False,
                xaxis_title=f"Jumlah proses bersama dengan {focus_role}", yaxis_title="",
                margin=dict(t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="insight-box"><b>{focus_role}</b> terlibat di '
                f'<b>{len(procs_with_role)} proses</b>, paling sering berdampingan dengan '
                f'<b>{co.idxmax()}</b> ({int(co.max())} proses bersama).</div>',
                unsafe_allow_html=True,
            )

# ============================================================
# TAB 5 — TUPOKSI COVERAGE
# ============================================================
with tab_tupoksi:
    st.markdown('<p class="sec-title">Cakupan Tupoksi per Role RACI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-note">Role yang muncul di matriks RACI tetapi tidak memiliki data '
        'Tugas Pokok yang cocok (misalnya level Direktur/GM, yang memang belum tersedia '
        'di katalog Tupoksi ini).</p>',
        unsafe_allow_html=True,
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
    covered = int(coverage_df["tupoksi_ditemukan"].sum())
    gap_n = int((~coverage_df["tupoksi_ditemukan"]).sum())

    # Gap yang paling berdampak: role tanpa tupoksi TAPI banyak memegang akuntabilitas
    gap_impact = (
        gap_df.merge(role_prof, on="role", how="left")
        .fillna({"n_A": 0, "total": 0})
        .nlargest(10, "n_A")
    )

    m1, m2, m3 = st.columns([1, 1, 1.4])
    m1.metric("✅ Role dengan data Tupoksi", covered)
    m2.metric("⚠️ Role tanpa data Tupoksi", gap_n)
    with m3:
        fig = px.pie(
            values=[covered, gap_n], names=["Ada Tupoksi", "Tanpa Tupoksi"], hole=.62,
            color=["Ada Tupoksi", "Tanpa Tupoksi"],
            color_discrete_map={"Ada Tupoksi": TEAL, "Tanpa Tupoksi": CORAL},
        )
        fig.update_traces(textinfo="percent", textfont_size=13,
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(height=210, legend=dict(orientation="h", y=-0.12),
                          margin=dict(t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    if len(gap_impact) and gap_impact["n_A"].max() > 0:
        top_gap = gap_impact.iloc[0]
        st.markdown(
            f'<div class="insight-box insight-warn">💡 <b>Gap paling berdampak:</b> '
            f'<b>{top_gap["role"]}</b> memegang akuntabilitas di '
            f'<b>{int(top_gap["n_A"])} proses</b> namun belum punya data Tugas Pokok. '
            f'Melengkapi tupoksi untuk peran seperti ini memberi dampak paling besar '
            f'terhadap kualitas analisis kesesuaian.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Prioritas Pelengkapan Tupoksi**")
        st.caption("Role tanpa data tupoksi, diurutkan dari yang paling banyak memegang akuntabilitas.")
        show_gap = gap_impact[["role", "n_A", "n_R", "total"]].rename(columns={
            "role": "Role", "n_A": "Sebagai Accountable", "n_R": "Sebagai Responsible",
            "total": "Total Keterlibatan",
        }).astype({"Sebagai Accountable": int, "Sebagai Responsible": int, "Total Keterlibatan": int})
        st.dataframe(show_gap.reset_index(drop=True), use_container_width=True)

    with st.expander(f"📄 Lihat seluruh {len(gap_df)} role tanpa data Tupoksi"):
        st.dataframe(gap_df[["role"]].reset_index(drop=True), use_container_width=True)
        st.download_button(
            "⬇️ Unduh daftar gap (CSV)",
            gap_df.to_csv(index=False).encode("utf-8"),
            file_name="tupoksi_coverage_gap.csv", mime="text/csv",
        )

    with st.expander("📄 Lihat role yang berhasil dicocokkan + skor kemiripan"):
        matched = (
            coverage_df[coverage_df["tupoksi_ditemukan"]]
            .sort_values("skor_kecocokan")
            [["role", "jabatan_cocok", "skor_kecocokan"]]
            .rename(columns={
                "role": "Role (RACI)", "jabatan_cocok": "Jabatan (Tupoksi)",
                "skor_kecocokan": "Skor Kecocokan",
            })
        )
        st.caption(
            f"Diurutkan dari skor terendah — baris teratas paling mungkin salah cocok "
            f"(ambang batas saat ini: {config.JABATAN_MATCH_THRESHOLD})."
        )
        st.dataframe(matched.reset_index(drop=True), use_container_width=True, height=320)

# ============================================================
# TAB 6 — ON-DEMAND ALIGNMENT CHECKER
# ============================================================
with tab_check:
    st.markdown('<p class="sec-title">Cek Kesesuaian Tugas Pokok (per Proses)</p>',
                unsafe_allow_html=True)
    st.caption(
        "Memanggil API embedding hanya saat tombol ditekan — tidak dihitung otomatis "
        "untuk seluruh data."
    )

    process_pick = st.selectbox(
        "Pilih proses bisnis", options=known_processes, index=None,
        placeholder="Ketik untuk mencari proses...",
    )

    if process_pick and st.button("🚀 Cek kesesuaian tugas pokok", type="primary"):
        with st.spinner("Menghitung kesesuaian..."):
            embed_client = get_embed_client()
            row = query_by_process(df, process_pick).iloc[0]
            context, role_candidates, proc_vec = build_process_context(row, embeddings, embedded_ids)

        st.markdown(f"**Proses:** {row['PROSES BISNIS']}  ·  `{row['HIERARCHY ID']}`")

        # Konteks cepat dari profil proses
        match_prof = proc_prof[proc_prof["hierarchy_id"] == str(row["HIERARCHY ID"])]
        if len(match_prof):
            p = match_prof.iloc[0]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Responsible", int(p["n_R"]))
            q2.metric("Accountable", int(p["n_A"]))
            q3.metric("Consulted", int(p["n_C"]))
            q4.metric("Informed", int(p["n_I"]))
            if p["n_A"] != 1:
                st.warning(
                    f"Proses ini memiliki {int(p['n_A'])} Accountable — "
                    "idealnya tepat satu pemilik keputusan."
                )

        if proc_vec is None:
            st.warning("Embedding untuk proses ini tidak ditemukan di cache — coba proses lain.")
        else:
            seen = set()
            for role_name, code in role_candidates:
                if role_name in seen:
                    continue
                seen.add(role_name)
                with st.container(border=True):
                    st.markdown(f"**[{config.RACI_LABELS[code]}] {role_name}**")
                    block = format_alignment_block(
                        role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi
                    )
                    st.text(block)
