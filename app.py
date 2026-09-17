import streamlit as st

from src.ask_engine import ask
from src.chart import render_chart
from src.data_loader import (
    get_chat_client,
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

st.set_page_config(page_title="RACI Chatbot - PLN", page_icon="🤖", layout="wide")

st.title("🤖 RACI Chatbot — PLN")
st.caption(
    "Tanya siapa yang bertanggung jawab untuk suatu proses, proses apa saja milik suatu role, "
    "tugas pokok suatu jabatan, hingga kesesuaian tugas pokok terhadap RACI."
)

# --- Load everything once; all cached by Streamlit ---
try:
    chat_client = get_chat_client()
    embed_client = get_embed_client()
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

with st.sidebar:
    st.subheader("Status Data")
    st.metric("Proses bisnis", len(df))
    st.metric("Role unik (RACI)", len(known_roles))
    st.metric("Jabatan dengan Tupoksi", len(known_jabatan_tupoksi))
    st.divider()
    debug_mode = st.checkbox("Tampilkan debug info", value=False)
    if st.button("Reset percakapan"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if debug_mode and msg.get("debug_info"):
            with st.expander("Debug info"):
                st.json(msg["debug_info"])

question = st.chat_input("Tanyakan sesuatu tentang RACI atau Tugas Pokok...")

if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": question, "debug_info": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Memproses..."):
            answer, debug_info, chart_data  = ask(
                question,
                chat_client=chat_client,
                embed_client=embed_client,
                df=df,
                structured_df=structured_df,
                embeddings=embeddings,
                embedded_ids=embedded_ids,
                known_roles=known_roles,
                known_processes=known_processes,
                level1_lookup=level1_lookup,
                jabatan_tugas=jabatan_tugas,
                known_jabatan_tupoksi=known_jabatan_tupoksi,
                history=history,
                debug=debug_mode,
            )
        st.markdown(answer)
        if chart_data:
            render_chart(chart_data)
        if debug_mode:
            with st.expander("Debug info"):
                st.json(debug_info)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "debug_info": debug_info if debug_mode else None}
    )
