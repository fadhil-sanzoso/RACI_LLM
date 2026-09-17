import os

import streamlit as st


def _get_secret(key, required=True):
    """Reads from Streamlit secrets first (works both locally via .streamlit/secrets.toml
    and on Streamlit Community Cloud via the Settings -> Secrets panel), falling back to
    a plain environment variable for other hosting setups."""
    val = None
    try:
        val = st.secrets[key]
    except Exception:
        val = os.environ.get(key)
    if required and not val:
        st.error(
            f"Missing required secret: `{key}`.\n\n"
            f"Add it in your Streamlit Community Cloud app's **Settings -> Secrets**, "
            f"or in a local `.streamlit/secrets.toml` file "
            f"(see `.streamlit/secrets.toml.example` for the format)."
        )
        st.stop()
    return val


CHAT_API_KEY = _get_secret("AI_LAB_API_KEY")
EMBED_API_KEY = _get_secret("AILAB_API_KEY")

BASE_URL_LLM = _get_secret("BASE_URL_LLM")
CHAT_MODEL = _get_secret("CHAT_MODEL")

BASE_URL_EMBED = _get_secret("BASE_URL_EMBED")
EMBED_MODEL = _get_secret("EMBED_MODEL")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

LIST_THRESHOLD = 20
JABATAN_MATCH_THRESHOLD = 75
ENTITY_MATCH_THRESHOLD = 70

RACI_LABELS = {"R": "Responsible", "A": "Accountable", "C": "Consulted", "I": "Informed"}

ALIAS_MAP = {
    r"\bTI\b": "Teknologi Informasi",
    r"\bSDM\b": "Sumber Daya Manusia",
    r"\bK3\b": "Keselamatan dan Kesehatan Kerja",
}

TUPOKSI_KEYWORDS = [r"\btupoksi\b", r"\btugas pokok\b", r"\buraian jabatan\b", r"\bjob\s*desc"]

# --- Conversation memory ---
MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS_PER_MESSAGE = 500
