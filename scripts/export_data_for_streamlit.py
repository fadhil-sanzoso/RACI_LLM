# ============================================================================
# Paste this as a NEW cell at the end of your existing Colab notebook,
# AFTER cells 1-12 have already run (so df, structured_df, embeddings,
# embedded_ids, jabatan_tugas, and level1_lookup already exist in memory).
#
# It writes everything the Streamlit app needs into a local export folder.
# Download that folder's contents and place them into the Streamlit repo's
# `data/` directory before deploying.
# ============================================================================

import os
import pickle
import shutil

EXPORT_DIR = f"{SAVE_DIR}/streamlit_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 1. Raw RACI process table (needs Semantic_Document, LEVEL, HIERARCHY ID, PROSES BISNIS,
#    and all four RACI columns intact) -> parquet keeps column names/spacing exactly as-is.
df.to_parquet(f"{EXPORT_DIR}/raci_processes.parquet", index=True)

# 2. Exploded structured RACI table (one row per role per process)
structured_df.to_parquet(f"{EXPORT_DIR}/structured_raci.parquet", index=False)

# 3. Process embeddings (bge-m3) + matching process_id index
#    (re-saved from the existing cache files, just copied into the export folder)
shutil.copy(EMBED_CACHE_PATH, f"{EXPORT_DIR}/embeddings.npy")
shutil.copy(EMBED_IDS_PATH, f"{EXPORT_DIR}/embedding_process_ids.npy")

# 4. Tugas Pokok (job duties) lookup dict: {sebutan_normalized: [duty strings]}
with open(f"{EXPORT_DIR}/jabatan_tugas.pkl", "wb") as f:
    pickle.dump(jabatan_tugas, f)

print(f"Export complete. Files written to: {EXPORT_DIR}")
print("\nFiles:")
for fname in os.listdir(EXPORT_DIR):
    fpath = os.path.join(EXPORT_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  - {fname}  ({size_mb:.2f} MB)")

print(
    "\nNext steps:\n"
    "1. Download these files from Google Drive (or use the Colab file browser's download option).\n"
    "2. Place them into your Streamlit repo's `data/` folder, using these exact filenames.\n"
    "3. Commit and push to GitHub, then deploy on Streamlit Community Cloud.\n"
    "   (Note: this export intentionally does NOT include tugas_pokok_embeddings_cache.pkl --\n"
    "   the deployed app computes and caches those lazily at runtime. See README.md.)"
)
