# RACI Chatbot — PLN (Streamlit)

A Streamlit app for the PLN RACI matrix chatbot: a chat interface (`app.py`) plus an
analytics dashboard (`pages/1_📊_Analytics_Dashboard.py`), built on top of the same logic
developed and validated in the project's Colab notebook.

## Project structure

```
raci_chatbot_streamlit/
├── app.py                              # Chat page (main entry point)
├── pages/
│   └── 1_📊_Analytics_Dashboard.py      # Analytics dashboard page
├── src/
│   ├── config.py                       # secrets, constants, thresholds, alias map
│   ├── data_loader.py                  # cached loaders for all bundled data files
│   ├── retrieval.py                    # semantic search, entity resolution, role/process queries
│   ├── aggregate.py                    # count/ranking/category aggregation
│   ├── alignment.py                    # tugas pokok lookup + alignment scoring
│   ├── intent.py                       # LLM intent classification prompt + call
│   └── ask_engine.py                   # the ask() orchestrator tying everything together
├── scripts/
│   └── export_data_for_streamlit.py    # run in Colab to produce data/ files
├── data/                                # bundled data files (see data/README.md)
├── .streamlit/
│   ├── config.toml                     # theme
│   └── secrets.toml.example            # template — copy into Cloud Secrets panel
├── requirements.txt
└── .gitignore
```

## 1. Get your data files

In your existing Colab notebook, after cells 1–12 have already run (so `df`, `structured_df`,
`embeddings`, `embedded_ids`, and `jabatan_tugas` all exist in memory), paste
`scripts/export_data_for_streamlit.py` as a new cell and run it. It writes 5 files into
`{SAVE_DIR}/streamlit_export/` on your Drive. Download them and place them into this
project's `data/` folder using the exact filenames listed in `data/README.md`.

## 2. Run locally (optional, to test before deploying)

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real AIR_API_KEY_CHAT / AIR_API_KEY_EMBED
streamlit run app.py
```

## 3. Deploy to Streamlit Community Cloud

1. Push this project (including the populated `data/` folder) to a GitHub repo.
   - `.gitignore` already excludes real secrets and the raw `.xlsx` sources — the bundled
     `.parquet`/`.npy`/`.pkl` files in `data/` **should** be committed, since "repo-bundled
     data" is the chosen deployment approach.
   - These files are small enough for a normal git repo (a few MB each); no Git LFS needed.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and set the
   main file path to `app.py`.
3. In the app's **Settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` with your real API key values filled in.
4. Deploy. The Analytics Dashboard page will show up automatically in the sidebar nav
   (Streamlit's multipage convention: any file under `pages/` becomes its own page).

## Known limitations / design tradeoffs

- **Tugas Pokok duty embeddings are computed lazily and cached in memory**, not bundled as a
  precomputed file. With ~79k individual duty rows across ~3,662 jabatan, a fully precomputed
  embedding cache would likely be too large to comfortably commit to a plain git repo. In
  practice this means: the first time anyone asks about a given jabatan after the app instance
  starts (or restarts/redeploys), that one lookup costs one extra embedding API call and a
  slightly longer wait; every question after that, for that same jabatan, is instant for as
  long as the instance stays up. If this becomes a real pain point, the fix is to precompute
  and bundle embeddings for your most-frequently-asked roles specifically, rather than all
  ~3,662 jabatan.
- **Direktur/GM-level roles have no Tugas Pokok data** — this is a known, expected gap in the
  source Tupoksi catalog (their duties live in a separate board-level governance document
  that hasn't been provided yet), not a bug. The Analytics Dashboard's "Cakupan Tupoksi"
  section surfaces exactly which roles fall into this gap.
- **Process embeddings are reused, not recomputed** — `build_process_context()` looks up a
  resolved process's already-computed embedding from `embeddings.npy` instead of calling the
  embed API again, since the whole point of bundling that file is to avoid paying for the
  same embedding twice. Free-text semantic-search fallbacks (when a process/topic can't be
  cleanly resolved) still need a fresh embedding call for the query text itself — that part
  is unavoidable.
- **The `topic_to_roles` intent** (e.g. "who's involved in EPC procurement," spanning multiple
  processes at once) is newer and hasn't been as thoroughly tested end-to-end as the other
  intents — keep an eye on it and adjust `resolve_category()`'s `sim_threshold` in
  `src/aggregate.py` if it pulls in irrelevant processes or misses relevant ones for your
  actual usage patterns.
