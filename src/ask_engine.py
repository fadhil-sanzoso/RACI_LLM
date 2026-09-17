import pandas as pd
from rapidfuzz import fuzz
from .memory import format_history
from . import config
from .chart import choose_chart

from .aggregate import (
    aggregate_count_by_role,
    aggregate_count_roles_in_category,
    aggregate_top_roles_by_raci,
    build_topic_context,
    summarize_by_category,
)
from .alignment import format_alignment_block, get_tupoksi_for_role, is_tupoksi_query
from .intent import extract_intent
from .retrieval import (
    expand_aliases,
    get_process_vector,
    query_by_process,
    query_by_role,
    resolve_entity,
    semantic_search,
    split_roles,
)


def build_process_context(row, embeddings, embedded_ids, raci_filter=None):
    fields = {
        "R": ("Responsible", row["RESPONSIBLE (R)"]),
        "A": ("Accountable", row["ACCOUNTABLE ( A )"]),
        "C": ("Consulted", row["CONSULTED (C)"]),
        "I": ("Informed", row["INFORMED (I)"]),
    }
    if raci_filter and raci_filter in fields:
        fields = {raci_filter: fields[raci_filter]}

    # Reuse the already-computed embedding instead of a fresh API call --
    # saves cost/latency on every deployed-app request.
    proc_vec = get_process_vector(row.name, embeddings, embedded_ids)

    context_lines = [f"Proses Bisnis: {row['PROSES BISNIS']} (ID: {row['HIERARCHY ID']})"]
    role_candidates = []  # (role_name, raci_code) -- filtered against the LLM's actual answer, later

    for code, (label, value) in fields.items():
        if pd.notna(value):
            roles_list = split_roles(value)
            context_lines.append(f"{label} ({code}): {', '.join(roles_list)}")
            role_candidates.extend((r, code) for r in roles_list)

    return "\n".join(context_lines), role_candidates, proc_vec


def extract_relevant_roles(answer_text, role_candidates, threshold=85):
    """Only build alignment blocks for roles the LLM's answer actually named -- avoids
    dumping alignment for every R/A/C/I role when raci_filter wasn't specified."""
    seen, relevant = set(), []
    for role_name, code in role_candidates:
        if role_name in seen:
            continue
        if fuzz.partial_ratio(role_name.lower(), answer_text.lower()) >= threshold:
            relevant.append((role_name, code))
            seen.add(role_name)
    return relevant


def ask(
    question,
    *,
    chat_client,
    embed_client,
    df,
    structured_df,
    embeddings,
    embedded_ids,
    known_roles,
    known_processes,
    level1_lookup,
    jabatan_tugas,
    known_jabatan_tupoksi,
    history=None,     # <-- add this
    debug=False,
):
    debug_info = {}
    chart_data = None
    #intent_data = extract_intent(question, chat_client)
    intent_data = extract_intent(question, chat_client, history=history)
    intent = intent_data.get("intent")
    entity = intent_data.get("entity")
    raci_filter = intent_data.get("raci_filter")
    agg_type = intent_data.get("agg_type")
    debug_info["intent_data"] = intent_data

    context = ""
    role_candidates = []
    proc_vec = None
    alignment_blocks = []

    # --- 1. Tupoksi direct lookup (keyword override takes precedence over LLM intent) ---
    if (intent == "role_to_tupoksi" or is_tupoksi_query(question)) and entity:
        entity_expanded = expand_aliases(entity)
        matched_jabatan, duties = get_tupoksi_for_role(entity_expanded, known_jabatan_tupoksi, jabatan_tugas)
        if not matched_jabatan:
            context = (
                f"Tidak ditemukan data tugas pokok untuk '{entity}' "
                f"(kemungkinan level Direktur/GM — belum tersedia di katalog Tupoksi ini — "
                f"atau nama jabatan tidak dikenali)."
            )
        else:
            lines = [f"- {d}" for d in duties]
            context = f"Tugas Pokok untuk jabatan '{matched_jabatan}' (total {len(duties)} tugas):\n" + "\n".join(lines)

    # --- 2. Process -> Role ---
    elif intent == "process_to_role" and entity:
        entity_expanded = expand_aliases(entity)
        resolved, _ = resolve_entity(entity_expanded, known_processes)
        if resolved:
            row = query_by_process(df, resolved).iloc[0]
            context, role_candidates, proc_vec = build_process_context(row, embeddings, embedded_ids, raci_filter)
        else:
            context = "\n\n".join(
                r["semantic_document"]
                for r in semantic_search(entity_expanded, embed_client, embeddings, embedded_ids, df, top_k=3)
            )

    # --- 3. Role -> Process ---
    elif intent == "role_to_process" and entity:
        entity_expanded = expand_aliases(entity)
        resolved, _ = resolve_entity(entity_expanded, known_roles)
        role_to_use = resolved if resolved else entity_expanded
        all_matches, total_count = query_by_role(structured_df, role_to_use, raci_type=raci_filter, limit=100000)
        if total_count == 0:
            context = f"Tidak ditemukan data untuk role '{entity}'."
        elif total_count <= config.LIST_THRESHOLD:
            lines = [
                f"- [{config.RACI_LABELS[m['raci_type']]}] {m['process_name']} (ID: {m['hierarchy_id']})"
                for _, m in all_matches.iterrows()
            ]
            context = f"Role: {role_to_use}\nTotal {total_count} proses (daftar lengkap):\n" + "\n".join(lines)
        else:
            context = (
                f"Role: {role_to_use}\nTotal {total_count} proses — terlalu banyak untuk ditampilkan satu per satu.\n"
                f"Ringkasan per kategori:\n{summarize_by_category(all_matches, level1_lookup)}\n"
                f"INSTRUKSI: Sampaikan total {total_count}, tampilkan SEMUA baris ringkasan kategori di atas apa adanya "
                f"(jangan pilih sebagian), lalu tanyakan kategori mana yang ingin dilihat detailnya."
            )

    # --- 4. Aggregate / count / ranking ---
    elif intent == "aggregate":
        if agg_type == "count_by_role" and entity:
            entity_expanded = expand_aliases(entity)
            role_to_use, total, breakdown = aggregate_count_by_role(structured_df, known_roles, entity_expanded, raci_filter)
            breakdown_lines = "\n".join(f"  - {config.RACI_LABELS[k]}: {v}" for k, v in breakdown.items())
            context = (
                f"Role: {role_to_use}\n"
                f"Total proses terkait: {total}"
                + (f" (filter: {config.RACI_LABELS.get(raci_filter, raci_filter)})" if raci_filter else "")
                + f"\nRincian per tipe RACI:\n{breakdown_lines}"
            )
            chart_data = choose_chart(
                breakdown,
                chart_requested=intent_data.get("chart_requested", False),
                title=f"Distribusi RACI - {role_to_use}",
            )
        

        elif agg_type == "top_by_raci":
            raci_for_rank = raci_filter or "R"
            counts = aggregate_top_roles_by_raci(structured_df, raci_for_rank, top_n=10)
            lines = [f"- {role}: {count} proses" for role, count in counts.items()]
            context = (
                f"Ranking role dengan jumlah {config.RACI_LABELS[raci_for_rank]} terbanyak (top 10):\n"
                + "\n".join(lines)
                + "\nINSTRUKSI: tampilkan SEMUA baris ranking di atas apa adanya, jangan memilih sebagian."
            )

        elif agg_type == "count_roles_in_category" and entity:
            entity_expanded = expand_aliases(entity)
            label, unique_roles, matched_count = aggregate_count_roles_in_category(
                structured_df, entity_expanded, df, level1_lookup, embed_client, embeddings, embedded_ids
            )
            shown = unique_roles[:30]
            context = (
                f"Kategori/topik: {label}\n"
                f"Jumlah proses terkait: {matched_count}\n"
                f"Jumlah role unik terlibat: {len(unique_roles)}\n"
                f"Daftar role: {', '.join(shown)}"
                + (f" ... (menampilkan 30 dari {len(unique_roles)} role)" if len(unique_roles) > 30 else "")
            )

        elif agg_type == "count_tupoksi_by_role" and entity:
            entity_expanded = expand_aliases(entity)
            matched_jabatan, duties = get_tupoksi_for_role(entity_expanded, known_jabatan_tupoksi, jabatan_tugas)
            if not matched_jabatan:
                context = f"Tidak ditemukan data tugas pokok untuk '{entity}'."
            else:
                context = f"Jabatan: {matched_jabatan}\nTotal tugas pokok: {len(duties)}"

        else:
            context = "Maaf, jenis pertanyaan agregat ini belum bisa diproses — coba ulangi dengan lebih spesifik."

    # --- 5. Topic -> roles (multi-process category) ---
    elif intent == "topic_to_roles" and entity:
        entity_expanded = expand_aliases(entity)
        context, matched_process_count, method = build_topic_context(
            structured_df, entity_expanded, df, level1_lookup, embed_client, embeddings, embedded_ids,
            config.RACI_LABELS, raci_filter,
        )
        debug_info["topic_resolution_method"] = method
        if matched_process_count > 20:
            context += (
                "\n\nINSTRUKSI: Topik ini mencakup banyak proses — tampilkan ringkasan role per tipe RACI "
                "di atas apa adanya (jangan pilih sebagian), sebutkan totalnya, dan tawarkan untuk melihat "
                "detail per proses bila diminta."
            )

    # --- 6. Fallback: generic semantic search ---
    else:
        context = "\n\n".join(
            r["semantic_document"]
            for r in semantic_search(question, embed_client, embeddings, embedded_ids, df, top_k=5)
        )

    debug_info["context"] = context

    history_text = format_history(history)
    debug_info["history_received"] = history_text or "(kosong)"
    history_block = (
        f"\nRiwayat percakapan sebelumnya (HANYA untuk menjaga alur percakapan -- "
        f"JANGAN gunakan sebagai sumber fakta baru, semua fakta harus dari 'Konteks' di atas):\n{history_text}\n"
        if history_text else ""
)

    answer_prompt = f"""Jawab pertanyaan user HANYA berdasarkan konteks berikut. Jika tidak ada di konteks, katakan tidak ditemukan. Jangan mengarang.
JIKA konteks berisi daftar atau ringkasan kategori, tampilkan SEMUA baris yang diberikan — jangan memilih sebagian atau meringkas lebih lanjut.
{history_block}
Konteks:
{context}

Pertanyaan: {question}
Jawaban:"""

    kwargs = dict(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    try:
        resp = chat_client.chat.completions.create(
            **kwargs, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    except Exception:
        resp = chat_client.chat.completions.create(**kwargs)

    msg = resp.choices[0].message
    primary_answer = (msg.content or getattr(msg, "reasoning_content", None) or "[Model returned empty response]").strip()

    # Alignment: only for roles the LLM's own answer actually named -- not every R/A/C/I role in context
    if role_candidates and proc_vec is not None:
        for role_name, code in extract_relevant_roles(primary_answer, role_candidates):
            alignment_blocks.append(
                format_alignment_block(role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi)
            )

    if alignment_blocks:
        primary_answer += "\n\nKesesuaian Tugas Pokok:\n" + "\n".join(alignment_blocks)

    return primary_answer, debug_info, chart_data
