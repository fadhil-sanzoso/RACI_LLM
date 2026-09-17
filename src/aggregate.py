from .retrieval import resolve_entity, semantic_search


def summarize_by_category(matches_df, level1_lookup):
    tmp = matches_df.copy()
    tmp["root"] = tmp["hierarchy_id"].astype(str).str.split(".").str[0]
    summary = tmp.groupby("root").size().sort_values(ascending=False)
    lines = []
    for root, count in summary.items():
        label = level1_lookup.get(root, f"Kategori {root}")
        lines.append(f"- {label}: {count} proses")
    return "\n".join(lines)


def aggregate_count_by_role(structured_df, known_roles, role_name, raci_filter=None):
    resolved, _ = resolve_entity(role_name, known_roles)
    role_to_use = resolved if resolved else role_name
    mask = structured_df["role_normalized"].str.contains(role_to_use, case=False, na=False)
    if raci_filter:
        mask &= structured_df["raci_type"] == raci_filter
    matches = structured_df[mask]
    breakdown = matches["raci_type"].value_counts().to_dict()
    return role_to_use, len(matches), breakdown


def aggregate_top_roles_by_raci(structured_df, raci_type, top_n=10):
    subset = structured_df[structured_df["raci_type"] == raci_type]
    return subset["role_normalized"].value_counts().head(top_n)


def resolve_category(category_text, df, level1_lookup, embed_client, embeddings, embedded_ids,
                      top_k=5, sim_threshold=0.3):
    """Try matching a formal Lv.1 category label first; fall back to semantic search over
    process documents (thresholded by similarity, not a hard top_k cutoff) for free-text
    topics like 'pengadaan EPC' that don't correspond to one formal category label."""
    labels = list(level1_lookup.values())
    resolved_label, _ = resolve_entity(category_text, labels, threshold=70)
    if resolved_label:
        root = [k for k, v in level1_lookup.items() if v == resolved_label][0]
        matched_ids = df[df["HIERARCHY ID"].astype(str).str.split(".").str[0] == root].index.tolist()
        return resolved_label, matched_ids, "level1_label_match"

    results = semantic_search(category_text, embed_client, embeddings, embedded_ids, df, top_k=top_k)
    matched_ids = [r["process_id"] for r in results if r["similarity"] >= sim_threshold]
    return category_text, matched_ids, "semantic_fallback"


def aggregate_count_roles_in_category(structured_df, category_text, df, level1_lookup,
                                       embed_client, embeddings, embedded_ids):
    label, matched_ids, _ = resolve_category(category_text, df, level1_lookup, embed_client, embeddings, embedded_ids)
    if not matched_ids:
        return label, [], 0
    subset = structured_df[structured_df["process_id"].isin(matched_ids)]
    unique_roles = sorted(subset["role_normalized"].unique().tolist())
    return label, unique_roles, len(matched_ids)


def build_topic_context(structured_df, topic_text, df, level1_lookup, embed_client, embeddings,
                         embedded_ids, raci_labels, raci_filter=None):
    label, matched_ids, method = resolve_category(
        topic_text, df, level1_lookup, embed_client, embeddings, embedded_ids
    )
    if not matched_ids:
        return f"Tidak ditemukan proses bisnis yang relevan dengan topik '{topic_text}'.", 0, method

    subset = structured_df[structured_df["process_id"].isin(matched_ids)]
    if raci_filter:
        subset = subset[subset["raci_type"] == raci_filter]

    if subset.empty:
        return f"Topik: {label}\nTidak ditemukan role untuk filter yang diminta.", len(matched_ids), method

    lines = [f"Topik/Kategori: {label}", f"Total proses relevan: {len(matched_ids)}", ""]
    for code in ["R", "A", "C", "I"]:
        if raci_filter and code != raci_filter:
            continue
        type_subset = subset[subset["raci_type"] == code]
        if type_subset.empty:
            continue
        lines.append(f"{raci_labels[code]} ({code}):")
        grouped = type_subset.groupby("role_normalized")["process_name"].apply(list)
        for role, procs in grouped.items():
            proc_list = procs[:5]
            suffix = f" ... (+{len(procs) - 5} proses lain)" if len(procs) > 5 else ""
            lines.append(f"  - {role}: {', '.join(proc_list)}{suffix}")
        lines.append("")

    return "\n".join(lines), len(matched_ids), method
