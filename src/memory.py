"""Conversation memory helpers -- shared by intent classification and answer generation
so both stages see the same recent history, formatted consistently."""

from . import config


def format_history(history, max_turns=None, max_chars=None):
    if not history:
        return ""

    max_turns = max_turns if max_turns is not None else config.MAX_HISTORY_TURNS
    max_chars = max_chars if max_chars is not None else config.MAX_HISTORY_CHARS_PER_MESSAGE

    recent = history[-(max_turns * 2):]

    lines = []
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Asisten"
        content = (msg.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + " ...(dipotong)"
        lines.append(f"{role_label}: {content}")

    return "\n".join(lines)