import json
from .memory import format_history
from . import config

INTENT_PROMPT = """Kamu adalah sistem klasifikasi query untuk chatbot RACI matrix PLN.
Tentukan dari pertanyaan user:

1. intent: salah satu dari "process_to_role", "role_to_process", "role_to_tupoksi", "aggregate", "topic_to_roles", atau "unclear"
   - "process_to_role": user bertanya SIAPA yang R/A/C/I untuk SATU proses bisnis tertentu
   - "role_to_process": user bertanya PROSES BISNIS apa saja (daftar) yang menjadi tanggung jawab satu role dalam RACI
   - "role_to_tupoksi": user bertanya TUGAS POKOK / uraian jabatan / job desc suatu role (BUKAN proses RACI)
   - "aggregate": pertanyaan JUMLAH, HITUNGAN, TOTAL, atau RANKING — biasanya mengandung kata
     "berapa", "berapa banyak", "berapa jumlah", "total", "paling banyak", "paling sedikit"
   - "topic_to_roles": user bertanya SIAPA SAJA yang terlibat/berperan dalam suatu TOPIK/KATEGORI
     yang mencakup BEBERAPA proses bisnis sekaligus (bukan satu proses spesifik)
   - "unclear": jika tidak jelas termasuk kategori mana

2. entity: nama proses bisnis, nama role/jabatan, ATAU nama kategori/topik (ambil sesuai teks asli user)

3. raci_filter: "R"/"A"/"C"/"I" HANYA jika kata RACI eksplisit disebut atau tersirat jelas dari kata kerja
   (misal "bertanggung jawab melakukan" -> R, "menyetujui"/"mengesahkan" -> A,
   "dikonsultasikan" -> C, "diinformasikan" -> I), else null

4. agg_type (ISI HANYA jika intent="aggregate"), salah satu dari:
   - "count_by_role": hitung total proses RACI milik SATU role tertentu
   - "top_by_raci": ranking role dengan jumlah RACI type tertentu terbanyak
   - "count_roles_in_category": hitung jumlah role unik dalam suatu kategori/topik proses
   - "count_tupoksi_by_role": hitung jumlah tugas pokok/tupoksi milik satu role tertentu

5. chart_requested:
   - true JIKA user secara eksplisit meminta chart, grafik, visualisasi, diagram, pie chart,
     bar chart, atau bentuk visualisasi lainnya
   - false JIKA user hanya meminta jawaban/data tanpa meminta visualisasi

Jawab HANYA dalam format JSON berikut, TANPA teks lain, TANPA markdown code fence:
{{"intent": "...", "entity": "...", "raci_filter": null, "agg_type": null, "chart_requested": false}}

Contoh:

Q: "siapa yang bertanggung jawab untuk mengelola konflik kepentingan"
A: {{"intent": "process_to_role", "entity": "mengelola konflik kepentingan", "raci_filter": "R", "agg_type": null, "chart_requested": false}}

Q: "siapa yang menyetujui anggaran belanja modal"
A: {{"intent": "process_to_role", "entity": "anggaran belanja modal", "raci_filter": "A", "agg_type": null, "chart_requested": false}}

Q: "siapa yang dikonsultasikan dalam proses pengadaan EPC"
A: {{"intent": "process_to_role", "entity": "pengadaan EPC", "raci_filter": "C", "agg_type": null, "chart_requested": false}}

Q: "proses apa saja yang menjadi tanggung jawab Manager Manajemen Integritas"
A: {{"intent": "role_to_process", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": null, "chart_requested": false}}

Q: "apa saja tupoksi Manager Manajemen Integritas"
A: {{"intent": "role_to_tupoksi", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": null, "chart_requested": false}}

Q: "tampilkan uraian jabatan Manager Manajemen Integritas"
A: {{"intent": "role_to_tupoksi", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": null, "chart_requested": false}}

Q: "berapa banyak proses yang menjadi tanggung jawab Manager Manajemen Integritas"
A: {{"intent": "aggregate", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": "count_by_role", "chart_requested": false}}

Q: "berapa banyak proses yang menjadi tanggung jawab Manager Manajemen Integritas? tampilkan chart"
A: {{"intent": "aggregate", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": "count_by_role", "chart_requested": true}}

Q: "buatkan grafik distribusi RACI Manager Manajemen Integritas"
A: {{"intent": "aggregate", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": "count_by_role", "chart_requested": true}}

Q: "role apa yang paling banyak menjadi Accountable"
A: {{"intent": "aggregate", "entity": null, "raci_filter": "A", "agg_type": "top_by_raci", "chart_requested": false}}

Q: "tampilkan chart role yang paling banyak menjadi Accountable"
A: {{"intent": "aggregate", "entity": null, "raci_filter": "A", "agg_type": "top_by_raci", "chart_requested": true}}

Q: "berapa banyak role yang terlibat dalam pengadaan EPC"
A: {{"intent": "aggregate", "entity": "pengadaan EPC", "raci_filter": null, "agg_type": "count_roles_in_category", "chart_requested": false}}

Q: "berapa banyak tugas pokok Manager Manajemen Integritas"
A: {{"intent": "aggregate", "entity": "Manager Manajemen Integritas", "raci_filter": null, "agg_type": "count_tupoksi_by_role", "chart_requested": false}}

Q: "siapa saja yang terlibat dalam pengadaan EPC pembangkit listrik"
A: {{"intent": "topic_to_roles", "entity": "pengadaan EPC pembangkit listrik", "raci_filter": null, "agg_type": null, "chart_requested": false}}

Q: "siapa saja yang berperan dalam kategori manajemen risiko"
A: {{"intent": "topic_to_roles", "entity": "manajemen risiko", "raci_filter": null, "agg_type": null, "chart_requested": false}}


Riwayat percakapan sebelumnya (kosong jika tidak ada):
{history}
Pertanyaan: {question}
Jawaban:"""


def extract_intent(question, chat_client, history):
    history_text = format_history(history)

    kwargs = dict(
        model=config.CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": INTENT_PROMPT.format(
                question=question,
                history=history_text or "(tidak ada)",
            ),
        }],
        max_tokens=500,
        temperature=0,
    )

    try:
        resp = chat_client.chat.completions.create(
            **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:
        resp = chat_client.chat.completions.create(**kwargs)

    msg = resp.choices[0].message
    raw = msg.content or getattr(msg, "reasoning_content", None)

    if not raw:
        return {
            "intent": "unclear",
            "entity": None,
            "raci_filter": None,
            "agg_type": None,
            "chart_requested": False,
        }

    raw = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)

        # Ensure the field always exists even if the model forgets it.
        result.setdefault("chart_requested", False)

        return result

    except json.JSONDecodeError:
        return {
            "intent": "unclear",
            "entity": None,
            "raci_filter": None,
            "agg_type": None,
            "chart_requested": False,
        }