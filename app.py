import os
import dotenv import load_dotenv
import re
import csv
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Tuple, List

import gradio as gr
from openai import OpenAI
from supabase import create_client

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# 1) ENV / CONNECTIONS
# =========================
load_dotenv()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in Space Secrets.")
if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is not set in Space Secrets.")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is not set in Space Secrets.")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 2) CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

LEBANON_TZ = ZoneInfo("Asia/Beirut")
LATENCY_COLUMN = "latency_s"

FAQ_SIMILARITY_THRESHOLD = 0.85
FAQ_TOP_K = 3
PDF_TOP_K = 4


# =========================
# 3) MODELS + SPLITTER
# =========================
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", "。", ".", "؟", "!", "؛", "،", " ", ""],
)


# =========================
# 4) CHAT SESSION + MESSAGES
# =========================
def create_chat_session(title: str | None = None) -> str:
    resp = sb.table("chat_sessions").insert({"title": title}).execute()

    if not resp.data:
        raise RuntimeError(f"Failed to create chat session: {resp}")

    return resp.data[0]["id"]


def save_chat_message(
    session_id: str | None,
    role: str,
    content: str,
    latency_s: float | None = None,
):
    if not session_id:
        return None

    payload = {
        "session_id": session_id,
        "role": role,
        "content": content,
    }

    if latency_s is not None:
        payload[LATENCY_COLUMN] = latency_s

    try:
        return sb.table("chat_messages").insert(payload).execute()
    except Exception as e:
        print(f"[Supabase chat insert error] {type(e).__name__}: {e}")
        return None


def parse_latency_to_seconds(latency_str: str) -> float | None:
    try:
        match = re.match(r"^\s*([0-9]*\.?[0-9]+)s", latency_str or "")
        return round(float(match.group(1)), 3) if match else None
    except Exception:
        return None


CHAT_SESSION_ID = None

try:
    CHAT_SESSION_ID = create_chat_session("Axion RAG Chat")
    print("Connected to Supabase. CHAT_SESSION_ID:", CHAT_SESSION_ID)
except Exception as e:
    print(f"Supabase startup warning: {type(e).__name__}: {e}")


# =========================
# 5) GENERAL HELPERS
# =========================
def detect_lang(text: str) -> str:
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text or ""):
        return "ar"
    return "en"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def now_in_lebanon() -> datetime:
    return datetime.now(LEBANON_TZ)


def format_now(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " (Asia/Beirut)"


def format_history(history: List[Tuple[str, str]], max_turns: int = 5) -> str:
    if not history:
        return "(none)"

    lines = []

    for user_msg, assistant_msg in history[-max_turns:]:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")

    return "\n".join(lines)


def build_system_prompt(lang: str) -> str:
    now_str = format_now(now_in_lebanon())

    return f"""You are an AI Receptionist and Customer Support Assistant for Axion.
CAPABILITIES:
1. FAQ Knowledge Base: Common questions and official answers.
2. PDF Knowledge Base: Company info, services, policies, and uploaded documents.
RESPONSE RULES:
- Answer in the user's language, Arabic or English.
- Keep answers accurate, brief, and professional.
- For greetings and small talk, answer directly and warmly.
- Use only the provided knowledge base context when answering company-specific questions.
- If the answer is not found in the context, say that you do not have enough information.
- Do not invent prices, policies, contact info, services, or company details.
Current date/time: {now_str}
"""


# =========================
# 6) FAQ HELPERS
# =========================
def retrieve_faq(question: str, k: int = FAQ_TOP_K) -> list[dict]:
    q_vec = embeddings.embed_query(question)

    resp = sb.rpc(
        "match_faq_items",
        {
            "query_embedding": q_vec,
            "match_count": k,
        },
    ).execute()

    return resp.data or []


def get_best_faq_answer(question: str) -> tuple[str | None, dict | None]:
    try:
        hits = retrieve_faq(question, k=FAQ_TOP_K)
    except Exception as e:
        print(f"FAQ retrieval error: {type(e).__name__}: {e}")
        return None, None

    if not hits:
        return None, None

    best = hits[0]
    similarity = float(best.get("similarity") or 0)

    if similarity >= FAQ_SIMILARITY_THRESHOLD:
        return best.get("answer"), best

    return None, best


def ingest_faq_csv(file, batch_size: int = 64) -> int:
    if not file:
        return 0

    faq_rows = []

    with open(file.name, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            category = (row.get("category") or "").strip()

            if question and answer:
                faq_rows.append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                    }
                )

    if not faq_rows:
        return 0

    total_inserted = 0

    for start_i in range(0, len(faq_rows), batch_size):
        batch_rows = faq_rows[start_i:start_i + batch_size]
        batch_questions = [row["question"] for row in batch_rows]

        vectors = embeddings.embed_documents(batch_questions)

        rows_to_insert = []

        for row, vector in zip(batch_rows, vectors):
            rows_to_insert.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "embedding": vector,
                }
            )

        resp = sb.table("faq_items").insert(rows_to_insert).execute()

        if resp.data:
            total_inserted += len(resp.data)
        else:
            total_inserted += len(rows_to_insert)

    return total_inserted


# =========================
# 7) PDF RAG HELPERS
# =========================
def supabase_create_document(filename: str) -> str:
    resp = sb.table("rag_documents").insert({"filename": filename}).execute()

    if not resp.data:
        raise RuntimeError(f"Failed to insert rag_documents row for {filename}: {resp}")

    return resp.data[0]["id"]


def supabase_insert_chunks(rows: List[dict]):
    if not rows:
        return

    resp = sb.table("rag_chunks").insert(rows).execute()

    if getattr(resp, "error", None):
        print("[Supabase rag_chunks insert error]", resp.error)


def ingest_pdfs_to_supabase(files, batch_size: int = 64) -> Tuple[int, int]:
    if not files:
        return 0, 0

    total_chunks = 0

    for file in files:
        filename = Path(file.name).name
        doc_id = supabase_create_document(filename)

        loader = PyPDFLoader(file.name)
        docs = loader.load()

        splits = splitter.split_documents(docs)

        chunk_texts = [doc.page_content.strip() for doc in splits if doc.page_content.strip()]
        chunk_pages = [
            doc.metadata.get("page", None)
            for doc in splits
            if doc.page_content.strip()
        ]

        for start_i in range(0, len(chunk_texts), batch_size):
            end_i = start_i + batch_size

            batch_texts = chunk_texts[start_i:end_i]
            batch_pages = chunk_pages[start_i:end_i]

            vectors = embeddings.embed_documents(batch_texts)

            rows = [
                {
                    "document_id": doc_id,
                    "chunk_index": start_i + j,
                    "page": page,
                    "content": text,
                    "embedding": vector,
                }
                for j, (text, page, vector) in enumerate(
                    zip(batch_texts, batch_pages, vectors)
                )
            ]

            supabase_insert_chunks(rows)

        total_chunks += len(chunk_texts)

    return len(files), total_chunks


def retrieve_chunks(question: str, k: int = PDF_TOP_K) -> List[dict]:
    q_vec = embeddings.embed_query(question)

    resp = sb.rpc(
        "match_rag_chunks",
        {
            "query_embedding": q_vec,
            "match_count": k,
        },
    ).execute()

    return resp.data or []


# =========================
# 8) CORE ANSWER FUNCTION
# =========================
def rag_answer_stream(
    question: str,
    history: List[Tuple[str, str]] | None = None,
    k: int = PDF_TOP_K,
):
    start = time.perf_counter()

    history = history or []
    question = (question or "").strip()
    lang = detect_lang(question)

    qnorm = normalize_text(question)

    smalltalk_set = {
        "hi",
        "hello",
        "hey",
        "السلام عليكم",
        "مرحبا",
        "هلا",
        "هاي",
        "شكرا",
        "thanks",
        "thank you",
    }

    if qnorm in smalltalk_set:
        answer = (
            "Hello! How can I assist you today?"
            if lang == "en"
            else "مرحباً! كيف يمكنني مساعدتك اليوم؟"
        )

        total = time.perf_counter() - start
        yield answer, f"{total:.2f}s", []
        return

    # =========================
    # FAQ FIRST
    # =========================
    t_faq = time.perf_counter()

    faq_answer, faq_hit = get_best_faq_answer(question)

    faq_s = time.perf_counter() - t_faq

    if faq_answer:
        total = time.perf_counter() - start

        final = faq_answer.strip()
        final += "\n\nSource: FAQ"

        latency = f"{total:.2f}s (FAQ {faq_s:.2f}s)"

        yield final, latency, []
        return

    # =========================
    # PDF RAG FALLBACK
    # =========================
    t_retrieval = time.perf_counter()

    try:
        rag_hits = retrieve_chunks(question, k=k)
    except Exception as e:
        rag_hits = []
        print(f"RAG retrieval error: {type(e).__name__}: {e}")

    retrieval_s = time.perf_counter() - t_retrieval

    pdf_context = ""

    if rag_hits:
        pdf_context = "\n\n".join(
            f"[p{hit.get('page', '?')}] {hit.get('content', '')}"
            for hit in rag_hits
        )

    system_prompt = build_system_prompt(lang)
    history_text = format_history(history)

    user_content = (
        f"Conversation so far:\n{history_text}\n\n"
        f"PDF Knowledge Base Context:\n{pdf_context or '(none)'}\n\n"
        f"User question: {question}"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    t_llm = time.perf_counter()
    full = ""

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""

            if token:
                full += token
                yield full.strip(), "", rag_hits

    except Exception as e:
        error_message = f"❌ LLM error: {type(e).__name__}: {e}"
        yield error_message, "", rag_hits
        return

    llm_s = time.perf_counter() - t_llm

    final = full.strip()

    if rag_hits:
        pages = sorted(
            {
                hit.get("page")
                for hit in rag_hits
                if hit.get("page") is not None
            }
        )

        if pages:
            final += "\n\nSources pages: " + ", ".join(
                str(page + 1) for page in pages
            )

    total = time.perf_counter() - start

    latency = (
        f"{total:.2f}s "
        f"(FAQ {faq_s:.2f}s, "
        f"RAG {retrieval_s:.2f}s, "
        f"LLM {llm_s:.2f}s)"
    )

    yield final, latency, rag_hits


# =========================
# 9) AUDIO / SPEECH TO TEXT
# =========================
def transcribe_audio(audio_path: str, language_hint: str | None = None) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return ""

    with open(audio_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language=language_hint,
        )

    return resp.text


# =========================
# 10) API FUNCTION FOR WEBSITE
# =========================
def chat_api(message: str):
    message = (message or "").strip()

    if not message:
        return {
            "answer": "",
            "latency": "",
            "sources_pages": [],
        }

    save_chat_message(CHAT_SESSION_ID, "user", message)

    final_answer = ""
    final_latency = ""
    final_hits = []

    for partial_answer, latency, hits in rag_answer_stream(message, history=[]):
        if partial_answer:
            final_answer = partial_answer

        if latency:
            final_latency = latency

        if hits is not None:
            final_hits = hits

    latency_s = parse_latency_to_seconds(final_latency)

    save_chat_message(
        CHAT_SESSION_ID,
        "assistant",
        final_answer,
        latency_s,
    )

    source_pages = sorted(
        {
            hit.get("page") + 1
            for hit in final_hits
            if hit.get("page") is not None
        }
    )

    return {
        "answer": final_answer,
        "latency": final_latency,
        "sources_pages": source_pages,
    }


# =========================
# 11) UI CHAT HELPERS
# =========================
def _append_user_message(user_text: str, chat_ui: list, history: list):
    chat_ui = chat_ui or []
    history = history or []

    if not user_text or not user_text.strip():
        return "", chat_ui, history

    chat_ui.append({"role": "user", "content": user_text})
    chat_ui.append({"role": "assistant", "content": ""})

    return "", chat_ui, history


def _append_voice_message(audio_path: str, chat_ui: list, history: list):
    chat_ui = chat_ui or []
    history = history or []

    transcript = transcribe_audio(audio_path)

    if not transcript.strip():
        return chat_ui, history, "⚠️ Please record audio first."

    chat_ui.append({"role": "user", "content": f"🎤 {transcript}"})
    chat_ui.append({"role": "assistant", "content": ""})

    return chat_ui, history, ""


def chat_stream_last(chat_ui: list, history: list):
    chat_ui = chat_ui or []
    history = history or []

    if len(chat_ui) < 2:
        yield chat_ui, history
        return

    raw_user_text = str(chat_ui[-2]["content"]).replace("🎤 ", "", 1)

    save_chat_message(CHAT_SESSION_ID, "user", raw_user_text)

    final_answer = ""
    final_latency = ""

    try:
        for partial_answer, latency, _hits in rag_answer_stream(
            raw_user_text,
            history=history,
        ):
            if partial_answer:
                final_answer = partial_answer
                chat_ui[-1]["content"] = final_answer

            if latency:
                final_latency = latency
                chat_ui[-1]["content"] = f"{final_answer}\n\n⏱️ {final_latency}"

            yield chat_ui, history

        latency_s = parse_latency_to_seconds(final_latency)

        save_chat_message(CHAT_SESSION_ID, "assistant", final_answer, latency_s)

        chat_ui[-1]["content"] = f"{final_answer}\n\n⏱️ {final_latency}"

        history = history + [(raw_user_text, final_answer)]

        yield chat_ui, history

    except Exception as e:
        chat_ui[-1]["content"] = f"❌ Error: {type(e).__name__}: {e}"
        yield chat_ui, history


# =========================
# 12) ADMIN INGEST HELPERS
# =========================
def upload_pdfs(files):
    try:
        doc_count, chunk_count = ingest_pdfs_to_supabase(files)
        return f"✅ Uploaded {doc_count} PDF(s), inserted {chunk_count} chunks."
    except Exception as e:
        return f"❌ PDF upload error: {type(e).__name__}: {e}"


def upload_faq(file):
    try:
        count = ingest_faq_csv(file)
        return f"✅ Uploaded {count} FAQ items."
    except Exception as e:
        return f"❌ FAQ upload error: {type(e).__name__}: {e}"


# =========================
# 13) UI
# =========================
CUSTOM_CSS = """
#chatbot {
    height: 520px;
    overflow: auto;
}
"""

with gr.Blocks() as demo:

    gr.Markdown("## Axion RAG Chatbot | FAQ + PDF RAG | Text + Voice")

    with gr.Tab("Chat"):
        history_state = gr.State([])
        chat_state = gr.State([])

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Chat"
        )

        with gr.Row():
            user_box = gr.Textbox(
                label="Message",
                placeholder="Ask anything about Axion…",
                lines=1,
            )

            voice_in = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Voice",
            )

            voice_send = gr.Button("Send voice")

        voice_status = gr.Textbox(visible=False)

        user_box.submit(
            _append_user_message,
            inputs=[
                user_box,
                chat_state,
                history_state,
            ],
            outputs=[
                user_box,
                chat_state,
                history_state,
            ],
            queue=False,
        ).then(
            chat_stream_last,
            inputs=[
                chat_state,
                history_state,
            ],
            outputs=[
                chatbot,
                history_state,
            ],
        )

        voice_send.click(
            _append_voice_message,
            inputs=[
                voice_in,
                chat_state,
                history_state,
            ],
            outputs=[
                chat_state,
                history_state,
                voice_status,
            ],
            queue=False,
        ).then(
            chat_stream_last,
            inputs=[
                chat_state,
                history_state,
            ],
            outputs=[
                chatbot,
                history_state,
            ],
        )

    with gr.Tab("Admin Upload"):
        gr.Markdown("### Upload PDF Knowledge Base")
        pdf_files = gr.File(
            label="Upload PDFs",
            file_count="multiple",
            file_types=[".pdf"],
        )
        pdf_upload_btn = gr.Button("Upload PDFs")
        pdf_status = gr.Textbox(label="PDF Upload Status")

        pdf_upload_btn.click(
            upload_pdfs,
            inputs=[pdf_files],
            outputs=[pdf_status],
        )

        gr.Markdown("### Upload FAQ CSV")
        gr.Markdown(
            "CSV columns must be: `question`, `answer`, `category`"
        )

        faq_file = gr.File(
            label="Upload FAQ CSV",
            file_count="single",
            file_types=[".csv"],
        )
        faq_upload_btn = gr.Button("Upload FAQ")
        faq_status = gr.Textbox(label="FAQ Upload Status")

        faq_upload_btn.click(
            upload_faq,
            inputs=[faq_file],
            outputs=[faq_status],
        )

    gr.api(
        fn=chat_api,
        api_name="chat",
        api_description="Send a text message to the Axion chatbot.",
    )

demo.launch(css=CUSTOM_CSS)