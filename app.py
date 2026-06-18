from __future__ import annotations

from typing import Any

import gradio as gr
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from agents import GroundingValidator, IntakeAgent, RAGOrchestrator, ResponseAgent, RetrievalAgent
from config import get_settings
from database import SupabaseGateway
from ingestion import KnowledgeIngestionService
from utils import parse_latency_to_seconds


settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)
embeddings = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)

db = SupabaseGateway(settings)
ingestion = KnowledgeIngestionService(settings, db, embeddings)

orchestrator = RAGOrchestrator(
    settings=settings,
    intake=IntakeAgent(),
    retrieval=RetrievalAgent(settings, db, embeddings),
    response=ResponseAgent(settings, openai_client),
    validator=GroundingValidator(settings),
)

CHAT_SESSION_ID = db.create_chat_session(settings.app_title)
print("CHAT_SESSION_ID:", CHAT_SESSION_ID)


def transcribe_audio(audio_path: str, language_hint: str | None = None) -> str:
    if not audio_path:
        return ""

    with open(audio_path, "rb") as audio_file:
        resp = openai_client.audio.transcriptions.create(
            model=settings.transcription_model,
            file=audio_file,
            language=language_hint,
        )
    return resp.text


def chat_api(message: str) -> dict[str, Any]:
    message = (message or "").strip()
    if not message:
        return {"answer": "", "latency": "", "sources_pages": [], "confidence": 0.0, "route": "empty"}

    db.save_chat_message(CHAT_SESSION_ID, "user", message)

    final_answer = ""
    final_latency = ""
    final_hits: list[dict[str, Any]] = []
    final_confidence = 0.0
    final_route = ""
    final_grounded = False

    for answer, latency, hits, confidence, route, grounded in orchestrator.answer_stream(message, history=[]):
        if answer:
            final_answer = answer
        if latency:
            final_latency = latency
        final_hits = hits or []
        final_confidence = confidence
        final_route = route
        final_grounded = grounded

    db.save_chat_message(
        CHAT_SESSION_ID,
        "assistant",
        final_answer,
        parse_latency_to_seconds(final_latency),
        metadata={"route": final_route, "confidence": final_confidence, "grounded": final_grounded},
    )

    source_pages = sorted({hit.get("page") + 1 for hit in final_hits if hit.get("page") is not None})
    return {
        "answer": final_answer,
        "latency": final_latency,
        "sources_pages": source_pages,
        "confidence": final_confidence,
        "route": final_route,
        "grounded": final_grounded,
    }


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
        return chat_ui, history, "Please record audio first."

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
    db.save_chat_message(CHAT_SESSION_ID, "user", raw_user_text)

    final_answer = ""
    final_latency = ""
    final_confidence = 0.0
    final_route = ""
    final_grounded = False

    try:
        for answer, latency, _hits, confidence, route, grounded in orchestrator.answer_stream(
            raw_user_text,
            history=history,
        ):
            if answer:
                final_answer = answer
                chat_ui[-1]["content"] = final_answer

            if latency:
                final_latency = latency
                final_confidence = confidence
                final_route = route
                final_grounded = grounded
                chat_ui[-1]["content"] = (
                    f"{final_answer}\n\n"
                    f"Latency: {final_latency}\n"
                    f"Route: {final_route} | Confidence: {final_confidence:.2f} | Grounded: {final_grounded}"
                )

            yield chat_ui, history

        db.save_chat_message(
            CHAT_SESSION_ID,
            "assistant",
            final_answer,
            parse_latency_to_seconds(final_latency),
            metadata={"route": final_route, "confidence": final_confidence, "grounded": final_grounded},
        )
        history = history + [(raw_user_text, final_answer)]
        yield chat_ui, history

    except Exception as exc:
        chat_ui[-1]["content"] = f"Error: {type(exc).__name__}: {exc}"
        yield chat_ui, history


def upload_pdfs(files):
    try:
        doc_count, chunk_count = ingestion.ingest_pdfs(files)
        return f"Uploaded {doc_count} PDF(s), inserted {chunk_count} chunks."
    except Exception as exc:
        return f"PDF upload error: {type(exc).__name__}: {exc}"


def upload_faq(file):
    try:
        count = ingestion.ingest_faq_csv(file)
        return f"Uploaded {count} FAQ items."
    except Exception as exc:
        return f"FAQ upload error: {type(exc).__name__}: {exc}"


CUSTOM_CSS = """
#chatbot {
    height: 520px;
    overflow: auto;
}
"""

with gr.Blocks(title=settings.app_title) as demo:
    gr.Markdown("## Axion RAG Chatbot | Modular RAG | FAQ + PDF | Text + Voice")

    with gr.Tab("Chat"):
        history_state = gr.State([])
        chat_state = gr.State([])

        chatbot = gr.Chatbot(elem_id="chatbot", label="Chat", type="messages")

        with gr.Row():
            user_box = gr.Textbox(label="Message", placeholder="Ask anything about Axion...", lines=1)
            voice_in = gr.Audio(sources=["microphone"], type="filepath", label="Voice")
            voice_send = gr.Button("Send voice")

        voice_status = gr.Textbox(visible=False)

        user_box.submit(
            _append_user_message,
            inputs=[user_box, chat_state, history_state],
            outputs=[user_box, chat_state, history_state],
            queue=False,
        ).then(
            chat_stream_last,
            inputs=[chat_state, history_state],
            outputs=[chatbot, history_state],
        )

        voice_send.click(
            _append_voice_message,
            inputs=[voice_in, chat_state, history_state],
            outputs=[chat_state, history_state, voice_status],
            queue=False,
        ).then(
            chat_stream_last,
            inputs=[chat_state, history_state],
            outputs=[chatbot, history_state],
        )

    with gr.Tab("Admin Upload"):
        gr.Markdown("### Upload PDF Knowledge Base")
        pdf_files = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
        pdf_upload_btn = gr.Button("Upload PDFs")
        pdf_status = gr.Textbox(label="PDF Upload Status")
        pdf_upload_btn.click(upload_pdfs, inputs=[pdf_files], outputs=[pdf_status])

        gr.Markdown("### Upload FAQ CSV")
        gr.Markdown("CSV columns must be: `question`, `answer`, `category`")
        faq_file = gr.File(label="Upload FAQ CSV", file_count="single", file_types=[".csv"])
        faq_upload_btn = gr.Button("Upload FAQ")
        faq_status = gr.Textbox(label="FAQ Upload Status")
        faq_upload_btn.click(upload_faq, inputs=[faq_file], outputs=[faq_status])

    gr.api(fn=chat_api, api_name="chat", api_description="Send a text message to the Axion chatbot.")


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
