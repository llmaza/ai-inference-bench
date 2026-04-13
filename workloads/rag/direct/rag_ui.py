#!/usr/bin/env python3
"""
Веб-интерфейс RAG по Трудовому кодексу (Streamlit).

Установка: pip install streamlit
Запуск:    streamlit run rag_ui.py

Нужны workloads/rag/.env (QDRANT_URL, локальные модели, …), Qdrant и чанки —
как для rag_pipeline.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from rag_pipeline import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_DEFAULT,
    LLM_DEFAULT,
    RERANKER_DEFAULT,
    TOP_K_AFTER_RERANK,
    TOP_K_EACH,
    _load_dotenv,
    build_chat_messages_for_rag,
    build_context_limited,
    build_context,
    load_chunks,
    resolve_reranker_backend,
    retrieve_top_chunks,
    run_llm,
    run_llm_chat,
    tokenize_ru,
)

_UI_DIR = Path(__file__).resolve().parent / "ui"
_CSS_FILE = _UI_DIR / "zanrag_streamlit.css"
_BRAND_LABEL = "Decentrathon 5.0 x AI INDRIVE"


def _inject_zan_theme() -> None:
    if _CSS_FILE.is_file():
        css = _CSS_FILE.read_text(encoding="utf-8")
    else:
        css = ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _render_zan_hero() -> None:
    cell = f'<span class="zan-rail__item">{_BRAND_LABEL}</span>'
    track_inner = (cell * 18) + (cell * 18)
    st.markdown(
        f"""
<div class="zan-hero-wrap">
  <div class="zan-rail zan-rail--a"><div class="zan-rail__track">{track_inner}</div></div>
  <div class="zan-rail zan-rail--b"><div class="zan-rail__track">{track_inner}</div></div>
  <div class="zan-hero-inner">
    <div class="zan-logo">iD</div>
    <div class="zan-hero-text">
      <span class="zan-eyebrow">{_BRAND_LABEL}</span>
      <h1 class="zan-title">ZanRAG</h1>
      <p class="zan-sub">Гибридный анализ по трудовому праву РК: BM25 + векторный поиск, локальный реранкинг и локальный ответ LLM с опорой на фрагменты Трудового кодекса.</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def _rag_backend(
    chunks_path_str: str,
    qdrant_url: str,
    qdrant_api_key: str | None,
    qdrant_collection: str,
    embed_model: str,
    llm_model: str,
) -> dict:
    _load_dotenv()
    path = Path(chunks_path_str)
    chunks = load_chunks(path)
    if not chunks:
        raise FileNotFoundError(f"Нет чанков: {path}")
    tokenized = [tokenize_ru(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    from workloads.small_llm.llm_inference import get_local_chat_backend

    llm_backend = get_local_chat_backend(model_name=llm_model)
    return {
        "chunks": chunks,
        "bm25": bm25,
        "qdrant": qdrant,
        "collection": qdrant_collection,
        "embed_model": embed_model,
        "llm_backend": llm_backend,
    }


def main() -> None:
    st.set_page_config(
        page_title="ZanRAG",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="⚖️",
    )
    _inject_zan_theme()
    _render_zan_hero()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    _load_dotenv()

    with st.sidebar:
        st.markdown(
            f'<div class="zan-sidebar-head"><span class="zan-eyebrow">{_BRAND_LABEL}</span>'
            '<h2 class="zan-sidebar-h2">Параметры</h2></div>',
            unsafe_allow_html=True,
        )
        mode = st.radio("Режим", ("Один вопрос", "Чат"), horizontal=False)
        llm_model = st.text_input("Локальная модель LLM", value=os.environ.get("HF_LLM_MODEL", LLM_DEFAULT))
        collection = st.text_input(
            "Qdrant collection",
            value=os.environ.get("QDRANT_COLLECTION", "labor_code_tk_e5"),
        )
        embed_model = st.text_input("Модель эмбеддингов", value=os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT))
        reranker_model = st.text_input("Модель реранкера", value=os.environ.get("HF_RERANKER_MODEL", RERANKER_DEFAULT))
        reranker_backend = st.selectbox(
            "Бэкенд реранкера",
            options=("torch", "onnx"),
            index=0 if resolve_reranker_backend(os.environ.get("RERANK_BACKEND")) == "torch" else 1,
        )
        top_each = st.slider("BM25 / dense, top каждый", 10, 80, TOP_K_EACH)
        top_final = st.slider("После реранка, чанков", 5, 40, TOP_K_AFTER_RERANK)
        max_context_chars = st.slider(
            "Лимит контекста (символы)",
            min_value=0,
            max_value=120_000,
            value=32_000,
            step=2000,
            help="0 = без лимита (все выбранные чанки). Иначе обрезка по порядку реранка.",
        )
        llm_max_tokens = st.slider("max_tokens ответа LLM", 256, 8192, 2048, step=256)
        max_history = st.slider("История в чате (последних сообщений)", 0, 32, 12, step=2)
        qdrant_url = st.text_input("QDRANT_URL", value=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        qdrant_key = st.text_input("QDRANT_API_KEY (опц.)", value=os.environ.get("QDRANT_API_KEY", ""), type="password")
        if mode == "Чат":
            if st.button("Очистить историю чата"):
                st.session_state.chat_messages = []
                st.rerun()

    chunks_path = str(CHUNKS_PATH.resolve())
    try:
        bk = _rag_backend(
            chunks_path,
            qdrant_url,
            qdrant_key or None,
            collection,
            embed_model,
            llm_model,
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    n_chunks = len(bk["chunks"])
    mode_chip = "Чат" if mode == "Чат" else "Один вопрос"
    st.markdown(
        f'<div class="zan-meta-row">'
        f'<span class="zan-chip">Корпус: <b>{n_chunks}</b> чанков</span>'
        f'<span class="zan-chip">Режим: <b>{mode_chip}</b></span>'
        f'<span class="zan-chip">Коллекция: <b>{collection}</b></span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    def do_rag(user_text: str, history: list[dict]) -> tuple[str, str]:
        top, _reranked, _timings = retrieve_top_chunks(
            bk["bm25"],
            bk["chunks"],
            bk["qdrant"],
            user_text,
            bk["collection"],
            bk["embed_model"],
            reranker_model,
            reranker_backend,
            top_each,
            top_final,
        )
        ctx = (
            build_context(top)
            if max_context_chars <= 0
            else build_context_limited(top, max_context_chars)
        )
        if mode == "Чат":
            prior = [m for m in history if m.get("role") in ("user", "assistant")]
            msgs = build_chat_messages_for_rag(prior, user_text, ctx, max_history)
            answer = run_llm_chat(bk["llm_backend"], llm_model, msgs, llm_max_tokens)
        else:
            answer = run_llm(bk["llm_backend"], llm_model, user_text, ctx, llm_max_tokens)
        return answer, ctx

    if mode == "Один вопрос":
        st.markdown('<p class="zan-panel-label">Запрос</p>', unsafe_allow_html=True)
        with st.container(border=True):
            q = st.text_area("Вопрос", height=100, placeholder="Например: срок испытательного срока?")
            if st.button("Ответить", type="primary") and q.strip():
                with st.spinner("Поиск и генерация…"):
                    ans, ctx = do_rag(q.strip(), [])
                st.markdown(ans)
                with st.expander("Контекст, ушедший в LLM"):
                    st.text(ctx[:200_000] + ("…" if len(ctx) > 200_000 else ""))
    else:
        st.markdown('<p class="zan-panel-label">Чат</p>', unsafe_allow_html=True)
        with st.container(border=True):
            for m in st.session_state.chat_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if prompt := st.chat_input("Вопрос по Трудовому кодексу РК…"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            hist_before = st.session_state.chat_messages[:-1]
            with st.spinner("Поиск и генерация…"):
                try:
                    ans, _ctx = do_rag(prompt, hist_before)
                except Exception as e:
                    ans = f"**Ошибка:** {e}"
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            st.rerun()


if __name__ == "__main__":
    main()
