import re
from pathlib import Path

from django.conf import settings

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _index_path(video_id: str) -> Path:
    return Path(settings.FAISS_INDEX_DIR) / video_id


def _get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )


def fetch_transcript(video_id: str) -> str:
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id)
        return " ".join(seg.text for seg in transcript)
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video.")
    except Exception as exc:
        raise ValueError(f"Failed to fetch transcript: {exc}")


def build_faiss_index(video_id: str, transcript: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript)
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    index_dir = _index_path(video_id)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def load_faiss_index(video_id: str) -> FAISS | None:
    index_dir = _index_path(video_id)
    if not index_dir.exists():
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def index_exists(video_id: str) -> bool:
    return _index_path(video_id).exists()


RAG_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on a YouTube video transcript.
Use ONLY the provided context to answer the question. If the answer is not in the context,
say "I couldn't find that information in the video."

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(video_id: str, question: str) -> dict:
    vectorstore = load_faiss_index(video_id)
    if vectorstore is None:
        raise ValueError("No index found for this video. Please load the video first.")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_K}
    )

    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    source_docs = retriever.invoke(question)

    chain = (
        {"context": lambda _: _format_docs(source_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    return {
        "answer": answer,
        "source_chunks": [doc.page_content for doc in source_docs],
    }