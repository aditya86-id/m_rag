import streamlit as st
import os
from dotenv import load_dotenv
import base64
import mimetypes
import tempfile
from urllib.parse import parse_qs, urlparse
import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── youtube-transcript-api v1.x compatible imports ──────────────────────────
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

load_dotenv()
os.environ.setdefault("USER_AGENT", "multimodal-rag-chatbot/1.0")

st.set_page_config(page_title="RAG Multimodal Chatbot", page_icon="🤖", layout="wide")

UPLOAD_FOLDER = "uploads"
VECTOR_STORE_PATH = "faiss_index"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------
# Model loaders
# --------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


@st.cache_resource
def load_asr_pipeline():
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Whisper fallback needs `transformers` and `torch` installed."
        ) from exc

    model_name = os.getenv("HF_ASR_MODEL", "openai/whisper-small")
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=device,
    )


embedding_model = load_embeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

# --------------------------
# Session initialization
# --------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [{"sender": "bot", "text": "Hello! Upload files and ask questions."}]

if "vector_store" not in st.session_state:
    if os.path.exists(VECTOR_STORE_PATH):
        st.session_state.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        st.session_state.vector_store = None

if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None


# --------------------------
# Vectorstore helpers
# --------------------------

def add_to_vectorstore(text: str, source: str):
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(documents, embedding_model)
    else:
        st.session_state.vector_store.add_documents(documents)

    st.session_state.vector_store.save_local(VECTOR_STORE_PATH)


def extract_text_from_pdf(file):
    path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader(path)
    text = "\n".join(p.page_content for p in loader.load())
    add_to_vectorstore(text, file.name)
    return text


def extract_text_from_txt(file):
    path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    loader = TextLoader(path)
    text = "\n".join(p.page_content for p in loader.load())
    add_to_vectorstore(text, file.name)
    return text


def extract_text_from_docx(file):
    path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    loader = Docx2txtLoader(path)
    docs = loader.load()
    text = docs[0].page_content
    add_to_vectorstore(text, file.name)
    return text


def extract_text_from_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n".join(d.page_content for d in docs)
    add_to_vectorstore(text, url)
    return text


# --------------------------
# YouTube helpers (v1.x API)
# --------------------------

def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]

    if not video_id:
        raise ValueError("Invalid YouTube URL. Please paste a full YouTube video link.")
    return video_id


def _fetch_transcript_v1(video_id: str) -> str:
    """
    Fetch transcript using youtube-transcript-api v1.x.

    v1.x changes vs v0.6.x:
      - YouTubeTranscriptApi is now instantiated, not used as a class with static methods.
      - .fetch() returns a FetchedTranscript object; iterate it to get snippet dicts.
      - Snippet dicts have keys: 'text', 'start', 'duration'.
    """
    ytt = YouTubeTranscriptApi()

    # Try preferred languages first
    for lang in (["en"], ["hi"], None):
        try:
            if lang:
                fetched = ytt.fetch(video_id, languages=lang)
            else:
                # Let the library pick any available language
                transcript_list = ytt.list(video_id)
                transcript = transcript_list.find_generated_transcript(
                    ["en", "hi", "en-US", "en-GB"]
                )
                fetched = transcript.fetch()

            # FetchedTranscript is iterable; each item is a FetchedTranscriptSnippet
            text = " ".join(snippet["text"] for snippet in fetched)
            if text.strip():
                return text
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            continue
        except Exception:
            continue

    raise ValueError(
        "No transcript could be found for this video in English or Hindi. "
        "Try a video that has captions enabled."
    )


def download_youtube_audio(url: str):
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "Whisper fallback needs `yt-dlp` installed: pip install yt-dlp"
        ) from exc

    temp_dir = tempfile.mkdtemp(prefix="yt_audio_", dir=UPLOAD_FOLDER)
    output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    return downloaded_path, temp_dir


def transcribe_youtube_with_whisper(url: str) -> str:
    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "FFmpeg is required for audio transcription but was not found.\n"
            "Install it with:  sudo apt install ffmpeg"
        )

    temp_dir = None
    try:
        audio_path, temp_dir = download_youtube_audio(url)
        asr_pipeline = load_asr_pipeline()
        result = asr_pipeline(audio_path)
        text = result["text"].strip()
    except Exception as exc:
        raise ValueError(
            f"Could not transcribe this video with Whisper. Details: {exc}"
        ) from exc
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    if not text:
        raise ValueError("Whisper ran but transcribed no speech from this video.")

    return text


def get_youtube_text(url: str) -> str:
    video_id = extract_youtube_video_id(url)

    # ── Step 1: Try transcript API (v1.x) ───────────────────────────────────
    try:
        text = _fetch_transcript_v1(video_id)
        add_to_vectorstore(text, url)
        return text
    except ValueError as transcript_err:
        pass  # fall through to Whisper

    # ── Step 2: Whisper fallback ─────────────────────────────────────────────
    try:
        text = transcribe_youtube_with_whisper(url)
        add_to_vectorstore(text, url)
        return text
    except ValueError as whisper_err:
        raise ValueError(
            f"Could not get transcript via API or Whisper.\n\n"
            f"Transcript API: No captions available.\n"
            f"Whisper: {whisper_err}"
        ) from whisper_err


# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("📤 Upload Knowledge Base")

file = st.sidebar.file_uploader("Upload File", type=["pdf", "txt", "docx"])
if file:
    with st.spinner("Processing..."):
        st.session_state.vector_store = None
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)

        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            extract_text_from_pdf(file)
        elif ext == ".txt":
            extract_text_from_txt(file)
        elif ext == ".docx":
            extract_text_from_docx(file)

        st.sidebar.success("✅ File processed!")

image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image:
    path = os.path.join(UPLOAD_FOLDER, image.name)
    with open(path, "wb") as f:
        f.write(image.getbuffer())
    st.session_state.last_image_path = path
    st.sidebar.success("🖼️ Image ready!")

url = st.sidebar.text_input("Webpage URL")
if st.sidebar.button("Add URL") and url:
    with st.spinner("Fetching webpage..."):
        extract_text_from_url(url)
        st.sidebar.success("🌐 Webpage added!")

yt = st.sidebar.text_input("YouTube Link")
if st.sidebar.button("Fetch Transcript") and yt:
    with st.spinner("Fetching transcript..."):
        try:
            get_youtube_text(yt)
            st.sidebar.success("🎬 Transcript added!")
        except ValueError as exc:
            st.sidebar.error(str(exc))

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# --------------------------
# Chat Display
# --------------------------

st.write("## 🤖 Chat")

for msg in st.session_state.messages:
    align = "flex-end" if msg["sender"] == "user" else "flex-start"
    bg = "#007BFF" if msg["sender"] == "user" else "#FFFFFF"
    color = "white" if msg["sender"] == "user" else "black"

    st.markdown(f"""
        <div style="display:flex; justify-content:{align}; margin:6px 0;">
            <div style="background:{bg}; color:{color}; padding:12px 16px;
                        border-radius:16px; max-width:72%; font-size:16px;">
                {msg["text"]}
            </div>
        </div>
    """, unsafe_allow_html=True)


# --------------------------
# Chat Input + Logic
# --------------------------

prompt = st.chat_input("Ask a question...", key="chat_input_main")

if prompt:
    st.session_state.messages.append({"sender": "user", "text": prompt})

    if st.session_state.vector_store is None:
        reply = "Please upload a PDF / TXT / DOCX file or add a URL / YouTube link first."
        st.session_state.messages.append({"sender": "bot", "text": reply})
        st.rerun()

    image_keywords = ["image", "photo", "picture", "see", "detect", "describe the image"]
    use_image = any(kw in prompt.lower() for kw in image_keywords)

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(prompt)
    context = "\n".join(d.page_content for d in docs)

    llm = load_llm()

    if use_image and st.session_state.last_image_path:
        with open(st.session_state.last_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        mime_type = mimetypes.guess_type(st.session_state.last_image_path)[0] or "image/jpeg"

        # Groq vision requires image_url as an object with a "url" key
        # and a vision-capable model
        vision_llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        from langchain_core.messages import HumanMessage
        vision_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt + "\n\nContext:\n" + context,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_b64}",
                    },
                },
            ]
        )
        reply = vision_llm.invoke([vision_message]).content

    else:
        template = PromptTemplate(
            template=(
                "Use ONLY the context to answer.\n"
                "If not found in context, reply 'I don't know'.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\nAnswer:"
            ),
            input_variables=["context", "question"],
        )
        chain = template | llm | StrOutputParser()
        reply = chain.invoke({"context": context, "question": prompt})

    st.session_state.messages.append({"sender": "bot", "text": reply})
    st.rerun()