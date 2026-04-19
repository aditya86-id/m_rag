# Multimodal RAG Chatbot

A Streamlit-based multimodal RAG chatbot that can ingest documents, webpages, YouTube transcripts, and images, then answer questions using retrieved context and Groq-hosted LLMs.

## Features

- Upload `PDF`, `TXT`, and `DOCX` files
- Add knowledge from webpage URLs
- Fetch YouTube transcripts
- Upload an image and ask image-aware questions
- Store embeddings in a local `FAISS` vector index
- Chat with retrieved context using Groq models

## Tech Stack

- Streamlit
- LangChain
- FAISS
- Hugging Face sentence-transformer embeddings
- Groq via `langchain-groq`
- `youtube-transcript-api`

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── runtime.txt
├── .env
├── uploads/
└── faiss_index/
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
HF_ASR_MODEL=openai/whisper-small
USER_AGENT=multimodal-rag-chatbot/1.0
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Select your repo and branch.
4. Set the main file path to:

```text
app.py
```

5. Add these secrets in Streamlit Cloud:

```toml
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL="llama-3.3-70b-versatile"
HF_ASR_MODEL="openai/whisper-small"
USER_AGENT="multimodal-rag-chatbot/1.0"
```

## Notes

- `uploads/` and `faiss_index/` are generated at runtime.
- Streamlit Cloud storage is ephemeral, so uploaded files and saved indexes may be lost on restart or redeploy.
- YouTube Whisper fallback may require `ffmpeg` to be available.
- Rotate your Groq API key before publishing if a real key has already been stored in `.env`.

## License

Add your preferred license here.
