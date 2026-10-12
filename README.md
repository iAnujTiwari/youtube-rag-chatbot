# YouTube Video Chatbot (RAG)

Ask questions about any YouTube video using its transcript. Answers include timestamps and clickable source links.

## Features
- Question answering over the video transcript (RAG with FAISS + Gemini)
- Timestamps and source links in answers
- Whole-video summaries
- Follow-up questions with chat memory

## Setup
1. Clone the repo and install packages: `pip install -r requirements.txt`
2. Get a free API key at https://aistudio.google.com/apikey
3. Copy `.env.example` to `.env` and paste your key
4. Run: `python yt_chatbot.py "https://www.youtube.com/watch?v=VIDEO_ID"`

## How it works
Transcript -> chunks (with timestamps) -> Gemini embeddings -> FAISS index -> retrieve relevant chunks -> Gemini writes the answer.