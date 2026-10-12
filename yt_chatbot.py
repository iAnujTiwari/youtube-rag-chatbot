import os
import re
import sys
import time

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

console = Console()

# ---------------- API KEY SETUP ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    sys.exit("GOOGLE_API_KEY missing. Create a .env file (see .env.example).")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

INDEX_DIR = "indexes_gemini" 

def test_key():
    if GOOGLE_API_KEY.startswith("paste-your"):
        sys.exit("Put your real API key in GOOGLE_API_KEY first.")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY)
        llm.invoke("Say hi")
        print("API key works.")
    except Exception as e:
        sys.exit(f"API key test failed: {e}")


def extract_video_id(url_or_id):
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    raise ValueError("Could not find a video ID")


def get_transcript(video_id):
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    return " ".join(s.text for s in fetched)


def build_vector_store(video_id, embeddings):
    path = os.path.join(INDEX_DIR, video_id)
    if os.path.isdir(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    transcript = get_transcript(video_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    print(f"{len(chunks)} chunks. Embedding in small batches (free tier is rate limited)...")

  
    batch_size = 20
    vs = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vs is None:
            vs = FAISS.from_documents(batch, embeddings)
        else:
            vs.add_documents(batch)
        print(f"  embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(15)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(path)
    return vs


def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = PromptTemplate(
        template="""
You are a friendly assistant that answers questions about a YouTube video using its transcript.

Rules:
- Use ONLY the transcript context below. If the answer isn't there, say "I couldn't find that in the video."
- Don't say "based on the transcript" or "the provided context". Just answer naturally.
- Start with a one-sentence direct answer.
- Then give the key points as short bullet points (maximum 5).
- Use simple language and keep it concise.
- Use markdown: **bold** for important names and numbers.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    return parallel | prompt | llm | StrOutputParser()


def main():
    if len(sys.argv) >= 2:
        url = sys.argv[1]
    else:
        url = input("Paste YouTube link: ").strip()
    test_key()

    try:
        video_id = extract_video_id(url)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        vs = build_vector_store(video_id, embeddings)
    except (TranscriptsDisabled, NoTranscriptFound):
        sys.exit("No English captions for this video.")
    except ValueError as e:
        sys.exit(str(e))

    chain = build_chain(vs)
    print("Ready. Type 'exit' to quit.")
    while True:
        q = input("ask your Q: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer = chain.invoke(q)
            console.print("\n[bold green]Chatbot:[/bold green]")
            console.print(Markdown(answer))
            console.print()
            time.sleep(7)

if __name__ == "__main__":
    main()



    





        





