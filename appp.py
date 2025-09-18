import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss

# -------------------
# Initialize SentenceTransformer ON CPU
# -------------------
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # small & fast

# -------------------
# Initialize HuggingFace QA pipeline with SMALL model
# -------------------
qa_pipeline = pipeline(
    "question-answering",
    model="sshleifer/tiny-distilbert-base-cased-distilled-squad",  # lightweight
    device=-1  # CPU
)

# -------------------
# Step 1: Scrape Website
# -------------------
def scrape_elitebody():
    url = "https://elitebodyhome.com/"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        text_blocks = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "h2", "h3"])]
        return [t for t in text_blocks if t]
    except Exception:
        return []

DOCUMENTS = scrape_elitebody()
if not DOCUMENTS:
    DOCUMENTS = [
        "Elite Body Home offers facials, weight loss, pain management.",
        "You can book appointments online or by phone.",
        "We offer IV drip, massage, and skin treatments in Dubai."
    ]
print("Number of scraped documents:", len(DOCUMENTS))

# -------------------
# Step 2: Embeddings + FAISS Index
# -------------------
doc_embeddings = embedder.encode(DOCUMENTS, convert_to_numpy=True, show_progress_bar=False)

if doc_embeddings.size == 0:
    raise ValueError("No document embeddings found!")

doc_embeddings = np.array(doc_embeddings).astype("float32")
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -------------------
# Step 3: Local QA retrieval + answer
# -------------------
def ai_response(message):
    # Embed query
    query_vec = embedder.encode([message], convert_to_numpy=True).astype("float32")
    if len(query_vec.shape) == 1:
        query_vec = np.expand_dims(query_vec, axis=0)

    # Retrieve top 3 docs
    D, I = index.search(query_vec, k=3)
    retrieved_docs = [DOCUMENTS[i] for i in I[0]]

    # Combine retrieved docs as context for QA
    context = " ".join(retrieved_docs)

    # Get answer from local QA pipeline
    try:
        result = qa_pipeline(question=message, context=context)
        answer = result["answer"]
    except Exception:
        answer = "Sorry, I couldn't find an answer to that."

    return answer, retrieved_docs

# -------------------
# Step 4: Streamlit UI
# -------------------
st.set_page_config(page_title="Elite Body Home", layout="centered")
st.title("Elite Body Home Clinic")
st.markdown("---")
st.header("AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document" not in st.session_state:
    st.session_state.document = []

user_input = st.text_input("Ask about facials, services, booking, or contact:", "")
if st.button("Send") and user_input:
    resp, relevant = ai_response(user_input)
    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Bot", resp))
    st.session_state.document.extend(relevant)

st.subheader("Chat History")
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")

st.markdown("---")
st.header("Dynamic Service Document")
if st.session_state.document:
    for entry in set(st.session_state.document):
        st.write("- " + entry)
else:
    st.write("Ask questions to see this section update!")

st.markdown("---")
st.caption("Elite Body Home - Aesthetic & Wellness | Dubai")
