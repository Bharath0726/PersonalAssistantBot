import streamlit as st
from streamlit_chat import message  # Ensure streamlit-chat is installed
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Constants
DEFAULT_PDF_PATH = "Docs/about.pdf"  # Update with your PDF path
VECTOR_STORE_FILENAME = "faiss_index"

# Initialize API Keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM and Embeddings
model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192",
    temperature=0.3
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Functions
def get_pdf_text(pdf_path):
    """Extract text from PDF."""
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_and_save_vector_store(text_chunks):
    """Create and save a FAISS vector store."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_FILENAME)
    return vector_store

def load_vector_store():
    """Load the FAISS vector store."""
    return FAISS.load_local(VECTOR_STORE_FILENAME, embeddings, allow_dangerous_deserialization=True)

# Initialize Vector Store
vector_store = None
if os.path.exists(VECTOR_STORE_FILENAME):
    vector_store = load_vector_store()
else:
    raw_text = get_pdf_text(DEFAULT_PDF_PATH)
    text_chunks = get_text_chunks(raw_text)
    vector_store = create_and_save_vector_store(text_chunks)

# Create QA Chain with Instructions
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",  # Chain type for simple retrieval-based answering
    verbose=True
)

# Add instructions to the chatbot
qa_chain.combine_documents_chain.llm_chain.prompt = (
    qa_chain.combine_documents_chain.llm_chain.prompt.copy(update={
        "template": """
Instructions:
- Answer based strictly on the provided context
- If the answer isn't in the context, respond with: "I don't have this information in my knowledge base, but I'll be happy to check with Bharath and get back to you on this!"
- Provide a clear, concise, and structured response
- If asked "Who are you?", respond "I am 'Jarvis,' Bharath's personal assistant chatbot"
- Maintain a friendly and helpful tone

Context:
{{context}}

Question:
{{question}}

Response:
"""
    })
)

# Streamlit App UI Enhancements
st.set_page_config(page_title="Bharath's Assistant", page_icon="🤖", layout="wide")

# CSS for background image and styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1522120691812-dcdfb625f397?q=80&w=2187&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        color: white;
    }
    .main {
        background: rgba(0, 0, 0, 0.5); /* Transparent dark background for content */
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #FFD700; /* Golden title text */
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .chat-message {
        color: white;
        font-size: 1.2rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.title("💬🤖 Meet Jarvis: Bharath's Personal Assistant")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, Jarvis here! Bharath's personal assistant chatbot. How can I assist you today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='chat-message'>{message['content']}</div>", unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message'>{prompt}</div>", unsafe_allow_html=True)

    # Generate response
    response = qa_chain.run(prompt)

    # Save response in history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat-message'>{response}</div>", unsafe_allow_html=True)
