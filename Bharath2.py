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

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


import streamlit as st

# Load secrets from Streamlit Cloud
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]


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
    /* Background Styling */
    body {
        background-color: #1f242d; /* Dark theme */
        color: white;
    }

    /* Main Chat Container */
    .main {
        background: rgba(50, 57, 70, 0.9); /* Semi-transparent background */
        padding: 20px;
        border-radius: 10px;
    }

    /* Title Styling */
    h1 {
        color: #0ef; /* Cyan */
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }

    /* Chat Message Styling (For Both User & Assistant) */
    .chat-message {
        font-size: 1.2rem;
        line-height: 1.5;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
        background: linear-gradient(135deg, #1E3A8A, #4A90E2); /* Same gradient for all messages */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Chat Input Box */
    .stChatInput {
        border-radius: 10px;
        border: 2px solid #0ef; /* Cyan border */
        font-size: 1.1rem;
    }

    /* Buttons Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0ef, #1E3A8A);
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: bold;
        font-size: 1rem;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #1E3A8A, #0ef);
        transform: scale(1.05);
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            f"<div class='chat-message'>{message['content']}</div>", unsafe_allow_html=True
        )

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
