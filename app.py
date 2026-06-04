import streamlit as st
import PyPDF2

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.chains import RetrievalQA

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Personal Financial Analyzer (RAG)",
    page_icon="💰",
    layout="wide",
)

# =====================================================
# API KEY
# =====================================================

GOOGLE_API_KEY = "AQ.Ab8RN6I8s0JZaOM4EhvPGeVgyStkPzjiSosvtGk6m-DwgbyZAw"

# =====================================================
# STYLING
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #eef5ff;
    }

    h1 {
        text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 AI Personal Financial Analyzer (RAG)")

st.write(
    """
Upload your UPI transaction PDF and generate financial insights
using Retrieval-Augmented Generation (RAG).
"""
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📌 Instructions")

    st.markdown(
        """
1. Upload a PDF statement

2. Extract transaction text

3. Create embeddings

4. Build FAISS vector database

5. Generate financial insights

6. Ask questions about your transactions

### Example Questions

- How much did I spend on food?
- Show shopping expenses.
- What is my biggest spending category?
- Which month had the highest expenses?
"""
    )

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text(pdf_file):

    text = ""

    try:
        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"PDF Error: {e}")

    return text

# =====================================================
# VECTOR STORE
# =====================================================

@st.cache_resource
def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )

    return vector_store

# =====================================================
# QA CHAIN
# =====================================================

@st.cache_resource
def create_qa_chain(_vector_store):

    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

# =====================================================
# REPORT QUERY
# =====================================================

REPORT_QUERY = """
Analyze the uploaded UPI transaction history.

Generate a detailed report with:

1. Monthly Summary
   - Total Transactions
   - Total Income
   - Total Expenses
   - Net Savings
   - Savings Rate

Use markdown tables.

2. Spending Breakdown
   - Food
   - Travel
   - Shopping
   - Utilities
   - Others

If Others is largest,
break it into subcategories.

3. Hidden Patterns

Include:
- Frequent Small Spends
- Impulse Purchases
- Peak Spending Days
- Unusual Spending Trends

4. Smart Suggestions

Include:
- Budget Strategy
- Saving Improvements
- Expense Reduction Ideas

5. One motivational quote.

Format everything cleanly using markdown.
"""

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📄 Upload UPI Transaction PDF",
    type=["pdf"],
)

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file:

    st.success("PDF uploaded successfully.")

    with st.spinner("Extracting PDF text..."):
        pdf_text = extract_text(uploaded_file)

    if not pdf_text.strip():
        st.error("No text extracted from PDF.")
        st.stop()

    with st.spinner("Creating embeddings and vector database..."):
        vector_store = create_vector_store(pdf_text)

    with st.spinner("Building Retrieval-Augmented Generation pipeline..."):
        qa_chain = create_qa_chain(vector_store)

    st.divider()

    # ==========================================
    # REPORT GENERATION
    # ==========================================

    st.subheader("📊 Generate Financial Report")

    if st.button("Generate Report"):

        with st.spinner("Analyzing transaction history..."):

            result = qa_chain.invoke(
                {"query": REPORT_QUERY}
            )

        st.markdown(result["result"])

    st.divider()

    # ==========================================
    # RAG CHATBOT
    # ==========================================

    st.subheader("💬 Ask Questions About Transactions")

    user_question = st.text_input(
        "Ask a question"
    )

    if user_question:

        with st.spinner("Searching relevant transactions..."):

            answer = qa_chain.invoke(
                {"query": user_question}
            )

        st.markdown("### Answer")
        st.write(answer["result"])

        with st.expander("Retrieved Context Chunks"):

            for idx, doc in enumerate(
                answer["source_documents"],
                start=1
            ):
                st.markdown(f"**Chunk {idx}**")
                st.write(doc.page_content[:1200])

    st.success("✅ RAG system ready.")
