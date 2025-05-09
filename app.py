import streamlit as st
from pdf_utils import extract_text_from_pdf
from qa_engine import create_vectorstore, get_answer

st.set_page_config(page_title="PDF Q&A Bot", layout="centered")
st.title("📄 Ask Questions About Your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf") 

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        vectorstore = create_vectorstore(raw_text)
        st.success("PDF loaded and processed!")

    question = st.text_input("Enter your question:")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = get_answer(vectorstore, question)
            st.markdown("### 💬 Answer:")
            st.write(answer)