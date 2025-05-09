import streamlit as st
from pdf_utils import extract_text_from_pdf,extract_zip,rank_cvs_against_job
from qa_engine import create_vectorstore, get_answer
import os

st.title("🔍 Match CVs with Job openings")

uploaded_zip = st.file_uploader("Upload a .zip Folder containing list of CVs", type="zip")
job_description = st.text_area("Paste your Job description here")

if uploaded_zip and job_description:
    with st.spinner("Analysing CVs..."):
        cv_files = extract_zip(uploaded_zip)
        #st.write(f"**{cv_files}**")
        cv_texts = [extract_text_from_pdf(f) for f in cv_files]
        results = rank_cvs_against_job(cv_texts, cv_files, job_description)

        st.success("Finishied Classement ✅")
        st.subheader("📋 Top CVs Matching :")
        # st.write(f"**{results}**")
        for path, score in results[:5]:
            st.write(f"**{os.path.basename(path).split('/')[-1]}** — Pertinence: {score:.2f}")