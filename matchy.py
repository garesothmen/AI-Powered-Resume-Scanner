import streamlit as st
from pdf_utils import extract_text_from_pdf,extract_zip,rank_cvs_against_job,generate_feedback
from qa_engine import create_vectorstore, get_answer
import os
results=""
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
    selected_cv = st.selectbox("Sélectionner un CV pour un feedback personnalisé", [os.path.basename(f[0]).split('/')[-1] for f in results])
    if selected_cv:
        selected_text = next(text for path, text in zip(cv_files, cv_texts) if os.path.basename(path) == selected_cv)
        selected_score = next(score for path, score in results if os.path.basename(path) == selected_cv)

        if st.button("Générer un feedback pour améliorer ce CV"):
            with st.spinner("Génération du feedback..."):
                feedback = generate_feedback(selected_text, job_description, selected_score)
                st.subheader("📝 Recommandations personnalisées :")
                st.write(feedback)
        