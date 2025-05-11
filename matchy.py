import streamlit as st
from pdf_utils import extract_text_from_pdf,extract_zip,rank_cvs_against_job,generate_feedback
#from qa_engine import create_vectorstore, get_answer
import os
results=""
st.title("🔍 Match CVs with Job openings")


# Section Token utilisateur
st.sidebar.title("🔐 Authentification")
hf_token = st.sidebar.text_input(
    "Hugging Face Token (obligatory)", 
    type="password",
    help="login to hugging face then follow Model url : https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 /n finally Create Token here: https://huggingface.co/settings/tokens "
)
mode = st.sidebar.selectbox("Choose your role:", ["Recruiter", "Candidate"])
if not hf_token:
    st.warning("❗ Entrez un token Hugging Face pour utiliser l'application.")
    st.stop()
if mode == "Recruiter":
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
        selected_cv = st.selectbox("Select a CV to get personalized feedback", [os.path.basename(f[0]).split('/')[-1] for f in results])
        if selected_cv:
            selected_text = next(text for path, text in zip(cv_files, cv_texts) if os.path.basename(path) == selected_cv)
            selected_score = next(score for path, score in results if os.path.basename(path) == selected_cv)

            if st.button("Generate feedback to improve this CV"):
                with st.spinner("Generating feedback..."):
                    feedback = generate_feedback(selected_text, job_description, selected_score,hf_token)
                    st.subheader("📝 Recommandations :")
                    st.write(feedback)
elif mode == "Candidate":
    st.header("Candidate Assistant 🤖")

    # 1. Upload CV
    uploaded_cv = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
    if uploaded_cv and 'cv_text' not in st.session_state:
        file_contents = uploaded_cv.read()
        with open("temp_resume.pdf", "wb") as f:
            f.write(file_contents)
        st.session_state['cv_text'] = [extract_text_from_pdf("temp_resume.pdf")]
        st.success("CV uploaded successfully.")

    # 2. Input Job Descriptions
    if 'cv_text' in st.session_state:
        st.subheader("Paste Job Descriptions:")
        job_texts = []
        for i in range(3):
            job_input = st.text_area(f"Job Description {i+1}", key=f"job_{i}")
            if job_input.strip():
                job_texts.append(job_input)

        if job_texts and st.button("Match with Jobs"):
            job_scores = []
            for job in job_texts:
                score = rank_cvs_against_job(st.session_state['cv_text'], ["temp_resume.pdf"], job)
                job_scores.append((job, score))

            st.session_state['sorted_jobs'] = sorted(job_scores, key=lambda x: x[1], reverse=True)

    # 3. Show results and improve CV
    if 'sorted_jobs' in st.session_state:
        st.subheader("Job Matching Results")
        sorted_jobs = st.session_state['sorted_jobs']
        for idx, (job, score) in enumerate(sorted_jobs):
            st.markdown(f"**Job {idx+1}** - Match: `{score[0][1]}`")
            with st.expander("View Job Description"):
                st.write(job)

        selected_value = st.selectbox("Select a job to get recommendations:", range(1, len(sorted_jobs)+1))
        selected_job_idx = selected_value - 1

        if st.button("Improve My CV for Selected Job"):
            selected_job = sorted_jobs[selected_job_idx][0]
            selected_score = sorted_jobs[selected_job_idx][1][0][1]
            feedback = generate_feedback(st.session_state['cv_text'][0], selected_job, selected_score, hf_token)
            st.subheader("📌 Recommended Improvements:")
            st.write(feedback)