💼 Matchy — AI-Powered CV and Job Description Matcher
Matchy is a smart assistant that helps both recruiters and candidates analyze and align CVs with job descriptions using open-source language models — no OpenAI API required.


🚀 Features
👤 Candidate Mode
Upload one CV (PDF)

Enter multiple job descriptions

Get match scores for each job

Get tailored CV improvement recommendations

👥 Recruiter Mode
Upload a folder of CVs (PDFs)

Provide one job description

Get a ranked list of matching candidates

Select a CV to get custom improvement suggestions

🛠️ Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/matchy-app.git
cd matchy-app
2. Create a Virtual Environment
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
🤖 Model Access (Hugging Face)
4. Create a Hugging Face Account
https://huggingface.co/join

5. Generate an Inference Token
Go to https://huggingface.co/settings/tokens

Click New token → name it → select Read access

Copy the token

▶️ Run the App
streamlit run matchy.py

The interface will launch in your browser

Paste your Hugging Face token in the box when asked

Select your user role (Recruiter or Candidate)

Start uploading files and viewing results!
