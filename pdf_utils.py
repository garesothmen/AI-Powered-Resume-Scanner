from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import faiss
import zipfile
import fitz
import requests

def chunk_text(text, chunk_size=500):
    paragraphs = text.split("\n")
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk) + len(para) < chunk_size:
            chunk += para + " "
        else:
            chunks.append(chunk.strip())
            chunk = para + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks


model = SentenceTransformer('google/flan-t5-small')

def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings



tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_most_relevant_chunks(query, chunks, embeddings, index, top_k=3):
    query_embedding = model.encode([query])
    _, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

def answer_question(query, chunks, embeddings, index):
    relevant_chunks = get_most_relevant_chunks(query, chunks, embeddings, index)
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)



def extract_zip(uploaded_zip, extract_to="cv_folder"):
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to+'/Resume', f) for f in os.listdir(extract_to+'/Resume') if f.endswith('.pdf')]



def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])



embedding_model = SentenceTransformer("google/flan-t5-small")

def embed_documents(doc_texts):
    if not doc_texts:
        return torch.empty(0)
    return embedding_model.encode(doc_texts, convert_to_tensor=True)

def rank_cvs_against_job(cv_texts, cv_paths, job_description):
    if not cv_texts or not cv_paths or not job_description:
        raise ValueError("cv_texts, cv_paths, and job_description must not be empty.")

    job_embedding = embedding_model.encode([job_description], convert_to_tensor=True)
    cv_embeddings = embed_documents(cv_texts)

    if cv_embeddings.shape[0] == 0:
        raise ValueError("CV embeddings are empty. Check that cv_texts are valid.")

    job_embedding_np = job_embedding.cpu().numpy().reshape(1, -1)
    cv_embeddings_np = cv_embeddings.cpu().numpy()

    scores = cosine_similarity(job_embedding_np, cv_embeddings_np)[0]
    ranked = sorted(zip(cv_paths, scores), key=lambda x: x[1], reverse=True)
    return ranked
### This  bloc replaces API calls by downloading the model locally
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# token=hugg_key
# tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=token)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",            # met "cpu" si tu n’as pas de GPU
#     torch_dtype="auto",           # ou torch.float16 avec GPU
#     use_auth_token=token
# )

# # Utiliser une pipeline pour simplifier
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# def generate_feedback(cv_text, job_description, score):
#     prompt = f"""
# Tu es un expert en ressources humaines.

# Voici la description de poste :
# \"\"\"{job_description}\"\"\"

# Voici un CV, avec une pertinence actuelle de {score:.2f} :
# \"\"\"{cv_text}\"\"\"

# Ta tâche : Identifie les manques, incohérences ou parties faibles du CV par rapport à l'offre. 
# Donne des conseils concrets pour l’améliorer afin de viser une pertinence de 0.90 ou plus.
# Réponds sous forme d’une liste claire et professionnelle.
# """

#     output = generator(prompt, max_new_tokens=512, temperature=0.7, do_sample=True)[0]["generated_text"]
#     return output.replace(prompt, "").strip()






API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

def generate_feedback(cv_text, job_description, score,hf_token):
    headers = {
    "Authorization": f"Bearer {hf_token}"
}
    prompt = f"""
You are a human resources expert.I am provinding you with a job description and a resume.

here is the Job Description :
\"\"\"{job_description}\"\"\"

here is a resume, with an actual pertinence of {score:.2f} :
\"\"\"{cv_text}\"\"\"

Your task: Identify gaps, inconsistencies, or weak points in your resume relative to the job offer.
Provide concrete advice on how to improve it to achieve a relevance score of 0.90 or higher.
Respond in a clear and professional manner.
"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.7}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"].replace(prompt, "").strip()
    else:
        return f"Erreur {response.status_code} : {response.text}"
