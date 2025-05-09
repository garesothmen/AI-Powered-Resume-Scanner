# import fitz  # PyMuPDF

# def extract_text_from_pdf(uploaded_file):
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text
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

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

import zipfile
import os

def extract_zip(uploaded_zip, extract_to="cv_folder"):
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to+'/Resume', f) for f in os.listdir(extract_to+'/Resume') if f.endswith('.pdf')]


import fitz

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
