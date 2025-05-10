import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def create_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Local embedding model (no OpenAI key needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def get_answer(vectorstore, query):
    # Local text generation model
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")  # use device=0 if you have a GPU
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vectorstore.similarity_search(query)
    return chain.run(input_documents=docs, question=query)
