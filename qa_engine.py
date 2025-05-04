import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

def create_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    return FAISS.from_texts(chunks, embeddings)

def get_answer(vectorstore, query):
    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vectorstore.similarity_search(query)
    return chain.run(input_documents=docs, question=query)