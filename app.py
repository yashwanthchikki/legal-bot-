import streamlit as st
import openai
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Set up OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Load FAISS index and metadata
def load_faiss_index():
    index = faiss.read_index('legal_faiss_index')
    metadata = pd.read_csv('legal_metadata.csv')
    return index, metadata

# Convert user's question into embedding and retrieve relevant legal document(s)
def retrieve_documents(question, faiss_index, metadata):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode([question])

    # Search in FAISS index
    D, I = faiss_index.search(question_embedding, k=5)  # Retrieve top 5 results

    # Fetch document names
    retrieved_docs = [metadata['file_name'].iloc[i] for i in I[0]]
    return retrieved_docs

# Query OpenAI GPT with the retrieved documents
def query_openai_with_context(question, retrieved_docs):
    # Concatenate the most relevant legal texts to provide context to GPT
    legal_context = "\n\n".join(retrieved_docs)

    # Use OpenAI's LLM to generate an answer
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Based on the following legal texts, answer the question: {question}\n\n{legal_context}",
        max_tokens=200,
        temperature=0.3
    )
    return response['choices'][0]['text']

# Streamlit Interface
def main():
    st.title("Legal Query AI")
    st.write("Ask questions about legal issues, and get responses based on the Indian Constitution, IPC, and other legal documents.")

    question = st.text_input("Enter your legal question:")
    if st.button("Get Answer"):
        if question:
            # Load FAISS index and metadata
            faiss_index, metadata = load_faiss_index()

            # Retrieve the most relevant documents
            retrieved_docs = retrieve_documents(question, faiss_index, metadata)

            # Get AI-based response
            answer = query_openai_with_context(question, retrieved_docs)
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
