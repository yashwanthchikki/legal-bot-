import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load and process legal documents
def load_legal_docs(folder_path):
    legal_texts = []
    file_names = []
    
    # Iterate over all text files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                legal_texts.append(file.read())
                file_names.append(file_name)
    return legal_texts, file_names

# Step 2: Convert text to embeddings
def embed_texts(legal_texts):
    # Using SentenceTransformer to convert text into embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and effective model for embeddings
    embeddings = model.encode(legal_texts)
    return embeddings

# Step 3: Store embeddings in FAISS
def store_embeddings(embeddings, file_names):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # FAISS index using L2 distance (Euclidean)

    index.add(embeddings)  # Add all embeddings to the index

    # Save FAISS index
    faiss.write_index(index, 'legal_faiss_index')
    
    # Save file names and their corresponding embeddings
    df = pd.DataFrame({'file_name': file_names})
    df.to_csv('legal_metadata.csv', index=False)

if __name__ == '__main__':
    # Step 4: Run the embedding process
    folder_path = 'legal_docs'  # Folder where all legal text files are stored
    legal_texts, file_names = load_legal_docs(folder_path)
    
    embeddings = embed_texts(legal_texts)
    store_embeddings(embeddings, file_names)
    print("Embeddings stored successfully!")
