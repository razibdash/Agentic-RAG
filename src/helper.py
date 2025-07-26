from langchain.embeddings import HuggingFaceEmbeddings



# the function to download Hugging Face embeddings
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings