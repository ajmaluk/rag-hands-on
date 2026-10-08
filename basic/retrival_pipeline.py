from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory,
)

query = "When GTX 970 Memory Issue Fixed"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.3
    }
)

relavent_docs = retriever.invoke(query)

print(query)

print(relavent_docs)
