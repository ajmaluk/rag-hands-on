import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_docs(docs_path="docs"):
    print("Loading all documents")
    if not os.path.exists(docs_path):
        raise FileNotFoundError("The Directory Docs is not Found")
    
    txt_loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    md_loader = DirectoryLoader(
        path=docs_path,
        glob="*.md",
        loader_cls=TextLoader
    )

    documents = txt_loader.load() + md_loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No txt or md file found in docs")

    for i, doc in enumerate(documents[:2]):
        print(f"\n\nDocument {i+1}")
        print(f"Source {doc.metadata['source']}")
        print(f"Content Lenght {len(doc.page_content)}")
        print(f"Content Preview : \n{doc.page_content[:100]}...")
        print(f"Metadata {doc.metadata}")

    return documents


def split_docs(documents, chunk_size=100, chunk_overlap=0):
    print("Splitting Documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chucks in enumerate(chunks[:5]):
            print(f"\nChunk {i+1}")
            print(f"Source {chucks.metadata['source']}")
            print(f"Content Lenght {len(chucks.page_content)}")
            print(f"Content Preview : \n{chucks.page_content[:100]}...")
            print(f"Metadata {chucks.metadata}")
            print("-"*50)
        return chunks


def create_vector_and_store(chunks, persist_directory="db/chroma_db"):
    print("Creating and Storing Vector Embeddings")

    #embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings(        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print(f"Vectors create and saved to {persist_directory}")
    return vector_store

def main():
    #load documents
    documents = load_docs("docs")

    #chunking files
    chucks = split_docs(documents)


    #create vectors and store
    vector_store = create_vector_and_store(chucks)



if __name__ == "__main__":
    main()