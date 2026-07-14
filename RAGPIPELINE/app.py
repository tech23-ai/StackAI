from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    #docs = load_all_documents("data")
    #store = FaissVectorStore("faiss_store")
    #store.build_from_docs(docs)
    #print(store.query("How to classify long horizon task?", top_k=3))
    rag_search = RAGSearch()
    query = "How to classify long horizon tasks?"
    summary= rag_search.search_and_summarize(query,top_k=3)
    print("Summary:", summary)