import os
from sys import meta_path
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

class RAGSearch:
    def __init__(self, persist_dir:str="faiss-store",embedding_model:str="all-MiniLM-L6-V2", llm_model:str="qwen/qwen3-32b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, embedding_model)
        meta_path=os.path.join(persist_dir,"metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vectorstore.build_from_docs(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq initialized :{llm_model}")

    def search_and_summarize(self, query:str, top_k:int=5)->str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts=[r["metadata"].get("text","") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant docs found."
        prompt = f""" Summarize the following context for the query: '{query}' \n\n Context: \n{context} \n\n Summary:"""
        response = self.llm.invoke([prompt])
        return response.content
        

