from importlib import metadata
from operator import index
import os, faiss, pickle, numpy as np
from sys import meta_path
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src import embedding
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir:str="faiss_store", embedding_model:str = "all-MiniLM-L6-v2", chunk_size:int =1000, chunk_overlap:int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedded model : {embedding_model}")

    def build_from_docs(self, documents:List[Any]):
        print(f"Building vectorstore from {len(documents)} raw documents")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embedding= emb_pipe.embed_chunks(chunks)
        metadata = [{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embedding).astype("float32"),metadata)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings:np.ndarray, metadata:List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadata:
            self.metadata.extend(metadata)
        print(f"[INFO] Added {embeddings.shape[0]} vector to FaissIndex")

    def save(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir,"metadata.pkl")
        faiss.write_index(self.index,faiss_path)
        with open(meta_path,"wb") as f:
            pickle.dump(self.metadata,f)
        print(f"[INFO] Saved faiss index and metadata to {self.persist_dir}")


    def load(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir,"metadata.pkl")
        self.index= faiss.read_index
        with open(meta_path,"rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded faiss index and metadata from {self.persist_dir}")

    
    def search(self,query_embedding:np.ndarray, top_k:int=5):
        D, I = self.index.search(query_embedding,top_k)
        results = []
        for idx,dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx<len(self.metadata) else None
            results.append({"index":idx,"distance":dist,"metadata":meta})
        return results

    
    def query(self, query_text:str, top_k:int=5):
        print(f"[INFO] querying vector store for {query_text}")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)


