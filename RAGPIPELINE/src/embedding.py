from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import embedding
from transformers.masking_utils import chunked_overlay
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name:str="all-MiniLM-L6-V2",chunk_size:int = 1000, chunk_overlap:int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded Embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any])-> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n","\n"," ",""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks. ")
        return chunks

    def embed_chunks(self, chunks:List[Any])-> np.ndarray:
        textx = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddng for {len (textx)} chunks ..." )
        embeddings = self.model.encode(textx,show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings