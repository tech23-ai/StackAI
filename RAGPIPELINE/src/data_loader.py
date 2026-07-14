# Created to load data from different sources

from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir:str)->List[Any]:
    """
    Load all supported documents from the directory and convert to LangChain document structure
    Supported : PDF, TXT, CSV, EXCEL, WORD, JSON
    """

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path : {data_path}")
    documents = []
    pdf_files = list (data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files : {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading pdf files: {pdf_file}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")

        except Exception as e:
            print(f"[ERROR] failed to load PDF files {pdf_file}:{e}")

    return documents