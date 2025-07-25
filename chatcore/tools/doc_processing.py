import os
from pathlib import Path
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters.docx import DOCXToDocument
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.joiners.document_joiner import DocumentJoiner

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.config_loader import config

# transformer model from yaml config file
class DocumentManager:
    def __init__(self, folder_path: str = "docs"):
        self.folder_path = Path(folder_path)
        self.document_store = InMemoryDocumentStore()
        self.embedder = SentenceTransformersDocumentEmbedder(
            model=config["embedding"]
        )
        self.embedder.warm_up()
        self.splitter = DocumentSplitter(
            split_by="sentence",
            split_length=150,
            split_overlap=30
        )

        self.splitter.warm_up()
        
        if not self.folder_path.exists():
            raise ValueError(f"Document folder not found: {self.folder_path}")

    def load_documents(self):
        """Load documents from local folder"""
        documents = []
        for file_path in self.folder_path.glob("**/*"):
            if file_path.suffix.lower() == ".txt":
                converter = TextFileToDocument()
                results = converter.run(sources=[Path(file_path)])
                docs_txt = results["documents"]
                for dt in docs_txt:
                    documents.append(dt)                
            elif file_path.suffix.lower() == ".docx":
                converter = DOCXToDocument()
                results = converter.run(sources=[file_path])
                docs_docx = results["documents"]
                for dd in docs_docx:
                    documents.append(dd)
            elif file_path.suffix.lower() == ".pdf":                
                converter = PyPDFToDocument()
                results = converter.run(sources=[file_path])
                docs_pdf = results["documents"]
                for dp in docs_pdf:
                    documents.append(dp)
        
        joiner = DocumentJoiner(join_mode="merge")
        joined = joiner.run(documents=[documents])
                
        return joined

    def process_documents(self):
        """Process and store documents with embeddings"""
        raw_documents = self.load_documents()["documents"]
        split_documents = self.splitter.run(documents=raw_documents)["documents"]
        embedded_documents = self.embedder.run(documents=split_documents)["documents"]
        self.document_store.write_documents(embedded_documents)
        return self.document_store

    def retrieve_documents(self, query: str, top_k: int = 5):
        """Retrieve relevant documents"""
        return self.document_store.search(query, top_k=top_k)
    
if __name__ == '__main__':
    from pprint import pprint
    doc_store = DocumentManager("docs/")
    docs = doc_store.load_documents()
    precessed_docs = doc_store.process_documents()

    print(precessed_docs)
    #print(len(docs))
