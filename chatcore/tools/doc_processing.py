import os
from pathlib import Path
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore

class DocumentManager:
    def __init__(self, folder_path: str = "docs"):
        self.folder_path = Path(folder_path)
        self.document_store = InMemoryDocumentStore()
        self.embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.splitter = DocumentSplitter(
            split_by="sentence",
            split_length=150,
            split_overlap=30
        )
        
        if not self.folder_path.exists():
            raise ValueError(f"Document folder not found: {self.folder_path}")

    def load_documents(self):
        """Load documents from local folder"""
        documents = []
        for file_path in self.folder_path.glob("**/*"):
            if file_path.suffix.lower() in [".txt", ".pdf", ".docx"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(content=content, meta={"source": str(file_path)})
                    )
        return documents

    def process_documents(self):
        """Process and store documents with embeddings"""
        raw_documents = self.load_documents()
        split_documents = self.splitter.run(documents=raw_documents)["documents"]
        embedded_documents = self.embedder.run(documents=split_documents)["documents"]
        self.document_store.write_documents(embedded_documents)

    def retrieve_documents(self, query: str, top_k: int = 5):
        """Retrieve relevant documents"""
        return self.document_store.search(query, top_k=top_k)