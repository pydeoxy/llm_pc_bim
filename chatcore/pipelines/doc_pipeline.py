from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from typing import Dict, Any

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.tools.doc_processing import DocumentManager
from chatcore.utils.config_loader import config

def create_doc_pipeline(
    document_store: Any,
    ) -> Pipeline:
    """
    Creates and configures the document processing pipeline
    
    Args:
        document_store: Initialized document store
        
    Returns:
        Configured Pipeline instance
    """
    
    pipeline = Pipeline()
    
    text_embedder = SentenceTransformersTextEmbedder(model=config["embedding"])
    retriever = InMemoryEmbeddingRetriever(document_store, top_k=5)       
        
    # Add components
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)

    # Connect components
    pipeline.connect("text_embedder", "retriever")


    return pipeline

if __name__ == "__main__":
    

    doc_store = DocumentManager("docs/")
    precessed_docs= doc_store.process_documents()

    doc_pipe = create_doc_pipeline(
        precessed_docs,        
        )
    
    # Visualizing the pipeline 
    #doc_pipe.draw(path="docs/doc_pipeline_diagram.png")
    
    # Testing Q&A
    #query = "What is the capital of Finland?"
    query = "Who are involved in the project SmartLab?"
    #query = "What is ifc schema?"

    result = doc_pipe.run({"text_embedder": {"text": query}})
    print(result)
