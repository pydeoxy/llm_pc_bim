from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import ConditionalRouter
from typing import Dict, Any

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.tools.doc_processing import DocumentManager

'''
Figure out how to deal with normal questions besides the two routes

'''

def create_doc_pipeline(
    document_store: Any,
    llm: Any,
    web_search: Any
    ) -> Pipeline:
    """
    Creates and configures the document processing pipeline
    
    Args:
        document_manager: Initialized document store
        llm: Configured LLM generator
        web_search: Initialized web search component
        
    Returns:
        Configured Pipeline instance
    """
    # Prompt template for document-based queries
    prompt_template = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Answer the following query given the documents.
    If the answer is not contained within the documents reply with 'no_answer'.
    If the answer is contained within the documents, start the answer with "FROM THE KNOWLEDGE BASE: ".

    Documents:
    {% for document in documents %}
    {{document.content}}
    {% endfor %}

    Query: {{query}}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

    prompt_builder = PromptBuilder(template=prompt_template)

    pipeline = Pipeline()

    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store, top_k=5)
    prompt_joiner  = BranchJoiner(str)

    routes = [
        {
            "condition": "{{'no_answer' in replies[0]}}",
            "output": "{{query}}",
            "output_name": "go_to_websearch",
            "output_type": str,
        },
        {
            "condition": "{{'no_answer' not in replies[0]}}",
            "output": "{{replies[0]}}",
            "output_name": "answer",
            "output_type": str,
        },
    ]

    router = ConditionalRouter(routes)

    prompt_template_after_websearch = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Answer the following query given the documents retrieved from the web.
    Start the answer with "FROM THE WEB: ".

    Documents:
    {% for document in documents %}
    {{document.content}}
    {% endfor %}

    Query: {{query}}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

    prompt_builder_after_websearch = PromptBuilder(template=prompt_template_after_websearch)
        
    # Add components
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("prompt_joiner", prompt_joiner)
    pipeline.add_component("llm", llm)
    pipeline.add_component("router", router)
    pipeline.add_component("web_search", web_search)
    pipeline.add_component("prompt_builder_after_websearch", prompt_builder_after_websearch)

    # Connect components
    pipeline.connect("text_embedder", "retriever")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "prompt_joiner")
    pipeline.connect("prompt_joiner", "llm")
    pipeline.connect("llm.replies", "router.replies")
    pipeline.connect("router.go_to_websearch", "web_search.query")
    pipeline.connect("router.go_to_websearch", "prompt_builder_after_websearch.query")
    pipeline.connect("web_search.documents", "prompt_builder_after_websearch.documents")
    pipeline.connect("prompt_builder_after_websearch", "prompt_joiner")

    return pipeline

if __name__ == "__main__":
    from chatcore.utils.config_loader import load_llm_config
    from duckduckgo_api_haystack import DuckduckgoApiWebSearch
    
    llm_config = load_llm_config()
    
    llm = HuggingFaceLocalGenerator(
        model=llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": llm_config["device_map"],
            "torch_dtype": llm_config["torch_dtype"],        
            #"model_kwargs": {"use_auth_token": llm_config["huggingface"]["use_auth_token"]}
        },
        generation_kwargs=llm_config["generation"]
    )

    llm.warm_up()

    doc_store = DocumentManager("docs/")
    precessed_docs= doc_store.process_documents()

    doc_pipe = create_doc_pipeline(
        precessed_docs,
        llm,
        web_search=DuckduckgoApiWebSearch(top_k=5)
        )
    
    # Visualizing the pipeline 
    # doc_pipe.draw(path="docs/doc_pipeline_diagram.png")

    def get_answer(query):
        result = doc_pipe.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}, "router": {"query": query}})
        print(result["router"]["answer"])
    
    query = "Where is smartLab?"

    get_answer(query)
