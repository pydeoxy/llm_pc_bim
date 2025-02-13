from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from typing import Dict, Any

from haystack.components.embedders import SentenceTransformersTextEmbedder
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


def create_doc_pipeline(
    document_store: Any,
    llm: HuggingFaceLocalGenerator,
    web_search: Any
) -> Pipeline:
    """
    Creates and configures the document processing pipeline
    
    Args:
        document_store: Initialized document store
        llm: Configured LLM generator
        web_search: Initialized web search component
        
    Returns:
        Configured Pipeline instance
    """
    # Prompt template for document-based queries
    prompt_template = """
    Answer the question using the context below. 
    If you don't know the answer, just say you don't know.
    
    Context:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    
    Question: {{ query }}
    
    Answer:
    """

    pipeline = Pipeline()
    
    # Add components
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=5))
    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component("llm", llm)
    pipeline.add_component("web_search", web_search)

    # Connect components
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    
    # Add decision node for unanswered questions
    def route_unanswered(result: Dict[str, Any]):
        if not result["llm"]["replies"][0] or "don't know" in result["llm"]["replies"][0].lower():
            return {"unanswered": result["query"]}
        return {"answered": result["llm"]["replies"][0]}

    pipeline.add_component("answer_router", component=route_unanswered)
    pipeline.connect("llm", "answer_router")
    pipeline.connect("answer_router.unanswered", "web_search.query")

    return pipeline