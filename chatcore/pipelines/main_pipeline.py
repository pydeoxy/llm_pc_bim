from haystack import Pipeline,component
from haystack.dataclasses import ChatMessage, Document
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import BranchJoiner
from typing import List, Dict, Any

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from chatcore.utils.config_loader import load_llm_config
from doc_pipeline import create_doc_pipeline
from ifc_pipeline import create_ifc_pipeline
from pc_pipeline import create_pc_pipeline

from chatcore.utils.prompts import prompt_template_doc,prompt_template_after_websearch

# Websearch in the last

'''
Main Pipeline:
 A[User Query] --> B(Router)
 B -->|IFC Route| C[IFC Pipeline]
 B -->|Seg Route| D[PC Pipeline]
 B -->|Doc Route| E[Doc Pipeline]
 E -->|Unanswered| F[Web Search]

 Document Pipeline
A[Query] --> B(Retriever)
 B --> C(Prompt Builder)
 C --> D(LLM)
 D --> E{Answer Check}
 E -->|Valid| F[Response]
 E -->|Invalid| G[Web Search]

'''


# Promt template missing.
# Tools class or Agents.
# Connections to be modified.

def create_main_pipeline(
    llm: Any,
    doc_pipeline: Pipeline,
    ifc_pipeline: Pipeline,
    pc_pipeline: Pipeline,
    web_search: Any
) -> Pipeline:
    """
    Creates and configures the main processing pipeline with routing logic
    
    Args:
        llm: HuggingFaceLocalGenerator or other LLMs
        ifc_pipeline: Configured ifc file pipeline
        pc_pipeline:  Configured point cloud pipeline
        doc_pipeline: Configured document pipeline
        web_search: Initialized web search component
                
    Returns:
        Configured Pipeline instance
    """

    prompt_builder_query = PromptBuilder(template=prompt_template_doc)
    prompt_joiner  = BranchJoiner(str)
    prompt_builder_after_websearch = PromptBuilder(template=prompt_template_after_websearch)


    # Define routing conditions
    query_conditions = [         
        {
            "condition": "{{'ifc' in query|lower}}", 
            "output": "{{query}}",
            "output_name": "go_to_ifcpipeline",
            "output_type": str,
        },
        {
            "condition": "{{'point cloud' in query|lower}}",
            "output": "{{query}}",
            "output_name": "go_to_pcpipeline",
            "output_type": str,
        },     
        {
            "condition": "{{'ifc' not in query|lower}}"
                        "and {{'point cloud' not in query|lower}}",
            "output": "{{query}}",
            "output_name": "go_to_docpipeline",
            "output_type": str,
        },         
    ]

    # Initialize the ConditionalRouter
    query_router = ConditionalRouter(query_conditions , unsafe=True)       

    reply_routes = [
        {
            "condition": "{{'no_answer' in replies[0]}}",
            "output": "{{messages[0].text}}",
            "output_name": "go_to_websearch",
            "output_type": str,
        },
        {
            "condition": "{{'No function' in pipe_message.text}}",
            "output": "{{pipe_message.text}}",
            "output_name": "go_to_websearch",
            "output_type": str,
        },
        {
            "condition": "{{'no_answer' not in replies[0]}}",
            "output": "{{documents}}",
            "output_name": "documents",
            "output_type": List[Document],
        },
        {
            "condition": "{{'No function' not in pipe_message.text}}",
            "output": "{{pipe_message}}",
            "output_name": "answer",
            "output_type": ChatMessage,
        },
    ]

    reply_router = ConditionalRouter(reply_routes)

    @component
    class IfcPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(pipe_message=ChatMessage)
        def run(self, query:str):
            return  {"pipe_message":ifc_pipeline.run({"query": query})}

    ifc_pipe = IfcPipeline()

    @component
    class PcPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(pipe_message=ChatMessage)
        def run(self, query:str) -> dict:
            return {"pipe_message":pc_pipeline.run({"query": query})}

    pc_pipe = PcPipeline()

    @component
    class DocPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(answer=str)
        def run(self, query:str) -> dict:
            responce = doc_pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}, "router": {"query": query}})
            return {"answer":responce["router"]["answer"]}
        
    message_joiner = BranchJoiner(ChatMessage)

    doc_pipe = DocPipeline()

    pipeline = Pipeline()
    pipeline.add_component("query_router", query_router)
    pipeline.add_component("ifc_pipe", ifc_pipe) 
    pipeline.add_component("pc_pipe", pc_pipe) 
    pipeline.add_component("doc_pipe", doc_pipe)    
    pipeline.add_component("message_joiner", message_joiner)
    #pipeline.add_component("prompt_builder_query", prompt_builder_query)
    #pipeline.add_component("prompt_joiner", prompt_joiner)
    pipeline.add_component("llm", llm)
    #pipeline.add_component("reply_router", reply_router)
    #pipeline.add_component("web_search", web_search)
    #pipeline.add_component("prompt_builder_after_websearch", prompt_builder_after_websearch)

    # Connect components based on routing
    # Prompt missing
    pipeline.connect("query_router.go_to_ifcpipeline", "ifc_pipe.query") 
    pipeline.connect("query_router.go_to_pcpipeline", "pc_pipe.query") 
    pipeline.connect("query_router.go_to_docpipeline", "doc_pipe.query")
    pipeline.connect("ifc_pipe.pipe_message", "message_joiner") 
    pipeline.connect("pc_pipe.pipe_message", "message_joiner") 
    pipeline.connect("message_joiner", "llm")
    #pipeline.connect("prompt_builder_query", "prompt_joiner")
    #pipeline.connect("prompt_joiner", "llm")

    #pipeline.connect("llm.replies", "reply_router.replies")
    #pipeline.connect("reply_router.go_to_websearch", "web_search.query")
    #pipeline.connect("reply_router.go_to_websearch", "prompt_builder_after_websearch.query")
    #pipeline.connect("web_search.documents", "prompt_builder_after_websearch.documents")
    #pipeline.connect("prompt_builder_after_websearch", "prompt_joiner")    
    
    return pipeline

if __name__ == "__main__":
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from chatcore.tools.doc_processing import DocumentManager
    from chatcore.utils.config_loader import load_llm_config
    from duckduckgo_api_haystack import DuckduckgoApiWebSearch
    from haystack.components.generators import HuggingFaceLocalGenerator
    from doc_pipeline import create_doc_pipeline
    
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

    doc_store = DocumentManager("docs/")
    precessed_docs= doc_store.process_documents()

    doc_pipeline = create_doc_pipeline(
        precessed_docs,    
        llm, 
        web_search=DuckduckgoApiWebSearch(top_k=5)   
        )
    
    ifc_pipeline = create_ifc_pipeline()
    pc_pipeline = create_pc_pipeline()   
    
    main_pipe = create_main_pipeline(
        llm=llm,
        doc_pipeline=doc_pipeline,
        ifc_pipeline=ifc_pipeline,
        pc_pipeline=pc_pipeline,
        web_search=DuckduckgoApiWebSearch(top_k=5)
    )

    # Visualizing the pipeline 
    main_pipe.draw(path="docs/main_pipeline_diagram.png")

    
    #query = "Where is the project smartLab?"
    #query = "Where is the Helsinki?"
    #query = "How many IfcWindow are there in the IFC file?"
    #query = "Visualize the point cloud."
    query = "What is IFC?"
    
    # Run the pipeline
    result = main_pipe.run({"query": query})

    print(result)
    #print(result['message_joiner']["value"]['pipe_message'].text)

   
